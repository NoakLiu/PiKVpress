# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Union
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from kvpress.presses.base_press import BasePress
from transformers import (
    LlamaForCausalLM,
    MistralForCausalLM,
    Phi3ForCausalLM,
    Qwen2ForCausalLM,
    Qwen3ForCausalLM,
    Gemma3ForCausalLM,
    GPT2LMHeadModel,
    PreTrainedModel,
)

logger = logging.getLogger(__name__)

# 支持的模型类型
SUPPORTED_MODELS = (
    LlamaForCausalLM,
    MistralForCausalLM,
    Phi3ForCausalLM,
    Qwen2ForCausalLM,
    Qwen3ForCausalLM,
    Gemma3ForCausalLM,
    GPT2LMHeadModel,  # 添加GPT2支持
)

class BaseMoERouter(nn.Module):
    """
    基础MoE路由器类，为KV缓存压缩提供路由逻辑
    """
    def __init__(
        self, 
        hidden_size: int, 
        num_experts: int, 
        top_k: int = 2, 
        capacity_factor: float = 1.5,
        dropout: float = 0.0
    ):
        super(BaseMoERouter, self).__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.capacity_factor = capacity_factor
        self.dropout = dropout
        
        # 路由网络
        self.router = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_experts)
        )
        
        # 路由统计信息
        self.register_buffer('total_tokens', torch.tensor(0.0))
        self.register_buffer('expert_usage_count', torch.zeros(num_experts))
        self.register_buffer('routing_decisions', torch.zeros(num_experts))
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化路由器权重"""
        for module in self.router:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _compute_capacity(self, batch_size: int, seq_len: int) -> int:
        """计算每个专家的容量"""
        total_tokens = batch_size * seq_len
        return int(total_tokens * self.capacity_factor * self.top_k / self.num_experts)
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        expert_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        路由输入到专家
        
        Args:
            hidden_states: 输入张量 [batch_size, seq_len, hidden_size]
            expert_mask: 专家可用性掩码 [num_experts]
            
        Returns:
            dispatch_tensor: 调度张量
            combine_tensor: 组合张量
            router_probs: 路由概率
            aux_loss: 辅助损失
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # 计算路由逻辑（分数）
        router_logits = self.router(hidden_states)  # [batch_size, seq_len, num_experts]
        
        # 应用专家掩码
        if expert_mask is not None:
            mask_value = torch.finfo(router_logits.dtype).min
            router_logits = router_logits + (1 - expert_mask) * mask_value
            
        # 计算路由概率
        router_probs = F.softmax(router_logits, dim=-1)  # [batch_size, seq_len, num_experts]
        
        # 获取top_k专家
        top_k_probs, top_k_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
        
        # 重新归一化top_k概率
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)
        
        # 创建调度和组合张量
        capacity = self._compute_capacity(batch_size, seq_len)
        
        # 初始化张量
        dispatch_tensor = torch.zeros(
            batch_size, seq_len, self.num_experts, capacity,
            device=top_k_indices.device, dtype=top_k_probs.dtype
        )
        combine_tensor = torch.zeros(
            batch_size, seq_len, self.num_experts, capacity,
            device=top_k_indices.device, dtype=top_k_probs.dtype
        )
        
        # 跟踪每个专家的当前容量使用
        expert_capacity_used = torch.zeros(
            self.num_experts, device=top_k_indices.device, dtype=torch.long
        )
        
        # 填充调度和组合张量
        for b in range(batch_size):
            for s in range(seq_len):
                for k in range(self.top_k):
                    expert_idx = top_k_indices[b, s, k].item()
                    prob = top_k_probs[b, s, k].item()
                    
                    # 检查专家容量
                    if expert_capacity_used[expert_idx] < capacity:
                        pos = expert_capacity_used[expert_idx].item()
                        dispatch_tensor[b, s, expert_idx, pos] = 1.0
                        combine_tensor[b, s, expert_idx, pos] = prob
                        expert_capacity_used[expert_idx] += 1
        
        # 计算辅助损失（负载平衡）
        aux_loss = self._compute_load_balancing_loss(router_probs, top_k_indices)
        
        # 更新统计信息
        with torch.no_grad():
            self.total_tokens += batch_size * seq_len
            # 更新专家使用计数
            for expert_idx in range(self.num_experts):
                expert_count = (top_k_indices == expert_idx).sum().float()
                self.expert_usage_count[expert_idx] += expert_count
                self.routing_decisions[expert_idx] += expert_count
        
        return dispatch_tensor, combine_tensor, router_probs, aux_loss
    
    def _compute_load_balancing_loss(
        self, 
        router_probs: torch.Tensor, 
        expert_indices: torch.Tensor
    ) -> torch.Tensor:
        """计算负载平衡损失"""
        # 计算每个专家的使用率
        router_prob_per_expert = router_probs.mean(dim=[0, 1])  # [num_experts]
        
        # 计算专家分配的实际比例
        expert_mask = F.one_hot(expert_indices, num_classes=self.num_experts).float()
        expert_usage_rate = expert_mask.mean(dim=[0, 1, 2])  # [num_experts]
        
        # 负载平衡损失：期望使用率与实际使用率的差异
        # 使用平方差损失鼓励均匀分布
        balance_loss = torch.sum(router_prob_per_expert * expert_usage_rate)
        
        return balance_loss * self.num_experts
    
    def get_routing_stats(self) -> Dict[str, torch.Tensor]:
        """获取路由统计信息"""
        if self.total_tokens > 0:
            expert_usage_ratios = self.expert_usage_count / self.total_tokens
        else:
            expert_usage_ratios = torch.zeros_like(self.expert_usage_count)
        
        return {
            "expert_usage_ratios": expert_usage_ratios,
            "expert_usage_count": self.expert_usage_count,
            "total_tokens": self.total_tokens,
            "routing_decisions": self.routing_decisions
        }
    
    def reset_stats(self):
        """重置路由统计信息"""
        self.total_tokens.zero_()
        self.expert_usage_count.zero_()
        self.routing_decisions.zero_()


class PiKVMoERouter(BaseMoERouter):
    """
    PiKV专用MoE路由器
    结合KV缓存使用情况进行路由决策
    """
    def __init__(
        self, 
        hidden_size: int, 
        num_experts: int, 
        top_k: int = 2,
        capacity_factor: float = 1.5,
        dropout: float = 0.0,
        cache_aware: bool = True,
        cache_update_interval: int = 100
    ):
        super(PiKVMoERouter, self).__init__(
            hidden_size, num_experts, top_k, capacity_factor, dropout
        )
        self.cache_aware = cache_aware
        self.cache_update_interval = cache_update_interval
        
        # 缓存使用情况跟踪
        self.register_buffer('cache_usage_history', torch.zeros(num_experts, 100))
        self.register_buffer('cache_hit_rates', torch.zeros(num_experts))
        self.register_buffer('cache_update_counter', torch.tensor(0))
        
        # 缓存感知路由调整网络
        if cache_aware:
            self.cache_router_adjustment = nn.Sequential(
                nn.Linear(hidden_size + num_experts, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, num_experts),
                nn.Tanh()  # 输出调整因子 [-1, 1]
            )
    
    def update_cache_usage(self, expert_idx: int, cache_hit_rate: float):
        """更新专家的缓存使用情况"""
        if 0 <= expert_idx < self.num_experts:
            # 更新命中率
            self.cache_hit_rates[expert_idx] = cache_hit_rate
            
            # 更新历史记录
            history_idx = self.cache_update_counter.item() % 100
            self.cache_usage_history[expert_idx, history_idx] = cache_hit_rate
            
        self.cache_update_counter += 1
    
    def _compute_cache_aware_adjustment(
        self, 
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor
    ) -> torch.Tensor:
        """计算缓存感知的路由调整"""
        if not self.cache_aware or not hasattr(self, 'cache_router_adjustment'):
            return torch.zeros_like(router_logits)
        
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # 计算输入特征的平均值
        avg_features = hidden_states.mean(dim=1)  # [batch_size, hidden_size]
        
        # 扩展缓存命中率到批次维度
        cache_rates_expanded = self.cache_hit_rates.unsqueeze(0).expand(batch_size, -1)
        
        # 组合特征
        combined_input = torch.cat([avg_features, cache_rates_expanded], dim=-1)
        
        # 计算调整因子
        adjustment_factors = self.cache_router_adjustment(combined_input)  # [batch_size, num_experts]
        
        # 扩展到序列维度
        adjustment_factors = adjustment_factors.unsqueeze(1).expand(-1, seq_len, -1)
        
        return adjustment_factors
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        kv_cache_states: Optional[torch.Tensor] = None,
        expert_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # 计算基础路由逻辑
        router_logits = self.router(hidden_states)
        
        # 应用缓存感知调整
        if self.cache_aware:
            cache_adjustments = self._compute_cache_aware_adjustment(hidden_states, router_logits)
            router_logits = router_logits + cache_adjustments
        
        # 应用专家掩码
        if expert_mask is not None:
            mask_value = torch.finfo(router_logits.dtype).min
            router_logits = router_logits + (1 - expert_mask) * mask_value
        
        # 计算路由概率
        router_probs = F.softmax(router_logits, dim=-1)
        
        # 获取top_k专家
        top_k_probs, top_k_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
        
        # 重新归一化
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)
        
        # 创建调度和组合张量
        capacity = self._compute_capacity(batch_size, seq_len)
        
        # 初始化张量
        dispatch_tensor = torch.zeros(
            batch_size, seq_len, self.num_experts, capacity,
            device=top_k_indices.device, dtype=top_k_probs.dtype
        )
        combine_tensor = torch.zeros(
            batch_size, seq_len, self.num_experts, capacity,
            device=top_k_indices.device, dtype=top_k_probs.dtype
        )
        
        # 跟踪每个专家的当前容量使用
        expert_capacity_used = torch.zeros(
            self.num_experts, device=top_k_indices.device, dtype=torch.long
        )
        
        # 填充调度和组合张量
        for b in range(batch_size):
            for s in range(seq_len):
                for k in range(self.top_k):
                    expert_idx = top_k_indices[b, s, k].item()
                    prob = top_k_probs[b, s, k].item()
                    
                    # 检查专家容量
                    if expert_capacity_used[expert_idx] < capacity:
                        pos = expert_capacity_used[expert_idx].item()
                        dispatch_tensor[b, s, expert_idx, pos] = 1.0
                        combine_tensor[b, s, expert_idx, pos] = prob
                        expert_capacity_used[expert_idx] += 1
        
        # 计算辅助损失
        aux_loss = self._compute_load_balancing_loss(router_probs, top_k_indices)
        
        # 更新统计信息
        with torch.no_grad():
            self.total_tokens += batch_size * seq_len
            for expert_idx in range(self.num_experts):
                expert_count = (top_k_indices == expert_idx).sum().float()
                self.expert_usage_count[expert_idx] += expert_count
        
        return dispatch_tensor, combine_tensor, router_probs, aux_loss


@dataclass
class MoERouterPress(BasePress):
    """
    MoE路由器Press，将MoE路由逻辑集成到KV缓存压缩中
    
    该Press使用MoE路由器来决定如何压缩KV缓存：
    - 根据输入特征路由到不同的专家
    - 每个专家负责不同的压缩策略
    - 支持缓存感知的路由决策
    """
    
    num_experts: int = 4
    top_k: int = 2
    capacity_factor: float = 1.5
    dropout: float = 0.1
    router_type: str = "pikv"  # "base", "pikv"
    cache_aware: bool = True
    compression_ratio: float = 0.5  # 目标压缩比
    aux_loss_weight: float = 0.01
    
    def __post_init__(self):
        self.routers = {}
        self.expert_compression_stats = {}
        self.total_aux_loss = 0.0
        self.forward_count = 0
        
        # 初始化专家压缩策略
        self.expert_strategies = {
            0: "aggressive",    # 激进压缩
            1: "moderate",      # 中等压缩
            2: "conservative",  # 保守压缩
            3: "selective"      # 选择性压缩
        }
    
    def _get_router(self, layer_idx: int, hidden_size: int) -> BaseMoERouter:
        """获取或创建路由器"""
        if layer_idx not in self.routers:
            if self.router_type == "pikv":
                router = PiKVMoERouter(
                    hidden_size=hidden_size,
                    num_experts=self.num_experts,
                    top_k=self.top_k,
                    capacity_factor=self.capacity_factor,
                    dropout=self.dropout,
                    cache_aware=self.cache_aware
                )
            else:
                router = BaseMoERouter(
                    hidden_size=hidden_size,
                    num_experts=self.num_experts,
                    top_k=self.top_k,
                    capacity_factor=self.capacity_factor,
                    dropout=self.dropout
                )
            self.routers[layer_idx] = router
            self.expert_compression_stats[layer_idx] = {
                "expert_usage": torch.zeros(self.num_experts),
                "compression_ratios": torch.zeros(self.num_experts),
                "cache_hit_rates": torch.zeros(self.num_experts)
            }
        
        return self.routers[layer_idx]
    
    def _apply_expert_compression(
        self, 
        keys: torch.Tensor, 
        values: torch.Tensor, 
        strategy: str,
        router_probs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """应用专家特定的压缩策略"""
        batch_size, num_heads, seq_len, head_dim = keys.shape
        
        if strategy == "aggressive":
            # 激进压缩：保留前20%和后10%
            keep_front = max(1, int(seq_len * 0.2))
            keep_back = max(1, int(seq_len * 0.1))
            if seq_len > keep_front + keep_back:
                keys = torch.cat([keys[:, :, :keep_front], keys[:, :, -keep_back:]], dim=2)
                values = torch.cat([values[:, :, :keep_front], values[:, :, -keep_back:]], dim=2)
                
        elif strategy == "moderate":
            # 中等压缩：保留前30%和后20%
            keep_front = max(1, int(seq_len * 0.3))
            keep_back = max(1, int(seq_len * 0.2))
            if seq_len > keep_front + keep_back:
                keys = torch.cat([keys[:, :, :keep_front], keys[:, :, -keep_back:]], dim=2)
                values = torch.cat([values[:, :, :keep_front], values[:, :, -keep_back:]], dim=2)
                
        elif strategy == "conservative":
            # 保守压缩：保留前50%和后30%
            keep_front = max(1, int(seq_len * 0.5))
            keep_back = max(1, int(seq_len * 0.3))
            if seq_len > keep_front + keep_back:
                keys = torch.cat([keys[:, :, :keep_front], keys[:, :, -keep_back:]], dim=2)
                values = torch.cat([values[:, :, :keep_front], values[:, :, -keep_back:]], dim=2)
                
        elif strategy == "selective":
            # 选择性压缩：基于注意力权重选择重要位置
            # 使用路由概率作为重要性指标
            importance_scores = router_probs.mean(dim=0)  # [seq_len, num_experts]
            importance_scores = importance_scores.mean(dim=-1)  # [seq_len]
            
            # 选择重要性最高的位置
            num_keep = max(1, int(seq_len * (1 - self.compression_ratio)))
            _, important_indices = torch.topk(importance_scores, k=num_keep, dim=-1)
            important_indices = torch.sort(important_indices)[0]  # 保持顺序
            
            keys = keys[:, :, important_indices, :]
            values = values[:, :, important_indices, :]
        
        return keys, values
    
    def _compute_cache_hit_rate(self, keys: torch.Tensor, values: torch.Tensor) -> float:
        """计算缓存命中率（简化版本）"""
        # 这里可以实现更复杂的缓存命中率计算
        # 目前使用序列长度作为简单指标
        return min(1.0, keys.shape[2] / 1000.0)  # 假设1000是理想长度
    
    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        使用MoE路由器进行KV缓存压缩
        """
        layer_idx = module.layer_idx
        hidden_size = hidden_states.shape[-1]
        
        # 获取路由器
        router = self._get_router(layer_idx, hidden_size)
        
        # 执行路由
        dispatch_tensor, combine_tensor, router_probs, aux_loss = router(
            hidden_states, expert_mask=None
        )
        
        # 累积辅助损失
        self.total_aux_loss += aux_loss.item()
        self.forward_count += 1
        
        # 获取每个专家的路由概率
        expert_probs = router_probs.mean(dim=[0, 1])  # [num_experts]
        
        # 选择概率最高的专家
        dominant_expert = torch.argmax(expert_probs).item()
        strategy = self.expert_strategies[dominant_expert]
        
        # 应用专家压缩策略
        compressed_keys, compressed_values = self._apply_expert_compression(
            keys, values, strategy, router_probs
        )
        
        # 更新统计信息
        with torch.no_grad():
            # 更新专家使用统计
            self.expert_compression_stats[layer_idx]["expert_usage"][dominant_expert] += 1
            
            # 计算压缩比
            original_size = keys.shape[2]
            compressed_size = compressed_keys.shape[2]
            compression_ratio = (original_size - compressed_size) / original_size
            self.expert_compression_stats[layer_idx]["compression_ratios"][dominant_expert] += compression_ratio
            
            # 更新缓存命中率
            cache_hit_rate = self._compute_cache_hit_rate(compressed_keys, compressed_values)
            self.expert_compression_stats[layer_idx]["cache_hit_rates"][dominant_expert] += cache_hit_rate
            
            # 如果是PiKV路由器，更新缓存使用情况
            if isinstance(router, PiKVMoERouter):
                router.update_cache_usage(dominant_expert, cache_hit_rate)
        
        logger.debug(f"Layer {layer_idx}: Expert {dominant_expert} ({strategy}) "
                    f"compressed {keys.shape[2]} -> {compressed_keys.shape[2]} "
                    f"(ratio: {compression_ratio:.3f})")
        
        return compressed_keys, compressed_values
    
    def get_stats(self) -> Dict[str, Union[float, Dict]]:
        """获取MoE路由器统计信息"""
        stats = {
            "total_aux_loss": self.total_aux_loss,
            "avg_aux_loss": self.total_aux_loss / max(1, self.forward_count),
            "forward_count": self.forward_count,
            "layer_stats": {}
        }
        
        for layer_idx, router in self.routers.items():
            layer_stats = {
                "router_stats": router.get_routing_stats(),
                "expert_compression_stats": self.expert_compression_stats[layer_idx]
            }
            stats["layer_stats"][layer_idx] = layer_stats
        
        return stats
    
    def reset_stats(self):
        """重置统计信息"""
        self.total_aux_loss = 0.0
        self.forward_count = 0
        
        for router in self.routers.values():
            router.reset_stats()
        
        for layer_stats in self.expert_compression_stats.values():
            layer_stats["expert_usage"].zero_()
            layer_stats["compression_ratios"].zero_()
            layer_stats["cache_hit_rates"].zero_()
    
    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        """
        重写forward_hook方法以支持GPT2和其他模型
        """
        # 检测模型类型并提取参数
        if isinstance(module, torch.nn.Module) and hasattr(module, '_model_type'):
            # GPT2模型：参数在input中
            if module._model_type == 'gpt2':
                hidden_states = input[0] if input else None
                # GPT2可能没有past_key_value，需要特殊处理
                if len(input) > 1 and input[1] is not None:
                    cache = input[1]
                else:
                    # 如果没有缓存，直接返回
                    return output
            else:
                # 其他模型：参数在kwargs中
                hidden_states = kwargs.get("hidden_states")
                cache = kwargs.get("past_key_value")
        else:
            # 默认尝试从kwargs获取
            hidden_states = kwargs.get("hidden_states")
            cache = kwargs.get("past_key_value")
        
        # 如果没有hidden_states，尝试从input获取
        if hidden_states is None and input:
            hidden_states = input[0]
        
        # 如果仍然没有hidden_states，直接返回
        if hidden_states is None:
            return output
        
        # 如果没有缓存，直接返回
        if cache is None:
            return output
        
        q_len = hidden_states.shape[1]
        
        # 检查是否需要压缩（简化版本，不检查cache_position）
        # 对于GPT2，我们总是尝试压缩
        
        # 获取缓存
        if hasattr(cache, 'key_cache') and hasattr(cache, 'value_cache'):
            # 标准缓存格式
            keys = cache.key_cache[module.layer_idx]
            values = cache.value_cache[module.layer_idx]
        else:
            # 可能是其他格式，直接返回
            return output
        
        # 执行压缩
        keys, values = self.compress(module, hidden_states, keys, values, output[1] if len(output) > 1 else None, kwargs)
        
        # 更新缓存
        cache.key_cache[module.layer_idx] = keys
        cache.value_cache[module.layer_idx] = values
        
        return output
    
    @contextmanager
    def __call__(self, model: PreTrainedModel):
        """
        应用MoE路由器Press到模型
        
        Args:
            model: 预训练模型
            
        Returns:
            context manager
        """
        if not isinstance(model, SUPPORTED_MODELS):
            logger.warning(f"Model {type(model)} not tested, supported models: {SUPPORTED_MODELS}")
        
        hooks = []
        
        try:
            # 根据模型类型选择不同的层访问方式
            if isinstance(model, GPT2LMHeadModel):
                # GPT2模型结构
                layers = model.transformer.h
                for i, layer in enumerate(layers):
                    layer.layer_idx = i
                    # 标记模型类型
                    layer.attn._model_type = 'gpt2'
                    # 注册到注意力层
                    hooks.append(layer.attn.register_forward_hook(self.forward_hook, with_kwargs=True))
            else:
                # 其他模型（Llama, Mistral等）
                layers = model.model.layers
                for i, layer in enumerate(layers):
                    layer.layer_idx = i
                    # 注册到注意力层
                    hooks.append(layer.self_attn.register_forward_hook(self.forward_hook, with_kwargs=True))
            
            yield
            
        finally:
            # 清理hooks
            for hook in hooks:
                hook.remove() 