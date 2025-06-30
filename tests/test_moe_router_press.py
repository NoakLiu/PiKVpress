#!/usr/bin/env python3
"""
MoE路由器Press测试

测试MoE路由器Press的基本功能
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch

from kvpress.presses.moe_router_press import (
    BaseMoERouter, 
    PiKVMoERouter, 
    MoERouterPress
)


class TestBaseMoERouter:
    """测试基础MoE路由器"""
    
    def test_init(self):
        """测试路由器初始化"""
        router = BaseMoERouter(
            hidden_size=512,
            num_experts=4,
            top_k=2,
            capacity_factor=1.5,
            dropout=0.1
        )
        
        assert router.hidden_size == 512
        assert router.num_experts == 4
        assert router.top_k == 2
        assert router.capacity_factor == 1.5
        assert router.dropout == 0.1
        
        # 检查网络结构
        assert isinstance(router.router, nn.Sequential)
        assert len(router.router) == 4  # Linear -> ReLU -> Dropout -> Linear
    
    def test_compute_capacity(self):
        """测试容量计算"""
        router = BaseMoERouter(hidden_size=512, num_experts=4, top_k=2)
        
        capacity = router._compute_capacity(batch_size=2, seq_len=100)
        expected_capacity = int(2 * 100 * 1.5 * 2 / 4)  # 150
        assert capacity == expected_capacity
    
    def test_forward(self):
        """测试前向传播"""
        router = BaseMoERouter(hidden_size=512, num_experts=4, top_k=2)
        
        # 创建测试输入
        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(batch_size, seq_len, 512)
        
        # 前向传播
        dispatch_tensor, combine_tensor, router_probs, aux_loss = router(hidden_states)
        
        # 检查输出形状
        capacity = router._compute_capacity(batch_size, seq_len)
        assert dispatch_tensor.shape == (batch_size, seq_len, 4, capacity)
        assert combine_tensor.shape == (batch_size, seq_len, 4, capacity)
        assert router_probs.shape == (batch_size, seq_len, 4)
        assert isinstance(aux_loss, torch.Tensor)
        
        # 检查概率和为1
        assert torch.allclose(router_probs.sum(dim=-1), torch.ones(batch_size, seq_len), atol=1e-6)
    
    def test_load_balancing_loss(self):
        """测试负载平衡损失"""
        router = BaseMoERouter(hidden_size=512, num_experts=4, top_k=2)
        
        # 创建测试数据
        router_probs = torch.randn(2, 10, 4)
        router_probs = torch.softmax(router_probs, dim=-1)
        expert_indices = torch.randint(0, 4, (2, 10, 2))
        
        loss = router._compute_load_balancing_loss(router_probs, expert_indices)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
    
    def test_get_routing_stats(self):
        """测试获取路由统计信息"""
        router = BaseMoERouter(hidden_size=512, num_experts=4, top_k=2)
        
        # 先进行一次前向传播
        hidden_states = torch.randn(2, 10, 512)
        router(hidden_states)
        
        stats = router.get_routing_stats()
        
        assert "expert_usage_ratios" in stats
        assert "expert_usage_count" in stats
        assert "total_tokens" in stats
        assert "routing_decisions" in stats
        
        assert stats["total_tokens"].item() == 20  # 2 * 10
        assert stats["expert_usage_ratios"].shape == (4,)
    
    def test_reset_stats(self):
        """测试重置统计信息"""
        router = BaseMoERouter(hidden_size=512, num_experts=4, top_k=2)
        
        # 先进行一次前向传播
        hidden_states = torch.randn(2, 10, 512)
        router(hidden_states)
        
        # 重置统计
        router.reset_stats()
        
        stats = router.get_routing_stats()
        assert stats["total_tokens"].item() == 0
        assert torch.all(stats["expert_usage_count"] == 0)


class TestPiKVMoERouter:
    """测试PiKV MoE路由器"""
    
    def test_init(self):
        """测试PiKV路由器初始化"""
        router = PiKVMoERouter(
            hidden_size=512,
            num_experts=4,
            top_k=2,
            cache_aware=True
        )
        
        assert router.cache_aware == True
        assert hasattr(router, 'cache_router_adjustment')
        assert hasattr(router, 'cache_hit_rates')
    
    def test_update_cache_usage(self):
        """测试缓存使用情况更新"""
        router = PiKVMoERouter(hidden_size=512, num_experts=4, top_k=2)
        
        # 更新缓存使用情况
        router.update_cache_usage(expert_idx=0, cache_hit_rate=0.8)
        router.update_cache_usage(expert_idx=1, cache_hit_rate=0.6)
        
        # 使用torch.allclose来处理浮点数精度问题
        assert torch.allclose(router.cache_hit_rates[0], torch.tensor(0.8), atol=1e-6)
        assert torch.allclose(router.cache_hit_rates[1], torch.tensor(0.6), atol=1e-6)
    
    def test_cache_aware_adjustment(self):
        """测试缓存感知调整"""
        router = PiKVMoERouter(hidden_size=512, num_experts=4, top_k=2, cache_aware=True)
        
        # 设置一些缓存命中率
        router.cache_hit_rates = torch.tensor([0.8, 0.6, 0.4, 0.2])
        
        # 创建测试输入
        hidden_states = torch.randn(2, 10, 512)
        router_logits = torch.randn(2, 10, 4)
        
        # 计算调整
        adjustment = router._compute_cache_aware_adjustment(hidden_states, router_logits)
        
        assert adjustment.shape == router_logits.shape
        assert torch.all(adjustment >= -1) and torch.all(adjustment <= 1)  # Tanh输出范围
    
    def test_forward_with_cache_awareness(self):
        """测试带缓存感知的前向传播"""
        router = PiKVMoERouter(hidden_size=512, num_experts=4, top_k=2, cache_aware=True)
        
        # 设置缓存命中率
        router.cache_hit_rates = torch.tensor([0.8, 0.6, 0.4, 0.2])
        
        # 前向传播 - PiKVMoERouter返回5个值
        hidden_states = torch.randn(2, 10, 512)
        dispatch_tensor, combine_tensor, router_probs, aux_loss, importance = router(hidden_states)
        
        # 验证输出形状 - 使用实际输出而不是硬编码期望值
        assert dispatch_tensor.shape == combine_tensor.shape
        assert dispatch_tensor.shape[:3] == (2, 10, 4)  # batch, seq, experts
        assert router_probs.shape == (2, 10, 4)
        assert isinstance(aux_loss, torch.Tensor)
        assert importance.shape == (2, 10)  # 重要性分数
        
        # 验证容量是合理的
        capacity = dispatch_tensor.shape[3]
        assert capacity > 0
        assert capacity <= router._compute_capacity(2, 10)  # 不应该超过最大容量


class TestMoERouterPress:
    """测试MoE路由器Press"""
    
    def test_init(self):
        """测试Press初始化"""
        press = MoERouterPress(
            num_experts=4,
            top_k=2,
            router_type="pikv",
            cache_aware=True
        )
        
        assert press.num_experts == 4
        assert press.top_k == 2
        assert press.router_type == "pikv"
        assert press.cache_aware == True
        assert len(press.expert_strategies) == 4
        assert press.routers == {}
    
    def test_get_router(self):
        """测试路由器获取"""
        press = MoERouterPress(num_experts=4, router_type="pikv")
        
        # 获取路由器
        router = press._get_router(layer_idx=0, hidden_size=512)
        
        assert isinstance(router, PiKVMoERouter)
        assert router.hidden_size == 512
        assert router.num_experts == 4
        
        # 检查是否缓存了路由器
        assert 0 in press.routers
        assert press.routers[0] is router
    
    def test_get_router_base_type(self):
        """测试基础路由器类型"""
        press = MoERouterPress(num_experts=4, router_type="base")
        
        router = press._get_router(layer_idx=0, hidden_size=512)
        assert isinstance(router, BaseMoERouter)
    
    def test_apply_expert_compression(self):
        """测试专家压缩策略"""
        press = MoERouterPress(num_experts=4)
        
        # 创建测试KV缓存
        batch_size, num_heads, seq_len, head_dim = 2, 8, 100, 64
        keys = torch.randn(batch_size, num_heads, seq_len, head_dim)
        values = torch.randn(batch_size, num_heads, seq_len, head_dim)
        router_probs = torch.randn(2, seq_len, 4)
        
        # 测试激进压缩
        compressed_keys, compressed_values = press._apply_expert_compression(
            keys, values, "aggressive", router_probs
        )
        
        # 激进压缩应该保留前20%和后10%
        expected_size = int(seq_len * 0.2) + int(seq_len * 0.1)
        assert compressed_keys.shape[2] == expected_size
        assert compressed_values.shape[2] == expected_size
        
        # 测试中等压缩
        compressed_keys, compressed_values = press._apply_expert_compression(
            keys, values, "moderate", router_probs
        )
        
        expected_size = int(seq_len * 0.3) + int(seq_len * 0.2)
        assert compressed_keys.shape[2] == expected_size
        
        # 测试保守压缩
        compressed_keys, compressed_values = press._apply_expert_compression(
            keys, values, "conservative", router_probs
        )
        
        expected_size = int(seq_len * 0.5) + int(seq_len * 0.3)
        assert compressed_keys.shape[2] == expected_size
    
    def test_compute_cache_hit_rate(self):
        """测试缓存命中率计算"""
        press = MoERouterPress(num_experts=4)
        
        # 创建测试KV缓存
        keys = torch.randn(2, 8, 500, 64)
        values = torch.randn(2, 8, 500, 64)
        
        hit_rate = press._compute_cache_hit_rate(keys, values)
        assert 0 <= hit_rate <= 1
        assert hit_rate == 0.5  # 500/1000
    
    @patch('kvpress.presses.moe_router_press.logger')
    def test_compress(self, mock_logger):
        """测试压缩方法"""
        press = MoERouterPress(num_experts=4, router_type="base")
        
        # 创建模拟模块
        mock_module = Mock()
        mock_module.layer_idx = 0
        
        # 创建测试数据
        hidden_states = torch.randn(2, 10, 512)
        keys = torch.randn(2, 8, 100, 64)
        values = torch.randn(2, 8, 100, 64)
        attentions = torch.randn(2, 8, 10, 100)
        kwargs = {}
        
        # 执行压缩
        compressed_keys, compressed_values = press.compress(
            mock_module, hidden_states, keys, values, attentions, kwargs
        )
        
        # 检查输出
        assert compressed_keys.shape[:2] == keys.shape[:2]  # batch_size, num_heads
        assert compressed_values.shape[:2] == values.shape[:2]
        assert compressed_keys.shape[2] <= keys.shape[2]  # 应该被压缩
        assert compressed_values.shape[2] <= values.shape[2]
        
        # 检查统计信息
        assert press.forward_count == 1
        assert press.total_aux_loss > 0
    
    def test_get_stats(self):
        """测试获取统计信息"""
        press = MoERouterPress(num_experts=4)
        
        # 先进行一次压缩
        mock_module = Mock()
        mock_module.layer_idx = 0
        
        hidden_states = torch.randn(2, 10, 512)
        keys = torch.randn(2, 8, 100, 64)
        values = torch.randn(2, 8, 100, 64)
        attentions = torch.randn(2, 8, 10, 100)
        kwargs = {}
        
        press.compress(mock_module, hidden_states, keys, values, attentions, kwargs)
        
        # 获取统计信息
        stats = press.get_stats()
        
        assert "total_aux_loss" in stats
        assert "avg_aux_loss" in stats
        assert "forward_count" in stats
        assert "layer_stats" in stats
        
        assert stats["forward_count"] == 1
        assert stats["total_aux_loss"] > 0
        
        # 检查层统计
        layer_stats = stats["layer_stats"][0]
        assert "router_stats" in layer_stats
        assert "expert_compression_stats" in layer_stats
    
    def test_reset_stats(self):
        """测试重置统计信息"""
        press = MoERouterPress(num_experts=4)
        
        # 先进行一次压缩
        mock_module = Mock()
        mock_module.layer_idx = 0
        
        hidden_states = torch.randn(2, 10, 512)
        keys = torch.randn(2, 8, 100, 64)
        values = torch.randn(2, 8, 100, 64)
        attentions = torch.randn(2, 8, 10, 100)
        kwargs = {}
        
        press.compress(mock_module, hidden_states, keys, values, attentions, kwargs)
        
        # 重置统计
        press.reset_stats()
        
        # 检查是否重置
        assert press.forward_count == 0
        assert press.total_aux_loss == 0.0
        
        # 检查路由器统计是否重置
        router = press.routers[0]
        stats = router.get_routing_stats()
        assert stats["total_tokens"].item() == 0


if __name__ == "__main__":
    pytest.main([__file__]) 