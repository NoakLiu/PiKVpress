#!/usr/bin/env python3
"""
MoE路由器Press使用示例

展示如何在kvpress中使用MoE路由器进行KV缓存压缩
"""

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from kvpress.presses import MoERouterPress
from kvpress import pipeline

def main():
    # 1. 加载模型和tokenizer
    model_name = "meta-llama/Llama-2-7b-hf"  # 或其他支持的模型
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 2. 创建MoE路由器Press
    moe_press = MoERouterPress(
        num_experts=4,           # 4个专家
        top_k=2,                 # 每个token路由到2个专家
        capacity_factor=1.5,     # 容量因子
        dropout=0.1,             # dropout率
        router_type="pikv",      # 使用PiKV路由器
        cache_aware=True,        # 启用缓存感知
        compression_ratio=0.5,   # 目标压缩比
        aux_loss_weight=0.01     # 辅助损失权重
    )
    
    # 3. 准备输入文本
    text = """
    The quick brown fox jumps over the lazy dog. This is a sample text for testing 
    the MoE router press in kvpress. The MoE router uses multiple experts to decide 
    how to compress the KV cache based on the input characteristics and cache usage patterns.
    """ * 10  # 重复文本以增加长度
    
    # 4. 使用MoE路由器进行推理
    print("使用MoE路由器进行推理...")
    
    with moe_press(model):
        # 编码输入
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # 生成文本
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 解码输出
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"生成的文本: {generated_text}")
    
    # 5. 获取MoE路由器统计信息
    stats = moe_press.get_stats()
    print("\n=== MoE路由器统计信息 ===")
    print(f"总辅助损失: {stats['total_aux_loss']:.4f}")
    print(f"平均辅助损失: {stats['avg_aux_loss']:.4f}")
    print(f"前向传播次数: {stats['forward_count']}")
    
    # 打印每层的统计信息
    for layer_idx, layer_stats in stats['layer_stats'].items():
        print(f"\n--- 第{layer_idx}层 ---")
        
        # 路由器统计
        router_stats = layer_stats['router_stats']
        print(f"专家使用比例: {router_stats['expert_usage_ratios'].tolist()}")
        print(f"专家使用计数: {router_stats['expert_usage_count'].tolist()}")
        print(f"总token数: {router_stats['total_tokens'].item()}")
        
        # 压缩统计
        compression_stats = layer_stats['expert_compression_stats']
        print(f"专家使用情况: {compression_stats['expert_usage'].tolist()}")
        print(f"压缩比例: {compression_stats['compression_ratios'].tolist()}")
        print(f"缓存命中率: {compression_stats['cache_hit_rates'].tolist()}")
    
    # 6. 重置统计信息
    moe_press.reset_stats()
    print("\n统计信息已重置")

def compare_with_baseline():
    """比较MoE路由器与基线方法的性能"""
    model_name = "meta-llama/Llama-2-7b-hf"
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    text = "The quick brown fox jumps over the lazy dog. " * 50
    
    # 基线：不使用压缩
    print("=== 基线测试（无压缩）===")
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    import time
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    baseline_time = time.time() - start_time
    print(f"基线推理时间: {baseline_time:.4f}秒")
    
    # MoE路由器测试
    print("\n=== MoE路由器测试 ===")
    moe_press = MoERouterPress(
        num_experts=4,
        top_k=2,
        router_type="pikv",
        cache_aware=True
    )
    
    start_time = time.time()
    
    with moe_press(model):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
    
    moe_time = time.time() - start_time
    print(f"MoE路由器推理时间: {moe_time:.4f}秒")
    print(f"加速比: {baseline_time / moe_time:.2f}x")
    
    # 获取压缩统计
    stats = moe_press.get_stats()
    print(f"平均辅助损失: {stats['avg_aux_loss']:.4f}")

def advanced_usage_example():
    """高级使用示例：自定义专家策略"""
    model_name = "meta-llama/Llama-2-7b-hf"
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 创建自定义MoE路由器
    class CustomMoERouterPress(MoERouterPress):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            # 自定义专家策略
            self.expert_strategies = {
                0: "ultra_aggressive",  # 超激进压缩
                1: "semantic_aware",    # 语义感知压缩
                2: "temporal_aware",    # 时序感知压缩
                3: "adaptive"           # 自适应压缩
            }
        
        def _apply_expert_compression(
            self, 
            keys: torch.Tensor, 
            values: torch.Tensor, 
            strategy: str,
            router_probs: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """自定义压缩策略"""
            batch_size, num_heads, seq_len, head_dim = keys.shape
            
            if strategy == "ultra_aggressive":
                # 只保留前10%和后5%
                keep_front = max(1, int(seq_len * 0.1))
                keep_back = max(1, int(seq_len * 0.05))
                if seq_len > keep_front + keep_back:
                    keys = torch.cat([keys[:, :, :keep_front], keys[:, :, -keep_back:]], dim=2)
                    values = torch.cat([values[:, :, :keep_front], values[:, :, -keep_back:]], dim=2)
                    
            elif strategy == "semantic_aware":
                # 基于语义重要性选择位置
                # 使用注意力权重的方差作为重要性指标
                attention_variance = torch.var(router_probs, dim=-1)  # [batch_size, seq_len]
                importance_scores = attention_variance.mean(dim=0)  # [seq_len]
                
                num_keep = max(1, int(seq_len * 0.4))
                _, important_indices = torch.topk(importance_scores, k=num_keep, dim=-1)
                important_indices = torch.sort(important_indices)[0]
                
                keys = keys[:, :, important_indices, :]
                values = values[:, :, important_indices, :]
                
            elif strategy == "temporal_aware":
                # 时序感知：保留开始、中间和结束部分
                keep_start = max(1, int(seq_len * 0.2))
                keep_middle = max(1, int(seq_len * 0.1))
                keep_end = max(1, int(seq_len * 0.2))
                
                if seq_len > keep_start + keep_middle + keep_end:
                    middle_start = (seq_len - keep_middle) // 2
                    middle_end = middle_start + keep_middle
                    
                    keys = torch.cat([
                        keys[:, :, :keep_start],
                        keys[:, :, middle_start:middle_end],
                        keys[:, :, -keep_end:]
                    ], dim=2)
                    values = torch.cat([
                        values[:, :, :keep_start],
                        values[:, :, middle_start:middle_end],
                        values[:, :, -keep_end:]
                    ], dim=2)
                    
            elif strategy == "adaptive":
                # 自适应：根据序列长度动态调整
                if seq_len < 100:
                    # 短序列：保守压缩
                    keep_ratio = 0.8
                elif seq_len < 500:
                    # 中等序列：中等压缩
                    keep_ratio = 0.6
                else:
                    # 长序列：激进压缩
                    keep_ratio = 0.4
                
                num_keep = max(1, int(seq_len * keep_ratio))
                keys = keys[:, :, :num_keep, :]
                values = values[:, :, :num_keep, :]
            
            return keys, values
    
    # 使用自定义MoE路由器
    custom_moe_press = CustomMoERouterPress(
        num_experts=4,
        top_k=2,
        router_type="pikv",
        cache_aware=True
    )
    
    text = "This is a test of the custom MoE router with advanced compression strategies. " * 30
    
    with custom_moe_press(model):
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"自定义MoE路由器生成的文本: {generated_text}")
    
    # 获取自定义统计
    stats = custom_moe_press.get_stats()
    print(f"自定义MoE路由器平均辅助损失: {stats['avg_aux_loss']:.4f}")

if __name__ == "__main__":
    print("=== MoE路由器Press示例 ===\n")
    
    try:
        main()
        print("\n" + "="*50 + "\n")
        compare_with_baseline()
        print("\n" + "="*50 + "\n")
        advanced_usage_example()
    except Exception as e:
        print(f"运行示例时出错: {e}")
        print("请确保已安装所需的依赖包") 