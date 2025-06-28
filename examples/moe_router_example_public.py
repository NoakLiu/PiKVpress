#!/usr/bin/env python3
"""
MoE路由器Press使用示例（使用公开模型）

展示如何在kvpress中使用MoE路由器进行KV缓存压缩
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from kvpress.presses import MoERouterPress

def main():
    # 1. 加载公开模型和tokenizer
    model_name = "microsoft/DialoGPT-medium"  # 使用公开的模型
    print(f"正在加载模型: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
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
    """ * 5  # 重复文本以增加长度
    
    # 4. 使用MoE路由器进行推理
    print("使用MoE路由器进行推理...")
    
    with moe_press(model):
        # 编码输入
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # 生成文本
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
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
    model_name = "microsoft/DialoGPT-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    text = "The quick brown fox jumps over the lazy dog. " * 20
    
    # 基线：不使用压缩
    print("=== 基线测试（无压缩）===")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    import time
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
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
                max_new_tokens=30,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
    
    moe_time = time.time() - start_time
    print(f"MoE路由器推理时间: {moe_time:.4f}秒")
    print(f"加速比: {baseline_time / moe_time:.2f}x")
    
    # 获取压缩统计
    stats = moe_press.get_stats()
    print(f"平均辅助损失: {stats['avg_aux_loss']:.4f}")

def test_with_small_model():
    """使用小型模型测试"""
    print("\n=== 小型模型测试 ===")
    
    # 使用更小的模型
    model_name = "distilgpt2"  # 更小的模型
    print(f"正在加载小型模型: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 创建MoE路由器Press
    moe_press = MoERouterPress(
        num_experts=2,  # 减少专家数量
        top_k=1,        # 减少top_k
        router_type="base",  # 使用基础路由器
        cache_aware=False
    )
    
    text = "Hello world, this is a test of the MoE router. " * 10
    
    with moe_press(model):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"生成的文本: {generated_text}")
    
    # 获取统计
    stats = moe_press.get_stats()
    print(f"小型模型测试完成，平均辅助损失: {stats['avg_aux_loss']:.4f}")

if __name__ == "__main__":
    print("=== MoE路由器Press示例（公开模型）===\n")
    
    try:
        main()
        print("\n" + "="*50 + "\n")
        compare_with_baseline()
        print("\n" + "="*50 + "\n")
        test_with_small_model()
    except Exception as e:
        print(f"运行示例时出错: {e}")
        print("请确保已安装所需的依赖包")
        import traceback
        traceback.print_exc() 