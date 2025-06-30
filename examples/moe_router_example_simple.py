#!/usr/bin/env python3
"""
MoE路由器Press简化示例

使用最小的模型进行测试
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from kvpress.presses import MoERouterPress

def test_moe_router_directly():
    """直接测试MoE路由器功能，不依赖模型"""
    print("\n=== 直接测试MoE路由器 ===")
    
    from kvpress.presses.moe_router_press import BaseMoERouter
    
    # 创建路由器
    router = BaseMoERouter(
        hidden_size=768,
        num_experts=2,
        top_k=1
    )
    
    # 创建测试数据
    batch_size, seq_len = 2, 10
    hidden_states = torch.randn(batch_size, seq_len, 768)
    
    print(f"输入形状: {hidden_states.shape}")
    
    # 执行路由
    dispatch_tensor, combine_tensor, router_probs, aux_loss = router(hidden_states)
    
    print(f"✓ 路由成功")
    print(f"  - 调度张量形状: {dispatch_tensor.shape}")
    print(f"  - 组合张量形状: {combine_tensor.shape}")
    print(f"  - 路由概率形状: {router_probs.shape}")
    print(f"  - 辅助损失: {aux_loss.item():.4f}")
    
    # 获取统计信息
    stats = router.get_routing_stats()
    print(f"✓ 路由统计:")
    print(f"  - 专家使用比例: {stats['expert_usage_ratios'].tolist()}")
    print(f"  - 专家使用计数: {stats['expert_usage_count'].tolist()}")
    print(f"  - 总token数: {stats['total_tokens'].item()}")
    
    return True

def test_moe_press_compression():
    """测试MoE Press的压缩功能"""
    print("\n=== 测试MoE Press压缩 ===")
    
    from kvpress.presses import MoERouterPress
    from unittest.mock import Mock
    
    # 创建MoE Press
    moe_press = MoERouterPress(
        num_experts=2,
        router_type="base",
        cache_aware=False
    )
    
    # 创建模拟模块
    mock_module = Mock()
    mock_module.layer_idx = 0
    
    # 创建测试数据
    hidden_states = torch.randn(2, 10, 768)
    keys = torch.randn(2, 12, 20, 64)  # [batch, heads, seq_len, head_dim]
    values = torch.randn(2, 12, 20, 64)
    attentions = torch.randn(2, 12, 10, 20)
    kwargs = {}
    
    print(f"原始KV形状: {keys.shape}")
    
    # 执行压缩
    compressed_keys, compressed_values = moe_press.compress(
        mock_module, hidden_states, keys, values, attentions, kwargs
    )
    
    print(f"✓ 压缩成功")
    print(f"  - 压缩后KV形状: {compressed_keys.shape}")
    print(f"  - 压缩比例: {(keys.shape[2] - compressed_keys.shape[2]) / keys.shape[2]:.2f}")
    
    # 获取统计信息
    stats = moe_press.get_stats()
    print(f"✓ MoE Press统计:")
    print(f"  - 总辅助损失: {stats['total_aux_loss']:.4f}")
    print(f"  - 前向传播次数: {stats['forward_count']}")
    
    if stats['layer_stats']:
        layer_0_stats = stats['layer_stats'][0]
        expert_usage = layer_0_stats['expert_compression_stats']['expert_usage']
        print(f"  - 专家使用情况: {expert_usage.tolist()}")
    
    return True

def main():
    # 使用最小的模型
    model_name = "distilgpt2"  # 只有82M参数
    print(f"正在加载模型: {model_name}")
    
    try:
        # 强制使用safetensors格式
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_safetensors=True,  # 强制使用safetensors
            torch_dtype=torch.float32,  # 使用float32避免精度问题
            low_cpu_mem_usage=True
        )
        
        # 设置pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("✓ 模型加载成功")
        
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("尝试使用快速测试...")
        return
    
    # 创建MoE路由器Press
    moe_press = MoERouterPress(
        num_experts=2,  # 减少专家数量
        top_k=1,        # 减少top_k
        router_type="base",  # 使用基础路由器
        cache_aware=False
    )
    
    print("✓ MoE路由器Press创建成功")
    
    # 准备输入文本
    text = "Hello world, this is a test of the MoE router. " * 5
    
    # 使用MoE路由器进行推理
    print("使用MoE路由器进行推理...")
    
    try:
        with moe_press(model):
            # 编码输入
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            
            # 生成文本
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # 解码输出
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"生成的文本: {generated_text}")
        
        # 获取MoE路由器统计信息
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
        
        print("\n✓ MoE路由器Press测试成功!")
        
    except Exception as e:
        print(f"推理过程中出错: {e}")
        import traceback
        traceback.print_exc()

def test_without_model():
    """不依赖模型的测试"""
    print("\n=== 不依赖模型的测试 ===")
    
    from unittest.mock import Mock
    
    # 创建模拟模块
    mock_module = Mock()
    mock_module.layer_idx = 0
    
    # 创建MoE路由器Press
    moe_press = MoERouterPress(num_experts=2, router_type="base")
    
    # 创建测试数据
    hidden_states = torch.randn(2, 10, 512)
    keys = torch.randn(2, 8, 100, 64)
    values = torch.randn(2, 8, 100, 64)
    attentions = torch.randn(2, 8, 10, 100)
    kwargs = {}
    
    try:
        # 执行压缩
        compressed_keys, compressed_values = moe_press.compress(
            mock_module, hidden_states, keys, values, attentions, kwargs
        )
        
        print(f"✓ 压缩成功")
        print(f"  - 原始KV形状: {keys.shape}")
        print(f"  - 压缩后KV形状: {compressed_keys.shape}")
        print(f"  - 压缩比例: {(keys.shape[2] - compressed_keys.shape[2]) / keys.shape[2]:.2f}")
        
        # 获取详细统计
        stats = moe_press.get_stats()
        print(f"✓ 详细统计:")
        print(f"  - 总辅助损失: {stats['total_aux_loss']:.4f}")
        print(f"  - 前向传播次数: {stats['forward_count']}")
        
        if stats['layer_stats']:
            layer_0_stats = stats['layer_stats'][0]
            expert_usage = layer_0_stats['expert_compression_stats']['expert_usage']
            print(f"  - 专家使用情况: {expert_usage.tolist()}")
        
        print("✓ 模拟测试成功!")
        
    except Exception as e:
        print(f"模拟测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=== MoE路由器Press简化示例 ===\n")
    
    # 检查PyTorch版本
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    # 首先运行直接测试
    try:
        test_moe_router_directly()
        test_moe_press_compression()
    except Exception as e:
        print(f"直接测试失败: {e}")
    
    try:
        main()
    except Exception as e:
        print(f"主测试失败: {e}")
        print("尝试不依赖模型的测试...")
        test_without_model()
    
    print("\n=== 测试完成 ===") 