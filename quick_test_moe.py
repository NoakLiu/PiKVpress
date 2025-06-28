#!/usr/bin/env python3
"""
快速测试MoE路由器Press
"""

import torch
from kvpress.presses import MoERouterPress

def test_moe_router():
    """测试MoE路由器的基本功能"""
    print("=== 测试MoE路由器Press ===")
    
    # 创建MoE路由器Press
    moe_press = MoERouterPress(
        num_experts=4,
        top_k=2,
        router_type="pikv",
        cache_aware=True
    )
    
    print(f"✓ MoE路由器Press创建成功")
    print(f"  - 专家数量: {moe_press.num_experts}")
    print(f"  - Top-K: {moe_press.top_k}")
    print(f"  - 路由器类型: {moe_press.router_type}")
    print(f"  - 缓存感知: {moe_press.cache_aware}")
    
    # 测试路由器创建
    router = moe_press._get_router(layer_idx=0, hidden_size=512)
    print(f"✓ 路由器创建成功: {type(router).__name__}")
    
    # 测试前向传播
    hidden_states = torch.randn(2, 10, 512)
    dispatch_tensor, combine_tensor, router_probs, aux_loss = router(hidden_states)
    
    print(f"✓ 前向传播成功")
    print(f"  - 调度张量形状: {dispatch_tensor.shape}")
    print(f"  - 组合张量形状: {combine_tensor.shape}")
    print(f"  - 路由概率形状: {router_probs.shape}")
    print(f"  - 辅助损失: {aux_loss.item():.4f}")
    
    # 测试压缩策略
    keys = torch.randn(2, 8, 100, 64)
    values = torch.randn(2, 8, 100, 64)
    
    for strategy in ["aggressive", "moderate", "conservative", "selective"]:
        compressed_keys, compressed_values = moe_press._apply_expert_compression(
            keys, values, strategy, router_probs
        )
        compression_ratio = (keys.shape[2] - compressed_keys.shape[2]) / keys.shape[2]
        print(f"✓ {strategy}策略: {keys.shape[2]} -> {compressed_keys.shape[2]} (压缩比: {compression_ratio:.2f})")
    
    # 测试统计信息
    stats = moe_press.get_stats()
    print(f"✓ 统计信息获取成功")
    print(f"  - 前向传播次数: {stats['forward_count']}")
    print(f"  - 平均辅助损失: {stats['avg_aux_loss']:.4f}")
    
    print("\n=== 所有测试通过! ===")

def test_with_mock_model():
    """使用模拟模型测试"""
    print("\n=== 模拟模型测试 ===")
    
    from unittest.mock import Mock
    
    # 创建模拟模块
    mock_module = Mock()
    mock_module.layer_idx = 0
    
    # 创建MoE路由器Press
    moe_press = MoERouterPress(num_experts=4, router_type="base")
    
    # 创建测试数据
    hidden_states = torch.randn(2, 10, 512)
    keys = torch.randn(2, 8, 100, 64)
    values = torch.randn(2, 8, 100, 64)
    attentions = torch.randn(2, 8, 10, 100)
    kwargs = {}
    
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

if __name__ == "__main__":
    try:
        test_moe_router()
        test_with_mock_model()
        print("\n🎉 MoE路由器Press测试完成!")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc() 