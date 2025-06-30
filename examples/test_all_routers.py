#!/usr/bin/env python3
"""
测试所有MoE路由器类型

验证以下路由器：
- BaseMoERouter: 基础路由器
- TopKBalancedRouter: 负载平衡路由器
- AdaptiveRouter: 自适应路由器
- PiKVMoERouter: PiKV专用路由器
- EPLBRouter: 精确负载平衡路由器
- HierarchicalRouter: 层次化路由器
"""

import torch
import torch.nn as nn
from kvpress.presses.moe_router_press import (
    BaseMoERouter,
    TopKBalancedRouter,
    AdaptiveRouter,
    PiKVMoERouter,
    EPLBRouter,
    HierarchicalRouter
)

def test_router(router_class, router_name, **kwargs):
    """测试单个路由器"""
    print(f"\n=== 测试 {router_name} ===")
    
    try:
        # 创建路由器
        router = router_class(
            hidden_size=768,
            num_experts=4,
            top_k=2,
            **kwargs
        )
        
        # 创建测试数据
        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(batch_size, seq_len, 768)
        
        print(f"输入形状: {hidden_states.shape}")
        
        # 执行路由
        if router_class in [AdaptiveRouter, PiKVMoERouter]:
            # 这些路由器返回5个值
            dispatch_tensor, combine_tensor, router_probs, aux_loss, importance = router(hidden_states)
            print(f"重要性分数形状: {importance.shape}")
        else:
            # 其他路由器返回4个值
            dispatch_tensor, combine_tensor, router_probs, aux_loss = router(hidden_states)
        
        print(f"✓ {router_name} 路由成功")
        print(f"  - 调度张量形状: {dispatch_tensor.shape}")
        print(f"  - 组合张量形状: {combine_tensor.shape}")
        print(f"  - 路由概率形状: {router_probs.shape}")
        print(f"  - 辅助损失: {aux_loss.item():.4f}")
        
        # 获取统计信息
        stats = router.get_routing_stats()
        print(f"✓ {router_name} 统计:")
        print(f"  - 专家使用比例: {stats['expert_usage_ratios'].tolist()}")
        print(f"  - 专家使用计数: {stats['expert_usage_count'].tolist()}")
        print(f"  - 总token数: {stats['total_tokens'].item()}")
        
        # 特殊统计信息
        if hasattr(router, 'get_balance_loss_stats'):
            balance_stats = router.get_balance_loss_stats()
            if balance_stats:
                print(f"  - 平衡损失统计: {balance_stats}")
        
        return True
        
    except Exception as e:
        print(f"✗ {router_name} 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_moe_press_with_different_routers():
    """测试MoE Press与不同路由器"""
    print("\n=== 测试MoE Press与不同路由器 ===")
    
    from kvpress.presses import MoERouterPress
    from unittest.mock import Mock
    
    router_types = [
        "base",
        "topk_balanced", 
        "adaptive",
        "pikv",
        "eplb",
        "hierarchical"
    ]
    
    for router_type in router_types:
        print(f"\n--- 测试 {router_type} 路由器 ---")
        
        try:
            # 创建MoE Press
            moe_press = MoERouterPress(
                num_experts=4,
                router_type=router_type,
                cache_aware=False
            )
            
            # 创建模拟模块
            mock_module = Mock()
            mock_module.layer_idx = 0
            
            # 创建测试数据
            hidden_states = torch.randn(2, 10, 768)
            keys = torch.randn(2, 12, 20, 64)
            values = torch.randn(2, 12, 20, 64)
            attentions = torch.randn(2, 12, 10, 20)
            kwargs = {}
            
            print(f"原始KV形状: {keys.shape}")
            
            # 执行压缩
            compressed_keys, compressed_values = moe_press.compress(
                mock_module, hidden_states, keys, values, attentions, kwargs
            )
            
            print(f"✓ {router_type} 压缩成功")
            print(f"  - 压缩后KV形状: {compressed_keys.shape}")
            print(f"  - 压缩比例: {(keys.shape[2] - compressed_keys.shape[2]) / keys.shape[2]:.2f}")
            
            # 获取统计信息
            stats = moe_press.get_stats()
            print(f"✓ {router_type} MoE Press统计:")
            print(f"  - 总辅助损失: {stats['total_aux_loss']:.4f}")
            print(f"  - 前向传播次数: {stats['forward_count']}")
            
            if stats['layer_stats']:
                layer_0_stats = stats['layer_stats'][0]
                expert_usage = layer_0_stats['expert_compression_stats']['expert_usage']
                print(f"  - 专家使用情况: {expert_usage.tolist()}")
            
        except Exception as e:
            print(f"✗ {router_type} MoE Press测试失败: {e}")
            import traceback
            traceback.print_exc()

def main():
    print("=== MoE路由器全面测试 ===\n")
    
    # 检查PyTorch版本
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    # 测试各个路由器
    test_results = []
    
    # 基础路由器
    test_results.append(test_router(BaseMoERouter, "BaseMoERouter"))
    
    # TopK平衡路由器
    test_results.append(test_router(TopKBalancedRouter, "TopKBalancedRouter", 
                                   balance_coefficient=0.01, balance_mode="entropy"))
    
    # 自适应路由器
    test_results.append(test_router(AdaptiveRouter, "AdaptiveRouter",
                                   importance_threshold=0.5, adaptive_top_k=True))
    
    # PiKV路由器
    test_results.append(test_router(PiKVMoERouter, "PiKVMoERouter",
                                   importance_threshold=0.5, cache_aware=True))
    
    # EPLB路由器
    test_results.append(test_router(EPLBRouter, "EPLBRouter",
                                   balance_coefficient=0.1, temperature=1.0))
    
    # 层次化路由器
    test_results.append(test_router(HierarchicalRouter, "HierarchicalRouter",
                                   num_groups=2, group_top_k=1))
    
    # 测试MoE Press
    test_moe_press_with_different_routers()
    
    # 总结
    print(f"\n=== 测试总结 ===")
    print(f"成功测试的路由器: {sum(test_results)}/{len(test_results)}")
    
    if all(test_results):
        print("✓ 所有路由器测试通过!")
    else:
        print("✗ 部分路由器测试失败")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    main() 