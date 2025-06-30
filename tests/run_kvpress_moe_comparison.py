#!/usr/bin/env python3
"""
KVPress + PiKV MoE Routing 对比实验运行脚本

运行不同的KVPress和不同PiKV routing MoE的结合效果对比实验
"""

import torch
import time
import numpy as np
from typing import Dict, List

# 导入kvpress模块
try:
    from kvpress import (
        MoERouterPress, 
        DuoAttentionPress, 
        ComposedPress,
        BasePress
    )
    print("✓ 成功导入kvpress模块")
except ImportError as e:
    print(f"✗ 导入kvpress模块失败: {e}")
    exit(1)


class KVPressMoEComparison:
    """KVPress + MoE Routing 对比实验"""
    
    def __init__(self):
        """初始化实验环境"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 创建测试数据
        self.batch_size, self.seq_len, self.hidden_size = 2, 100, 512
        self.num_heads, self.head_dim = 8, 64
        
        self.test_data = {
            'hidden_states': torch.randn(self.batch_size, self.seq_len, self.hidden_size, device=self.device),
            'keys': torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim, device=self.device),
            'values': torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim, device=self.device),
            'attentions': torch.randn(self.batch_size, self.num_heads, self.seq_len, self.seq_len, device=self.device),
            'kwargs': {}
        }
        
        # 创建模拟模块
        self.mock_module = type('MockModule', (), {'layer_idx': 0})()
    
    def measure_compression_metrics(self, press: BasePress) -> Dict[str, float]:
        """测量压缩指标"""
        start_time = time.time()
        
        try:
            # 执行压缩
            compressed_keys, compressed_values = press.compress(
                self.mock_module,
                self.test_data['hidden_states'],
                self.test_data['keys'],
                self.test_data['values'],
                self.test_data['attentions'],
                self.test_data['kwargs']
            )
            
            end_time = time.time()
            
            # 计算指标
            original_size = self.test_data['keys'].shape[2]
            compressed_size = compressed_keys.shape[2]
            compression_ratio = (original_size - compressed_size) / original_size
            inference_time = end_time - start_time
            
            # 计算内存使用（简化版本）
            memory_usage = (compressed_keys.numel() + compressed_values.numel()) * 4 / (1024**2)  # MB
            
            return {
                'compression_ratio': compression_ratio,
                'inference_time': inference_time,
                'memory_usage_mb': memory_usage,
                'original_size': original_size,
                'compressed_size': compressed_size,
                'success': True
            }
            
        except Exception as e:
            print(f"  错误: {e}")
            return {
                'compression_ratio': 0.0,
                'inference_time': 0.0,
                'memory_usage_mb': 0.0,
                'original_size': self.test_data['keys'].shape[2],
                'compressed_size': self.test_data['keys'].shape[2],
                'success': False,
                'error': str(e)
            }
    
    def test_moe_router_types(self):
        """测试不同MoE路由器类型"""
        print("\n" + "="*60)
        print("MoE路由器类型对比实验")
        print("="*60)
        
        router_types = ["base", "pikv", "eplb", "hierarchical"]
        results = {}
        
        for router_type in router_types:
            print(f"\n测试 {router_type.upper()} 路由器...")
            
            try:
                # 创建MoE路由器Press
                moe_press = MoERouterPress(
                    num_experts=4,
                    top_k=2,
                    router_type=router_type,
                    compression_ratio=0.5,
                    cache_aware=True
                )
                
                # 测量指标
                metrics = self.measure_compression_metrics(moe_press)
                results[router_type] = metrics
                
                if metrics['success']:
                    print(f"  压缩比: {metrics['compression_ratio']:.3f}")
                    print(f"  推理时间: {metrics['inference_time']:.4f}s")
                    print(f"  内存使用: {metrics['memory_usage_mb']:.2f} MB")
                    print(f"  序列长度: {metrics['original_size']} -> {metrics['compressed_size']}")
                else:
                    print(f"  失败: {metrics.get('error', '未知错误')}")
                    
            except Exception as e:
                print(f"  创建失败: {e}")
                results[router_type] = {'success': False, 'error': str(e)}
        
        # 打印对比结果
        print(f"\n{'路由器类型':<15} {'压缩比':<10} {'时间(s)':<10} {'内存(MB)':<10} {'状态':<10}")
        print("-" * 60)
        for router_type, metrics in results.items():
            if metrics.get('success', False):
                print(f"{router_type.upper():<15} {metrics['compression_ratio']:<10.3f} "
                      f"{metrics['inference_time']:<10.4f} {metrics['memory_usage_mb']:<10.2f} {'✓':<10}")
            else:
                print(f"{router_type.upper():<15} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'✗':<10}")
    
    def test_kvpress_types(self):
        """测试不同KVPress类型"""
        print("\n" + "="*60)
        print("KVPress类型对比实验")
        print("="*60)
        
        # 定义不同的KVPress配置
        kvpress_configs = {
            'duo_attention': DuoAttentionPress(head_compression_ratio=0.3),
            'moe_base': MoERouterPress(router_type="base", compression_ratio=0.3),
            'moe_pikv': MoERouterPress(router_type="pikv", compression_ratio=0.3),
        }
        
        results = {}
        
        for name, press in kvpress_configs.items():
            print(f"\n测试 {name.upper()}...")
            
            # 测量指标
            metrics = self.measure_compression_metrics(press)
            results[name] = metrics
            
            if metrics['success']:
                print(f"  压缩比: {metrics['compression_ratio']:.3f}")
                print(f"  推理时间: {metrics['inference_time']:.4f}s")
                print(f"  内存使用: {metrics['memory_usage_mb']:.2f} MB")
            else:
                print(f"  失败: {metrics.get('error', '未知错误')}")
        
        # 打印对比结果
        print(f"\n{'Press类型':<15} {'压缩比':<10} {'时间(s)':<10} {'内存(MB)':<10} {'状态':<10}")
        print("-" * 60)
        for name, metrics in results.items():
            if metrics.get('success', False):
                print(f"{name.upper():<15} {metrics['compression_ratio']:<10.3f} "
                      f"{metrics['inference_time']:<10.4f} {metrics['memory_usage_mb']:<10.2f} {'✓':<10}")
            else:
                print(f"{name.upper():<15} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'✗':<10}")
    
    def test_combined_press(self):
        """测试组合Press"""
        print("\n" + "="*60)
        print("组合Press对比实验")
        print("="*60)
        
        # 定义不同的组合配置
        combined_configs = {
            'duo_moe_base': ComposedPress([
                DuoAttentionPress(head_compression_ratio=0.2),
                MoERouterPress(router_type="base", compression_ratio=0.3)
            ]),
            'duo_moe_pikv': ComposedPress([
                DuoAttentionPress(head_compression_ratio=0.2),
                MoERouterPress(router_type="pikv", compression_ratio=0.3)
            ]),
        }
        
        results = {}
        
        for name, press in combined_configs.items():
            print(f"\n测试 {name.upper()}...")
            
            # 测量指标
            metrics = self.measure_compression_metrics(press)
            results[name] = metrics
            
            if metrics['success']:
                print(f"  压缩比: {metrics['compression_ratio']:.3f}")
                print(f"  推理时间: {metrics['inference_time']:.4f}s")
                print(f"  内存使用: {metrics['memory_usage_mb']:.2f} MB")
            else:
                print(f"  失败: {metrics.get('error', '未知错误')}")
        
        # 打印对比结果
        print(f"\n{'组合Press':<20} {'压缩比':<10} {'时间(s)':<10} {'内存(MB)':<10} {'状态':<10}")
        print("-" * 70)
        for name, metrics in results.items():
            if metrics.get('success', False):
                print(f"{name.upper():<20} {metrics['compression_ratio']:<10.3f} "
                      f"{metrics['inference_time']:<10.4f} {metrics['memory_usage_mb']:<10.2f} {'✓':<10}")
            else:
                print(f"{name.upper():<20} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'✗':<10}")
    
    def test_compression_ratio_sensitivity(self):
        """测试压缩比敏感性"""
        print("\n" + "="*60)
        print("压缩比敏感性实验")
        print("="*60)
        
        compression_ratios = [0.1, 0.3, 0.5, 0.7]
        router_type = "pikv"
        results = {}
        
        for ratio in compression_ratios:
            print(f"\n测试压缩比 {ratio}...")
            
            try:
                # 创建MoE路由器Press
                moe_press = MoERouterPress(
                    num_experts=4,
                    top_k=2,
                    router_type=router_type,
                    compression_ratio=ratio,
                    cache_aware=True
                )
                
                # 测量指标
                metrics = self.measure_compression_metrics(moe_press)
                results[ratio] = metrics
                
                if metrics['success']:
                    print(f"  实际压缩比: {metrics['compression_ratio']:.3f}")
                    print(f"  推理时间: {metrics['inference_time']:.4f}s")
                    print(f"  内存使用: {metrics['memory_usage_mb']:.2f} MB")
                else:
                    print(f"  失败: {metrics.get('error', '未知错误')}")
                    
            except Exception as e:
                print(f"  创建失败: {e}")
                results[ratio] = {'success': False, 'error': str(e)}
        
        # 打印对比结果
        print(f"\n{'目标压缩比':<15} {'实际压缩比':<15} {'时间(s)':<10} {'内存(MB)':<10} {'状态':<10}")
        print("-" * 70)
        for ratio, metrics in results.items():
            if metrics.get('success', False):
                print(f"{ratio:<15} {metrics['compression_ratio']:<15.3f} "
                      f"{metrics['inference_time']:<10.4f} {metrics['memory_usage_mb']:<10.2f} {'✓':<10}")
            else:
                print(f"{ratio:<15} {'N/A':<15} {'N/A':<10} {'N/A':<10} {'✗':<10}")
    
    def run_all_experiments(self):
        """运行所有实验"""
        print("🚀 开始KVPress + PiKV MoE Routing 对比实验")
        print("="*80)
        
        # 运行各种实验
        self.test_moe_router_types()
        self.test_kvpress_types()
        self.test_combined_press()
        self.test_compression_ratio_sensitivity()
        
        print("\n" + "="*80)
        print("✅ 所有实验完成！")
        print("="*80)


def main():
    """主函数"""
    try:
        # 创建实验实例
        experiment = KVPressMoEComparison()
        
        # 运行所有实验
        experiment.run_all_experiments()
        
    except Exception as e:
        print(f"❌ 实验运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 