# MoE路由器使用指南

本指南介绍如何在kvpress中使用各种MoE路由器类型进行KV缓存压缩。

## 概述

MoE路由器Press提供了6种不同的路由器类型，每种都有其特定的优势和适用场景：

1. **BaseMoERouter**: 基础路由器，提供标准的路由功能
2. **TopKBalancedRouter**: 负载平衡路由器，增强的负载平衡策略
3. **AdaptiveRouter**: 自适应路由器，基于输入重要性调整策略
4. **PiKVMoERouter**: PiKV专用路由器，结合KV缓存感知
5. **EPLBRouter**: 精确负载平衡路由器，严格的负载平衡约束
6. **HierarchicalRouter**: 层次化路由器，两级路由策略

## 基本用法

### 1. 基础路由器

```python
from kvpress.presses import MoERouterPress

# 使用基础路由器
moe_press = MoERouterPress(
    num_experts=4,
    top_k=2,
    router_type="base"
)
```

### 2. TopK平衡路由器

```python
# 使用负载平衡路由器
moe_press = MoERouterPress(
    num_experts=4,
    top_k=2,
    router_type="topk_balanced",
    balance_coefficient=0.01,  # 平衡损失权重
    balance_mode="entropy"     # 平衡模式: "entropy", "variance", "gini"
)
```

**平衡模式说明**:
- `entropy`: 最大化熵以实现均匀分布
- `variance`: 最小化方差
- `gini`: 最小化基尼系数

### 3. 自适应路由器

```python
# 使用自适应路由器
moe_press = MoERouterPress(
    num_experts=4,
    top_k=2,
    router_type="adaptive",
    importance_threshold=0.5,  # 重要性阈值
    adaptive_top_k=True        # 启用自适应top_k
)
```

**特性**:
- 自动计算输入重要性分数
- 动态调整top_k值
- 重要token获得更高权重

### 4. PiKV专用路由器

```python
# 使用PiKV路由器
moe_press = MoERouterPress(
    num_experts=4,
    top_k=2,
    router_type="pikv",
    importance_threshold=0.5,
    cache_aware=True,          # 启用缓存感知
    cache_update_interval=100  # 缓存更新间隔
)
```

**特性**:
- 结合KV缓存使用情况
- 自适应重要性路由
- 缓存感知的路由调整

### 5. 精确负载平衡路由器

```python
# 使用EPLB路由器
moe_press = MoERouterPress(
    num_experts=4,
    top_k=2,
    router_type="eplb",
    balance_coefficient=0.1,   # 平衡损失权重
    temperature=1.0            # 温度参数
)
```

**特性**:
- 严格的负载平衡约束
- 动态专家权重调整
- 温度缩放控制

### 6. 层次化路由器

```python
# 使用层次化路由器
moe_press = MoERouterPress(
    num_experts=8,
    top_k=2,
    router_type="hierarchical",
    num_groups=4,              # 专家组数量
    group_top_k=1              # 组级top_k
)
```

**特性**:
- 两级路由策略
- 先路由到专家组，再组内路由
- 层次化负载平衡

## 高级配置

### 专家压缩策略

每种路由器都支持4种专家压缩策略：

```python
# 专家策略配置（内部自动分配）
expert_strategies = {
    0: "aggressive",    # 激进压缩：保留前20%和后10%
    1: "moderate",      # 中等压缩：保留前30%和后20%
    2: "conservative",  # 保守压缩：保留前50%和后30%
    3: "selective"      # 选择性压缩：基于重要性选择
}
```

### 缓存感知配置

```python
# 启用缓存感知
moe_press = MoERouterPress(
    router_type="pikv",
    cache_aware=True,
    cache_update_interval=100
)
```

### 负载平衡配置

```python
# 多种负载平衡策略
moe_press = MoERouterPress(
    router_type="topk_balanced",
    balance_mode="entropy",     # 熵最大化
    # balance_mode="variance",  # 方差最小化
    # balance_mode="gini",      # 基尼系数最小化
    balance_coefficient=0.01
)
```

## 实际使用示例

### 示例1: 基础文本生成

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from kvpress.presses import MoERouterPress

# 加载模型
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

# 创建MoE路由器Press
moe_press = MoERouterPress(
    num_experts=4,
    router_type="adaptive",
    importance_threshold=0.5
)

# 使用MoE路由器进行推理
text = "Hello world, this is a test of the MoE router."
inputs = tokenizer(text, return_tensors="pt")

with moe_press(model):
    outputs = model.generate(**inputs, max_new_tokens=20)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"生成的文本: {generated_text}")

# 获取统计信息
stats = moe_press.get_stats()
print(f"总辅助损失: {stats['total_aux_loss']:.4f}")
print(f"平均辅助损失: {stats['avg_aux_loss']:.4f}")
```

### 示例2: 不同路由器对比

```python
import torch
from kvpress.presses import MoERouterPress

# 测试数据
hidden_states = torch.randn(2, 10, 768)
keys = torch.randn(2, 12, 20, 64)
values = torch.randn(2, 12, 20, 64)

# 测试不同路由器
router_types = ["base", "topk_balanced", "adaptive", "pikv", "eplb", "hierarchical"]

for router_type in router_types:
    print(f"\n测试 {router_type} 路由器:")
    
    moe_press = MoERouterPress(
        num_experts=4,
        router_type=router_type
    )
    
    # 执行压缩
    compressed_keys, compressed_values = moe_press.compress(
        module=None,  # 模拟模块
        hidden_states=hidden_states,
        keys=keys,
        values=values,
        attentions=torch.randn(2, 12, 10, 20),
        kwargs={}
    )
    
    # 计算压缩比
    compression_ratio = (keys.shape[2] - compressed_keys.shape[2]) / keys.shape[2]
    print(f"压缩比例: {compression_ratio:.2f}")
    
    # 获取统计
    stats = moe_press.get_stats()
    print(f"辅助损失: {stats['total_aux_loss']:.4f}")
```

## 性能优化建议

### 1. 路由器选择

- **基础任务**: 使用 `base` 路由器
- **需要负载平衡**: 使用 `topk_balanced` 或 `eplb` 路由器
- **复杂输入**: 使用 `adaptive` 路由器
- **KV缓存优化**: 使用 `pikv` 路由器
- **大规模模型**: 使用 `hierarchical` 路由器

### 2. 参数调优

```python
# 根据任务调整参数
moe_press = MoERouterPress(
    num_experts=8,              # 更多专家用于复杂任务
    top_k=3,                    # 更高top_k提高质量
    capacity_factor=2.0,        # 更高容量因子
    compression_ratio=0.3,      # 更激进的压缩
    balance_coefficient=0.05    # 更强的负载平衡
)
```

### 3. 监控和调试

```python
# 获取详细统计信息
stats = moe_press.get_stats()

# 层级统计
for layer_idx, layer_stats in stats['layer_stats'].items():
    print(f"第{layer_idx}层:")
    print(f"  专家使用比例: {layer_stats['router_stats']['expert_usage_ratios']}")
    print(f"  压缩比例: {layer_stats['expert_compression_stats']['compression_ratios']}")

# 重置统计
moe_press.reset_stats()
```

## 故障排除

### 常见问题

1. **内存不足**: 减少 `num_experts` 或 `capacity_factor`
2. **压缩效果差**: 调整 `compression_ratio` 或专家策略
3. **负载不平衡**: 增加 `balance_coefficient` 或使用 `eplb` 路由器
4. **性能下降**: 检查 `top_k` 和 `cache_aware` 设置

### 调试技巧

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查路由器状态
router = moe_press._get_router(0, 768)
print(f"路由器类型: {type(router)}")
print(f"专家数量: {router.num_experts}")
print(f"Top-k: {router.top_k}")
```

## 总结

MoE路由器Press提供了灵活且强大的KV缓存压缩解决方案。通过选择合适的路由器类型和参数配置，可以在保持模型性能的同时实现有效的缓存压缩。建议根据具体任务需求和性能要求来选择最适合的路由器类型。 