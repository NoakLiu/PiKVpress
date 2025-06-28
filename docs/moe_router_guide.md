# MoE路由器Press使用指南

## 概述

MoE路由器Press是kvpress中的一个高级KV缓存压缩方法，它使用Mixture of Experts (MoE) 架构来智能地决定如何压缩KV缓存。该方法通过多个专家网络来分析输入特征和缓存使用模式，为每个层选择最合适的压缩策略。

## 特性

- **多专家路由**: 使用4个专家，每个专家负责不同的压缩策略
- **缓存感知**: PiKV路由器能够根据缓存使用情况调整路由决策
- **自适应压缩**: 根据输入特征动态选择压缩策略
- **负载平衡**: 内置负载平衡机制，确保专家使用均匀
- **统计监控**: 提供详细的路由和压缩统计信息

## 专家策略

MoE路由器包含4个专家，每个专家使用不同的压缩策略：

1. **Expert 0 - 激进压缩**: 保留前20%和后10%的KV对
2. **Expert 1 - 中等压缩**: 保留前30%和后20%的KV对
3. **Expert 2 - 保守压缩**: 保留前50%和后30%的KV对
4. **Expert 3 - 选择性压缩**: 基于注意力权重选择重要位置

## 基本使用

### 1. 导入和初始化

```python
from kvpress.presses import MoERouterPress
from transformers import LlamaForCausalLM, LlamaTokenizer

# 创建MoE路由器Press
moe_press = MoERouterPress(
    num_experts=4,           # 专家数量
    top_k=2,                 # 每个token路由到的专家数
    capacity_factor=1.5,     # 容量因子
    dropout=0.1,             # dropout率
    router_type="pikv",      # 路由器类型
    cache_aware=True,        # 启用缓存感知
    compression_ratio=0.5,   # 目标压缩比
    aux_loss_weight=0.01     # 辅助损失权重
)
```

### 2. 应用到模型

```python
# 加载模型
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 使用MoE路由器进行推理
with moe_press(model):
    inputs = tokenizer("Your input text here", return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100)
```

### 3. 获取统计信息

```python
# 获取详细统计信息
stats = moe_press.get_stats()

print(f"总辅助损失: {stats['total_aux_loss']}")
print(f"平均辅助损失: {stats['avg_aux_loss']}")
print(f"前向传播次数: {stats['forward_count']}")

# 查看每层的专家使用情况
for layer_idx, layer_stats in stats['layer_stats'].items():
    router_stats = layer_stats['router_stats']
    print(f"第{layer_idx}层专家使用比例: {router_stats['expert_usage_ratios']}")
```

## 高级配置

### 路由器类型

支持两种路由器类型：

1. **BaseMoERouter**: 基础路由器，适用于一般场景
2. **PiKVMoERouter**: PiKV专用路由器，支持缓存感知

```python
# 使用基础路由器
moe_press = MoERouterPress(router_type="base")

# 使用PiKV路由器
moe_press = MoERouterPress(router_type="pikv", cache_aware=True)
```

### 自定义专家策略

你可以继承`MoERouterPress`类来自定义专家策略：

```python
class CustomMoERouterPress(MoERouterPress):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 自定义专家策略
        self.expert_strategies = {
            0: "ultra_aggressive",
            1: "semantic_aware", 
            2: "temporal_aware",
            3: "adaptive"
        }
    
    def _apply_expert_compression(self, keys, values, strategy, router_probs):
        # 实现自定义压缩逻辑
        if strategy == "semantic_aware":
            # 基于语义重要性的压缩
            pass
        # ... 其他策略
        return keys, values
```

## 性能优化

### 1. 调整专家数量

根据模型大小和任务复杂度调整专家数量：

```python
# 小型模型
moe_press = MoERouterPress(num_experts=2)

# 大型模型
moe_press = MoERouterPress(num_experts=8)
```

### 2. 优化容量因子

容量因子影响每个专家的处理能力：

```python
# 高容量（更多并行处理）
moe_press = MoERouterPress(capacity_factor=2.0)

# 低容量（更少并行处理）
moe_press = MoERouterPress(capacity_factor=1.0)
```

### 3. 调整辅助损失权重

辅助损失权重影响负载平衡：

```python
# 强负载平衡
moe_press = MoERouterPress(aux_loss_weight=0.1)

# 弱负载平衡
moe_press = MoERouterPress(aux_loss_weight=0.001)
```

## 监控和调试

### 1. 专家使用情况

```python
stats = moe_press.get_stats()
for layer_idx, layer_stats in stats['layer_stats'].items():
    expert_usage = layer_stats['expert_compression_stats']['expert_usage']
    print(f"第{layer_idx}层专家使用情况: {expert_usage}")
```

### 2. 压缩效果

```python
for layer_idx, layer_stats in stats['layer_stats'].items():
    compression_ratios = layer_stats['expert_compression_stats']['compression_ratios']
    print(f"第{layer_idx}层压缩比例: {compression_ratios}")
```

### 3. 缓存命中率

```python
for layer_idx, layer_stats in stats['layer_stats'].items():
    cache_hit_rates = layer_stats['expert_compression_stats']['cache_hit_rates']
    print(f"第{layer_idx}层缓存命中率: {cache_hit_rates}")
```

## 最佳实践

### 1. 模型选择

MoE路由器适用于以下模型：
- Llama系列模型
- Mistral模型
- Qwen系列模型
- Gemma模型

### 2. 序列长度

- **短序列 (< 100 tokens)**: 使用保守压缩策略
- **中等序列 (100-500 tokens)**: 使用中等压缩策略
- **长序列 (> 500 tokens)**: 使用激进压缩策略

### 3. 内存管理

```python
# 定期重置统计信息以释放内存
moe_press.reset_stats()

# 在长时间推理后清理路由器
del moe_press.routers
```

## 故障排除

### 1. 专家使用不均匀

如果某些专家使用过多或过少：

```python
# 增加辅助损失权重
moe_press = MoERouterPress(aux_loss_weight=0.05)

# 调整top_k值
moe_press = MoERouterPress(top_k=1)  # 减少路由到单个专家
```

### 2. 压缩效果不佳

如果压缩效果不理想：

```python
# 调整压缩比例
moe_press = MoERouterPress(compression_ratio=0.7)

# 使用更激进的策略
moe_press.expert_strategies[0] = "ultra_aggressive"
```

### 3. 性能下降

如果推理性能下降：

```python
# 减少专家数量
moe_press = MoERouterPress(num_experts=2)

# 降低容量因子
moe_press = MoERouterPress(capacity_factor=1.0)

# 使用基础路由器
moe_press = MoERouterPress(router_type="base")
```

## 示例代码

完整的使用示例请参考 `examples/moe_router_example.py`。

## 技术细节

### 路由机制

1. **特征提取**: 从hidden states中提取特征
2. **路由计算**: 使用神经网络计算每个专家的概率
3. **专家选择**: 选择概率最高的专家
4. **压缩应用**: 应用专家特定的压缩策略

### 负载平衡

使用辅助损失来确保专家使用均匀：

```
aux_loss = sum(router_prob_per_expert * expert_usage_rate)
```

### 缓存感知

PiKV路由器通过以下方式实现缓存感知：

1. **缓存使用跟踪**: 记录每个专家的缓存命中率
2. **动态调整**: 根据缓存使用情况调整路由决策
3. **历史学习**: 使用历史数据优化路由策略

## 贡献

欢迎提交Issue和Pull Request来改进MoE路由器Press。请确保：

1. 遵循现有的代码风格
2. 添加适当的测试
3. 更新文档
4. 提供性能基准测试结果 