# KVPress + PiKV MoE Routing 对比实验

这个目录包含了用于比较不同KVPress和不同PiKV routing MoE结合效果的实验代码。

## 文件说明

- `test_kvpress_moe_comparison.py`: 完整的pytest测试套件，包含详细的对比实验
- `run_kvpress_moe_comparison.py`: 简化的运行脚本，可以直接执行对比实验
- `README_kvpress_moe_comparison.md`: 本说明文件

## 实验内容

### 1. MoE路由器类型对比
测试不同的MoE路由器类型：
- **base**: 基础MoE路由器
- **pikv**: PiKV专用MoE路由器（带缓存感知）
- **eplb**: 精确负载平衡路由器
- **hierarchical**: 层次化路由器

### 2. KVPress类型对比
测试不同的KVPress类型：
- **duo_attention**: DuoAttentionPress
- **moe_base**: 基础MoE路由器Press
- **moe_pikv**: PiKV MoE路由器Press

### 3. 组合Press对比
测试不同的组合配置：
- **duo_moe_base**: DuoAttention + 基础MoE路由器
- **duo_moe_pikv**: DuoAttention + PiKV MoE路由器
- **adakov_moe_eplb**: AdaKV + EPLB MoE路由器
- **moe_hierarchical_duo**: 层次化MoE路由器 + DuoAttention

### 4. 压缩比敏感性实验
测试不同压缩比参数的影响：
- 压缩比: 0.1, 0.3, 0.5, 0.7

### 5. 专家策略对比
测试不同的专家压缩策略：
- **aggressive**: 激进压缩（保留前20%和后10%）
- **moderate**: 中等压缩（保留前30%和后20%）
- **conservative**: 保守压缩（保留前50%和后30%）
- **selective**: 选择性压缩（基于重要性选择）

### 6. 性能基准测试
包含多次运行的统计测试，计算平均值和标准差。

## 使用方法

### 方法1: 运行简化脚本
```bash
cd tests
python run_kvpress_moe_comparison.py
```

### 方法2: 运行pytest测试
```bash
cd tests
python -m pytest test_kvpress_moe_comparison.py -v -s
```

### 方法3: 运行特定测试
```bash
# 只运行MoE路由器类型对比
python -m pytest test_kvpress_moe_comparison.py::TestKVPressMoEComparison::test_moe_router_types_comparison -v -s

# 只运行KVPress类型对比
python -m pytest test_kvpress_moe_comparison.py::TestKVPressMoEComparison::test_kvpress_types_comparison -v -s

# 只运行组合Press对比
python -m pytest test_kvpress_moe_comparison.py::TestKVPressMoEComparison::test_combined_press_comparison -v -s
```

## 输出指标

每个实验都会输出以下指标：

1. **压缩比 (Compression Ratio)**: 原始序列长度与压缩后序列长度的比例
2. **推理时间 (Inference Time)**: 压缩操作所需的时间
3. **内存使用 (Memory Usage)**: 压缩后KV缓存的内存占用
4. **序列长度变化**: 原始长度 -> 压缩后长度

## 实验配置

### 测试数据配置
- 批次大小: 2
- 序列长度: 100
- 隐藏层大小: 512
- 注意力头数: 8
- 头维度: 64

### MoE路由器配置
- 专家数量: 4
- Top-K: 2
- 容量因子: 1.5
- 缓存感知: True

### 压缩策略配置
- 默认压缩比: 0.3-0.5
- 专家策略: aggressive, moderate, conservative, selective

## 预期结果

### 性能排序（预期）
1. **组合Press** (DuoAttention + PiKV MoE) - 最佳压缩效果
2. **PiKV MoE路由器** - 良好的压缩效果和缓存感知
3. **EPLB MoE路由器** - 平衡的负载分布
4. **基础MoE路由器** - 基础压缩效果
5. **DuoAttention** - 注意力层面的压缩

### 压缩比影响
- 更高的压缩比 → 更小的压缩后大小
- 更高的压缩比 → 可能更长的推理时间
- 更高的压缩比 → 更低的内存使用

## 故障排除

### 常见问题

1. **导入错误**
   ```bash
   pip install -e .  # 在项目根目录执行
   ```

2. **CUDA内存不足**
   - 减少批次大小或序列长度
   - 使用CPU设备

3. **测试失败**
   - 检查kvpress模块是否正确安装
   - 检查依赖包版本

### 调试模式
```bash
# 启用详细输出
python run_kvpress_moe_comparison.py --debug

# 使用pytest的详细输出
python -m pytest test_kvpress_moe_comparison.py -v -s --tb=short
```

## 扩展实验

### 添加新的路由器类型
在`test_moe_router_types_comparison`中添加新的路由器类型：
```python
router_types = ["base", "pikv", "eplb", "hierarchical", "your_new_router"]
```

### 添加新的Press类型
在`test_kvpress_types_comparison`中添加新的Press配置：
```python
kvpress_configs = {
    'duo_attention': DuoAttentionPress(head_compression_ratio=0.3),
    'your_new_press': YourNewPress(compression_ratio=0.3),
}
```

### 自定义测试数据
修改`test_data`fixture来使用不同的数据配置：
```python
@pytest.fixture
def test_data(self):
    # 自定义配置
    batch_size, seq_len, hidden_size = 4, 200, 768
    # ...
```

## 贡献

欢迎提交新的实验配置和测试用例！请确保：
1. 新测试遵循现有的命名规范
2. 包含适当的错误处理
3. 提供清晰的文档说明
4. 通过所有现有的测试 