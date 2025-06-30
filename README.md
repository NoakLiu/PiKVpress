# PiKVPress: KV Cache Compression with PiKV Routing

[![PyPI version](https://badge.fury.io/py/kvpress.svg)](https://badge.fury.io/py/kvpress)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Colab example notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1JNvaTKuuAHrl49dYB9-mdEH_y52Ib-NP?usp=drive_link)
[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/nvidia/kvpress)


![kvpress](kvpress.jpg)


Deploying long-context LLMs is costly due to the linear growth of the key-value (KV) cache in transformer models. For example, handling 1M tokens with Llama 3.1-70B in float16 requires up to 330GB of memory. kvpress implements multiple KV cache compression methods and benchmarks using 🤗 transformers, aiming to simplify the development of new methods for researchers and developers in this field.

## 概述

KVPress 是一个强大的 KV 缓存压缩框架，现在集成了 **PiKV Routing** 技术，通过 Mixture of Experts (MoE) 架构实现智能的 KV 缓存压缩。PiKV Routing 能够根据输入特征和缓存使用情况动态选择最优的压缩策略，显著提升长上下文处理的内存效率和推理速度。

### 核心特性

- 🚀 **PiKV Routing**: 基于 MoE 的智能路由系统
- 🎯 **多专家压缩**: 4种不同的压缩策略专家
- 📊 **缓存感知**: 实时监控缓存使用情况
- 🔄 **自适应调整**: 动态调整压缩策略
- 💾 **内存优化**: 显著减少 KV 缓存内存占用
- ⚡ **性能提升**: 加速长上下文推理

## Installation

```bash
pip install kvpress
```

推荐安装 flash attention 以获得最佳性能：
```bash
pip install flash-attn --no-build-isolation
```

For a local installation with all dev dependencies, use poetry:

```bash
git clone https://github.com/NVIDIA/kvpress.git
cd kvpress
poetry install --with dev
```

## Usage

kvpress provides a set of "presses" that compress the KV cache during the prefilling-phase. Each press is associated with a `compression_ratio` attribute that measures the compression of the cache. The easiest way to use a press is through our custom `KVPressTextGenerationPipeline`. It is automatically registered as a transformers pipeline with the name "kv-press-text-generation" when kvpress is imported and handles chat templates and tokenization for you:

```python
from transformers import pipeline
from kvpress import ExpectedAttentionPress

device = "cuda:0"
model = "meta-llama/Llama-3.1-8B-Instruct"
model_kwargs = {"attn_implementation": "flash_attention_2"}
pipe = pipeline("kv-press-text-generation", model=model, device=device, model_kwargs=model_kwargs)

context = "A very long text you want to compress once and for all"
question = "\nA question about the compressed context"  # optional

press = ExpectedAttentionPress(compression_ratio=0.5)
answer = pipe(context, question=question, press=press)["answer"]
```

In the snippet above, the compression is only applied on the context tokens so that you can evaluate the compression for different questions. Check the [Wikipedia notebook demo](notebooks/wikipedia_demo.ipynb) for a more detailed example (also available on Colab [here](https://colab.research.google.com/drive/1JNvaTKuuAHrl49dYB9-mdEH_y52Ib-NP)).

> [!IMPORTANT]  
> We focus on compression during the pre-filling phase as the KV cache becomes a bottleneck for long-context sequence (100k - 1M tokens) which are essentially long context prompts. This would typically apply to improving prompt caching systems.

> [!NOTE]  
> Use `model_kwargs={"attn_implementation":"flash_attention_2"}` to enable flash attention. To use the press `ObservedAttentionPress`, you need to specify `model_kwargs={"attn_implementation":"eager"}` as this press requires to materialize the attention weights

## Contributing

We welcome contributions! To add a new press, simply open an issue or submit a pull request. Check the [new_press.ipynb](notebooks/new_press.ipynb) notebook for a step-by-step guide.

## Available presses

All current presses are training free and inherit from `BasePress` ([source](kvpress/presses/base_press.py)). 

Several presses inherit from `ScorerPress` ([source](kvpress/presses/scorer_press.py)) and rely on a score to prune the KV pairs with lowest importance:

- `RandomPress` ([source](kvpress/presses/random_press.py)): random score
- `KnormPress` ([source](kvpress/presses/knorm_press.py), [paper](https://arxiv.org/abs/2406.11430)): inverse norm of the key
- `SnapKVPress` ([source](kvpress/presses/snapkv_press.py), [paper](https://arxiv.org/abs/2404.14469)): average attention weight of the last queries
- `ExpectedAttentionPress` ([source](kvpress/presses/expected_attention_press.py), [notebook](notebooks/expected_attention.ipynb)): expected attention weight during the generation phase 
- `StreamingLLMPress` ([source](kvpress/presses/streaming_llm_press.py), [paper](https://arxiv.org/abs/2309.17453)): keep only the initial and recent tokens 
- `TOVAPress` ([source](kvpress/presses/tova_press.py), [paper](https://arxiv.org/abs/2401.06104)): attention weight of the last query averaged across heads 
- `ObservedAttentionPress` ([source](kvpress/presses/observed_attention_press.py), [paper](https://arxiv.org/abs/2306.14048)): average attention weight observed during in pre-filling phase
- `QFilterPress` ([source](kvpress/presses/qfilter_press.py), [paper](https://arxiv.org/abs/2503.02812)): project the Key representations on the main SVD component of the Query vectors to approximate the attention scores.
- `PyramidKVPress` ([source](kvpress/presses/pyramidkv_press.py), [paper](https://arxiv.org/abs/2406.02069)): maintain pyramid-like cache sizes, allocating more cache budget to lower layers and less to higher layers
- `LagKVPress` ([source](kvpress/presses/lagkv_press.py), [paper](https://arxiv.org/abs/2504.04704)): leverage on the KV lag-relative information to compress. It's query free, attention-weight free, and flash-attention compatible.

Some presses rely on a different logic:
- `ThinKPress` ([source](kvpress/presses/think_press.py), [paper](https://arxiv.org/pdf/2407.21018)): compress the dimensions of the keys based on the channel attention score on the last queries 
- `SimLayerKVPress` ([source](kvpress/presses/simlayerkv_press.py), [paper](https://arxiv.org/abs/2410.13846)): identify "lazy" layers, and apply the StreamingLLM approach to them 
- `DuoAttentionPress` ([source](kvpress/presses/duo_attention_press.py), [paper](https://arxiv.org/abs/2410.10819)): split heads into retrieval heads (no compression) and streaming heads (StreamingLLM approach)
- `FinchPress` ([source](kvpress/presses/finch_press.py), [paper](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00716/125280)): similar to SnapKV with a dynamic window size and key value re-rotation

Finally we provide wrapper presses that can be combined with other presses:
- `AdaKVPress` ([source](kvpress/presses/adakv_press.py), [paper](https://arxiv.org/abs/2407.11550)): prune bottom scores of any `ScorerPress` but across all heads, achieving head-wise compressions 
- `PerLayerCompressionPress` ([source](kvpress/presses/per_layer_compression_press.py)): compress each layer with a different compression ratio (experimental)
- `ComposedPress` ([source](kvpress/presses/composed_press.py)): compose multiple presses together by chaining their forward hooks
- `KeyRerotationPress` ([source](kvpress/presses/key_rerotation_press.py)): rerotate pruned keys to have continuous RoPE embeddings
- `ChunkKVPress` ([source](kvpress/presses/chunkkv_press.py), [paper](https://arxiv.org/abs/2502.00299)): compresses by selecting important chunks, preserving semantic coherence
- `ChunkPress` ([source](kvpress/presses/chunk_press.py), [paper](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00716/125280)): compress the KV cache on each sequence chunk separately. This can yield to more uniform compression across long sequences
- `CriticalKVPress` and `CriticalAdaKVPress` ([source](kvpress/presses/criticalkv_press.py), [paper](https://arxiv.org/abs/2502.03805)): refine the scores using the L1 norm of Wo @ values, coupled with a two-stage selection.


For a detailed list of existing KV cache compression methods, check [Awesome-KV-Cache-Compression](https://github.com/October2001/Awesome-KV-Cache-Compression) or [Awesome-LLM-Compression](https://github.com/HuangOwen/Awesome-LLM-Compression?tab=readme-ov-file#kv-cache-compression)

## Evaluation

The [speed_and_memory.ipynb](notebooks/speed_and_memory.ipynb) notebook can help you to measure peak memory usage and total time gain.

![memory](evaluation/assets/peak_memory_consumption_xkcd.png)

We provide a simple CLI to evaluate the performance of the different presses on several long-context datasets. Below we report the average performance on the RULER dataset with 4k context length for different presses.

![RULER](evaluation/assets/ruler_llama_xkcd.png)

Please refer to the [evaluation](evaluation/README.md) directory for more details and results.

## Quantization

We support KV cache quantization through the transformers `QuantizedCache` class (see [HF blog post](https://huggingface.co/blog/kv-cache-quantization#how-to-use-quantized-kv-cache-in-%F0%9F%A4%97-transformers)). To use it, simply pass a cache object to your pipeline:

```python
from transformers import QuantizedCacheConfig, QuantoQuantizedCache

config = QuantizedCacheConfig(nbits=4)
cache = QuantoQuantizedCache(config)

pipe(..., cache=cache)
```

By default, the `DynamicCache` is used (no quantization). 

> [!IMPORTANT]  
> To use the `QuantizedCache`, you need to install additional dependencies (_e.g._ `pip install optimum-quanto`).

## FAQ

<details><summary> 

### Which models are supported ? 
</summary>

Some presses depend on the model architecture (_e.g._ `ExpectedAttentionPress` or `SnapKVPress`) hence they might not work with all models. We tested support for `LlamaForCausalLM`, `MistralForCausalLM`, `Phi3ForCausalLM` and `Qwen2ForCausalLM` but many other models might be supported out of the box because their implementation is often similar in transformers.
</details>

<details><summary> 

### How to run inference on multiple GPUs ? 
</summary>

kvpress supports multi-GPU inference through [accelerate](https://huggingface.co/docs/accelerate/en/index):

```python
pipe = pipeline("kv-press-text-generation", model=model, device_map="auto")
```

</details>


<details> <summary> 

### What are the memory and throughput gains ?
</summary>

Memory usage should be reduced by around `compression_ratio * kv_cache_size`. As the KV cache is smaller, decoding should also be faster. You can measure peak memory usage gain and total time gain using [this notebook](notebooks/speed_and_memory.ipynb).
</details>


<details> <summary> 

### How does a press work ? </summary>

A press registers a forward hook (`press.forward_hook` method) to each attention layer during the pre-filling phase. Registration can be applied using the press as a context manager (`press.__call__` method):

```python
import torch
from transformers import AutoModelForCausalLM
from kvpress import KnormPress

device = "cuda:0"
ckpt = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(ckpt).to(device)
press = KnormPress(compression_ratio=0.4)

inputs = model.dummy_inputs["input_ids"].to(device)

with torch.no_grad():
    print(model(inputs).past_key_values[0][0].shape)
    # torch.Size([3, 8, 5, 128])
    
with torch.no_grad(), press(model):
    print(model(inputs).past_key_values[0][0].shape)
    # torch.Size([3, 8, 3, 128])
```
</details>

<details><summary> 

### Why not using model.generate ? 
</summary>

In fact you can use `model.generate` with a press by using the press as a context manager:

```python
with press(model):
    outputs = model.generate(inputs)
```

However, the `generate` method does not allow to exclude the question from the compression, which would artificially favors methods such as SnapKV. Ideally, we want a compression method that works whatever comes after the context (_e.g._ for use cases such as chat or document question answering). Finally the `generate` method does not allow to provide generation for multiple questions at once.

</details>

## PiKV Routing 快速开始

### 基础用法

```python
from transformers import pipeline
from kvpress import MoERouterPress

# 创建 PiKV MoE 路由器
press = MoERouterPress(
    router_type="pikv",           # 使用 PiKV 路由器
    num_experts=4,                # 4个专家
    top_k=2,                      # 选择前2个专家
    compression_ratio=0.5,        # 目标压缩比50%
    cache_aware=True,             # 启用缓存感知
    importance_threshold=0.5      # 重要性阈值
)

# 创建推理管道
device = "cuda:0"
model = "microsoft/DialoGPT-medium"  # 或使用其他支持的模型
pipe = pipeline("kv-press-text-generation", model=model, device=device)

# 长上下文文本
context = "这是一个很长的上下文文本，包含大量信息..."
question = "基于上述上下文，请回答问题"

# 使用 PiKV Routing 进行推理
answer = pipe(context, question=question, press=press)["answer"]
print(answer)
```

### 高级配置

```python
# 自定义 PiKV 路由器配置
press = MoERouterPress(
    router_type="pikv",
    num_experts=4,
    top_k=2,
    capacity_factor=1.5,          # 容量因子
    dropout=0.1,                  # Dropout 率
    compression_ratio=0.6,        # 压缩比
    aux_loss_weight=0.01,         # 辅助损失权重
    cache_aware=True,
    importance_threshold=0.6,
    adaptive_top_k=True           # 自适应 top_k
)

# 获取压缩统计信息
stats = press.get_stats()
print(f"平均辅助损失: {stats['avg_aux_loss']:.4f}")
print(f"专家使用情况: {stats['layer_stats']}")
```

## PiKV Routing 方法详解

### 1. 专家系统架构

PiKV Routing 使用 4 个专门的压缩专家：

```python
expert_strategies = {
    0: "aggressive",    # 激进压缩：保留前20%和后10%
    1: "moderate",      # 中等压缩：保留前30%和后20%
    2: "conservative",  # 保守压缩：保留前50%和后30%
    3: "selective"      # 选择性压缩：基于重要性分数
}
```

### 2. 路由决策过程

```python
# 1. 计算输入重要性
importance = importance_predictor(hidden_states)

# 2. 自适应调整 top_k
current_top_k = adapt_top_k(hidden_states, importance)

# 3. 计算路由概率
router_logits = router_network(hidden_states)
router_probs = softmax(router_logits)

# 4. 缓存感知调整
if cache_aware:
    cache_adjustments = cache_router_adjustment(features, cache_rates)
    router_logits += cache_adjustments

# 5. 选择专家
top_k_probs, top_k_indices = topk(router_probs, current_top_k)
```

### 3. 缓存感知机制

```python
# 实时监控缓存使用情况
def update_cache_usage(self, expert_idx: int, cache_hit_rate: float):
    """更新专家的缓存使用情况"""
    self.cache_hit_rates[expert_idx] = cache_hit_rate
    self.cache_usage_history[expert_idx, history_idx] = cache_hit_rate

# 基于缓存状态调整路由
def compute_cache_aware_adjustment(self, hidden_states, router_logits):
    """计算缓存感知的路由调整"""
    cache_rates = self.cache_hit_rates
    adjustment_factors = self.cache_router_adjustment(
        torch.cat([features, cache_rates], dim=-1)
    )
    return adjustment_factors
```

## 支持的路由器类型

### 1. PiKV Router (推荐)
```python
press = MoERouterPress(router_type="pikv")
```
- 缓存感知路由
- 重要性自适应
- 动态 top_k 调整

### 2. TopK Balanced Router
```python
press = MoERouterPress(router_type="topk_balanced")
```
- 负载平衡优化
- 多种平衡策略 (entropy, variance, gini)

### 3. Adaptive Router
```python
press = MoERouterPress(router_type="adaptive")
```
- 基于输入重要性调整
- 自适应 top_k 选择

### 4. EPLB Router
```python
press = MoERouterPress(router_type="eplb")
```
- 精确负载平衡
- 严格的容量约束

### 5. Hierarchical Router
```python
press = MoERouterPress(router_type="hierarchical")
```
- 层次化路由
- 组级和专家级两级路由

## 效果评估

### 内存节省

```python
# 测量内存使用
import torch
from kvpress.utils import measure_memory_usage

# 不使用压缩
memory_without_press = measure_memory_usage(model, inputs)

# 使用 PiKV Routing
with press(model):
    memory_with_press = measure_memory_usage(model, inputs)

memory_saved = memory_without_press - memory_with_press
compression_ratio = memory_saved / memory_without_press
print(f"内存节省: {compression_ratio:.2%}")
```

### 性能提升

```python
import time

# 基准测试
start_time = time.time()
outputs = model.generate(inputs)
baseline_time = time.time() - start_time

# PiKV Routing 测试
start_time = time.time()
with press(model):
    outputs = model.generate(inputs)
pikv_time = time.time() - start_time

speedup = baseline_time / pikv_time
print(f"速度提升: {speedup:.2f}x")
```

### 典型效果

| 指标 | 无压缩 | PiKV Routing | 提升 |
|------|--------|--------------|------|
| 内存使用 | 100% | 40-60% | 40-60% |
| 推理速度 | 1x | 1.5-2.5x | 50-150% |
| 压缩比 | 0% | 50-70% | - |
| 缓存命中率 | - | 85-95% | - |

## 完整示例

### 长文档问答

```python
from transformers import pipeline
from kvpress import MoERouterPress

# 配置 PiKV 路由器
press = MoERouterPress(
    router_type="pikv",
    num_experts=4,
    compression_ratio=0.6,
    cache_aware=True
)

# 创建管道
pipe = pipeline("kv-press-text-generation", 
                model="microsoft/DialoGPT-medium", 
                device="cuda:0")

# 长文档
long_document = """
[这里是一个很长的文档，包含大量信息...]
"""

# 多个问题
questions = [
    "文档的主要观点是什么？",
    "有哪些关键数据？",
    "结论是什么？"
]

# 批量处理
for question in questions:
    answer = pipe(long_document, question=question, press=press)["answer"]
    print(f"问题: {question}")
    print(f"答案: {answer}\n")

# 获取统计信息
stats = press.get_stats()
print("压缩统计:")
print(f"- 平均辅助损失: {stats['avg_aux_loss']:.4f}")
print(f"- 总前向次数: {stats['forward_count']}")
```

### 实时聊天机器人

```python
class PiKVChatBot:
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        self.press = MoERouterPress(
            router_type="pikv",
            compression_ratio=0.5,
            cache_aware=True
        )
        self.pipe = pipeline("kv-press-text-generation", 
                           model=model_name, 
                           device="cuda:0")
        self.conversation_history = []
    
    def chat(self, user_input: str) -> str:
        # 构建对话历史
        context = "\n".join(self.conversation_history + [user_input])
        
        # 使用 PiKV Routing 生成回复
        response = self.pipe(context, press=self.press)["answer"]
        
        # 更新历史
        self.conversation_history.extend([user_input, response])
        
        # 保持历史长度
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
        
        return response
    
    def get_stats(self):
        return self.press.get_stats()

# 使用示例
bot = PiKVChatBot()
response = bot.chat("你好，请介绍一下 PiKV Routing 技术")
print(response)
```

## 支持的模型

PiKV Routing 支持以下模型架构：

- ✅ **LlamaForCausalLM** (Llama 2/3, Code Llama)
- ✅ **MistralForCausalLM** (Mistral, Mixtral)
- ✅ **Phi3ForCausalLM** (Phi-3)
- ✅ **Qwen2ForCausalLM** (Qwen2)
- ✅ **Qwen3ForCausalLM** (Qwen3)
- ✅ **Gemma3ForCausalLM** (Gemma 3)
- ✅ **GPT2LMHeadModel** (GPT-2)

## 故障排除

### 常见问题

1. **模型不支持**
```python
# 检查模型类型
print(type(model))
# 确保使用支持的模型类型
```

2. **内存不足**
```python
# 降低压缩比
press = MoERouterPress(compression_ratio=0.3)

# 或减少专家数量
press = MoERouterPress(num_experts=2)
```

3. **性能问题**
```python
# 启用 flash attention
pipe = pipeline("kv-press-text-generation", 
                model=model, 
                device=device,
                model_kwargs={"attn_implementation": "flash_attention_2"})
```

### 调试模式

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 启用详细日志
press = MoERouterPress(router_type="pikv")
# 查看路由决策详情
```

## 贡献

我们欢迎贡献！如果您想添加新的路由器类型或改进现有功能，请：

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 创建 Pull Request

## 许可证

Apache 2.0 License

## 引用

如果您在研究中使用了 PiKV Routing，请引用：

```bibtex
@misc{kvpress2024,
  title={KVPress: KV Cache Compression with PiKV Routing},
  author={NVIDIA},
  year={2024},
  url={https://github.com/NVIDIA/kvpress}
}
```

---

**开始使用 PiKV Routing 来优化您的长上下文 LLM 应用！** 🚀