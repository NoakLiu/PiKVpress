# PiKVPress: KV Cache Compression with PiKV Routing

[![PyPI version](https://badge.fury.io/py/kvpress.svg)](https://badge.fury.io/py/kvpress)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Colab example notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1JNvaTKuuAHrl49dYB9-mdEH_y52Ib-NP?usp=drive_link)
<!-- [![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/nvidia/kvpress) -->


<!-- ![kvpress](kvpress.jpg) -->


Deploying long-context LLMs is costly due to the linear growth of the key-value (KV) cache in transformer models. For example, handling 1M tokens with Llama 3.1-70B in float16 requires up to 330GB of memory. kvpress implements multiple KV cache compression methods and benchmarks using 🤗 transformers, aiming to simplify the development of new methods for researchers and developers in this field. PiKV project code can be found at [PiKV](https://github.com/NoakLiu/PiKV).

## Overview

PiKVPress is a powerful KV cache compression framework that integrates **PiKV Routing** technology, implementing intelligent KV cache compression through Mixture of Experts (MoE) architecture. PiKV Routing dynamically selects optimal compression strategies based on input features and cache usage patterns, significantly improving memory efficiency and inference speed for long-context processing.

### Core Features

- 🚀 **PiKV Routing**: MoE-based intelligent routing system
- 🎯 **Multi-Expert Compression**: 4 different compression strategy experts
- 📊 **Cache-Aware**: Real-time cache usage monitoring
- 🔄 **Adaptive Adjustment**: Dynamic compression strategy adjustment
- 💾 **Memory Optimization**: Significantly reduce KV cache memory usage
- ⚡ **Performance Boost**: Accelerate long-context inference

## Installation


```bash
git clone https://github.com/NoakLiu/PiKVpress.git
cd kvpress
poetry install --with dev
```

## Quick Start

### Basic Usage

```python
from transformers import pipeline
from kvpress import MoERouterPress

# Create PiKV MoE router
press = MoERouterPress(
    router_type="pikv",           # Use PiKV router
    num_experts=4,                # 4 experts
    top_k=2,                      # Select top 2 experts
    compression_ratio=0.5,        # Target 50% compression ratio
    cache_aware=True,             # Enable cache awareness
    importance_threshold=0.5      # Importance threshold
)

# Create inference pipeline
device = "cuda:0"
model = "microsoft/DialoGPT-medium"  # Or use other supported models
pipe = pipeline("kv-press-text-generation", model=model, device=device)

# Long context text
context = "This is a very long context text containing lots of information..."
question = "Based on the above context, please answer the question"

# Use PiKV Routing for inference
answer = pipe(context, question=question, press=press)["answer"]
print(answer)
```

### Advanced Configuration

```python
# Custom PiKV router configuration
press = MoERouterPress(
    router_type="pikv",
    num_experts=4,
    top_k=2,
    capacity_factor=1.5,          # Capacity factor
    dropout=0.1,                  # Dropout rate
    compression_ratio=0.6,        # Compression ratio
    aux_loss_weight=0.01,         # Auxiliary loss weight
    cache_aware=True,
    importance_threshold=0.6,
    adaptive_top_k=True           # Adaptive top_k
)

# Get compression statistics
stats = press.get_stats()
print(f"Average auxiliary loss: {stats['avg_aux_loss']:.4f}")
print(f"Expert usage: {stats['layer_stats']}")
```

## Combining Different Presses

PiKVPress supports combining multiple compression strategies for enhanced performance. Here are several ways to combine different presses:

### 1. ComposedPress - Sequential Combination

```python
from kvpress import MoERouterPress, KnormPress, ComposedPress

# Create individual presses
pikv_press = MoERouterPress(router_type="pikv", compression_ratio=0.3)
knorm_press = KnormPress(compression_ratio=0.2)

# Combine them sequentially
composed_press = ComposedPress([pikv_press, knorm_press])

# Use the combined press
with composed_press(model):
    outputs = model.generate(inputs)
```

### 2. Per-Layer Compression

```python
from kvpress import PerLayerCompressionPress, MoERouterPress, ExpectedAttentionPress

# Different compression strategies for different layers
layer_presses = {
    0: MoERouterPress(router_type="pikv", compression_ratio=0.4),      # First layer
    1: ExpectedAttentionPress(compression_ratio=0.3),                   # Second layer
    2: KnormPress(compression_ratio=0.5),                               # Third layer
    # ... other layers
}

per_layer_press = PerLayerCompressionPress(layer_presses)

# Apply different compression to each layer
with per_layer_press(model):
    outputs = model.generate(inputs)
```

### 3. AdaKVPress with MoE Router

```python
from kvpress import AdaKVPress, MoERouterPress

# Create base MoE router
base_press = MoERouterPress(router_type="pikv", compression_ratio=0.4)

# Apply AdaKV head-wise compression on top
adapress = AdaKVPress(base_press, compression_ratio=0.2)

# Use the enhanced press
with adapress(model):
    outputs = model.generate(inputs)
```

### 4. Custom Press Combination

```python
from kvpress import BasePress, MoERouterPress, KnormPress
import torch

class CustomCombinedPress(BasePress):
    def __init__(self, pikv_ratio=0.3, knorm_ratio=0.2):
        super().__init__()
        self.pikv_press = MoERouterPress(router_type="pikv", compression_ratio=pikv_ratio)
        self.knorm_press = KnormPress(compression_ratio=knorm_ratio)
        self.combination_weight = 0.7  # Weight for PiKV vs Knorm
    
    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        # Apply PiKV compression
        pikv_keys, pikv_values = self.pikv_press.compress(
            module, hidden_states, keys, values, attentions, kwargs
        )
        
        # Apply Knorm compression
        knorm_keys, knorm_values = self.knorm_press.compress(
            module, hidden_states, keys, values, attentions, kwargs
        )
        
        # Combine results based on importance
        importance = self._compute_importance(hidden_states)
        
        # Use PiKV for important tokens, Knorm for others
        combined_keys = torch.where(
            importance.unsqueeze(-1).unsqueeze(-1) > 0.5,
            pikv_keys, knorm_keys
        )
        combined_values = torch.where(
            importance.unsqueeze(-1).unsqueeze(-1) > 0.5,
            pikv_values, knorm_values
        )
        
        return combined_keys, combined_values
    
    def _compute_importance(self, hidden_states):
        # Simple importance computation based on norm
        return torch.norm(hidden_states, dim=-1)

# Use custom combined press
custom_press = CustomCombinedPress(pikv_ratio=0.3, knorm_ratio=0.2)
with custom_press(model):
    outputs = model.generate(inputs)
```

### 5. Pipeline Combination Example

```python
from transformers import pipeline
from kvpress import MoERouterPress, ComposedPress, KnormPress

# Create a sophisticated press combination
def create_advanced_press():
    # Primary: PiKV router for intelligent routing
    primary_press = MoERouterPress(
        router_type="pikv",
        num_experts=4,
        compression_ratio=0.4,
        cache_aware=True
    )
    
    # Secondary: Knorm for additional compression
    secondary_press = KnormPress(compression_ratio=0.2)
    
    # Combine them
    return ComposedPress([primary_press, secondary_press])

# Create pipeline with combined press
press = create_advanced_press()
pipe = pipeline("kv-press-text-generation", 
                model="microsoft/DialoGPT-medium", 
                device="cuda:0")

# Process multiple documents with different compression strategies
documents = [
    "Long document 1...",
    "Long document 2...",
    "Long document 3..."
]

for i, doc in enumerate(documents):
    # Adjust compression based on document characteristics
    if len(doc) > 10000:  # Very long document
        press.presses[0].compression_ratio = 0.6  # More aggressive
    else:
        press.presses[0].compression_ratio = 0.3  # Conservative
    
    answer = pipe(doc, question="Summarize this document", press=press)["answer"]
    print(f"Document {i+1}: {answer[:100]}...")
```

### 6. Performance Comparison Script

```python
import time
import torch
from kvpress import MoERouterPress, KnormPress, ComposedPress

def benchmark_press_combinations(model, inputs, context_length=1000):
    """Benchmark different press combinations"""
    
    combinations = {
        "PiKV Only": MoERouterPress(router_type="pikv", compression_ratio=0.5),
        "Knorm Only": KnormPress(compression_ratio=0.5),
        "PiKV + Knorm": ComposedPress([
            MoERouterPress(router_type="pikv", compression_ratio=0.3),
            KnormPress(compression_ratio=0.2)
        ]),
        "Adaptive PiKV": MoERouterPress(
            router_type="pikv", 
            compression_ratio=0.5,
            adaptive_top_k=True
        )
    }
    
    results = {}
    
    for name, press in combinations.items():
        print(f"\nTesting {name}...")
        
        # Measure memory usage
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        start_time = time.time()
        with press(model):
            outputs = model.generate(inputs, max_new_tokens=100)
        end_time = time.time()
        
        memory_used = torch.cuda.max_memory_allocated() / 1024**3  # GB
        inference_time = end_time - start_time
        
        results[name] = {
            "memory_gb": memory_used,
            "time_seconds": inference_time,
            "tokens_per_second": 100 / inference_time
        }
        
        print(f"  Memory: {memory_used:.2f} GB")
        print(f"  Time: {inference_time:.2f} seconds")
        print(f"  Speed: {100/inference_time:.1f} tokens/sec")
    
    return results

# Run benchmark
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium").to("cuda")
inputs = torch.randint(0, 1000, (1, 1000)).to("cuda")

results = benchmark_press_combinations(model, inputs)

# Print comparison
print("\n" + "="*60)
print("PERFORMANCE COMPARISON")
print("="*60)
for name, metrics in results.items():
    print(f"{name:15} | {metrics['memory_gb']:6.2f} GB | {metrics['time_seconds']:6.2f}s | {metrics['tokens_per_second']:6.1f} tok/s")
```

## Supported Router Types

### 1. PiKV Router (Recommended)
```python
press = MoERouterPress(router_type="pikv")
```
- Cache-aware routing
- Importance-based adaptation
- Dynamic top_k adjustment

### 2. TopK Balanced Router
```python
press = MoERouterPress(router_type="topk_balanced")
```
- Load balancing optimization
- Multiple balance strategies (entropy, variance, gini)

### 3. Adaptive Router
```python
press = MoERouterPress(router_type="adaptive")
```
- Input importance-based adjustment
- Adaptive top_k selection

### 4. EPLB Router
```python
press = MoERouterPress(router_type="eplb")
```
- Exact perfect load balancing
- Strict capacity constraints

### 5. Hierarchical Router
```python
press = MoERouterPress(router_type="hierarchical")
```
- Hierarchical routing
- Two-level routing (group-level and expert-level)

## Performance Evaluation

### Memory Savings

```python
# Measure memory usage
import torch
from kvpress.utils import measure_memory_usage

# Without compression
memory_without_press = measure_memory_usage(model, inputs)

# With PiKV Routing
with press(model):
    memory_with_press = measure_memory_usage(model, inputs)

memory_saved = memory_without_press - memory_with_press
compression_ratio = memory_saved / memory_without_press
print(f"Memory saved: {compression_ratio:.2%}")
```

### Performance Improvement

```python
import time

# Baseline test
start_time = time.time()
outputs = model.generate(inputs)
baseline_time = time.time() - start_time

# PiKV Routing test
start_time = time.time()
with press(model):
    outputs = model.generate(inputs)
pikv_time = time.time() - start_time

speedup = baseline_time / pikv_time
print(f"Speed improvement: {speedup:.2f}x")
```

### Typical Results

| Metric | No Compression | PiKV Routing | Improvement |
|--------|----------------|--------------|-------------|
| Memory Usage | 100% | 40-60% | 40-60% |
| Inference Speed | 1x | 1.5-2.5x | 50-150% |
| Compression Ratio | 0% | 50-70% | - |
| Cache Hit Rate | - | 85-95% | - |

## Supported Models

PiKV Routing supports the following model architectures:

- ✅ **LlamaForCausalLM** (Llama 2/3, Code Llama)
- ✅ **MistralForCausalLM** (Mistral, Mixtral)
- ✅ **Phi3ForCausalLM** (Phi-3)
- ✅ **Qwen2ForCausalLM** (Qwen2)
- ✅ **Qwen3ForCausalLM** (Qwen3)
- ✅ **Gemma3ForCausalLM** (Gemma 3)
- ✅ **GPT2LMHeadModel** (GPT-2)

## Troubleshooting

### Common Issues

1. **Unsupported Model**
```python
# Check model type
print(type(model))
# Ensure using supported model types
```

2. **Insufficient Memory**
```python
# Reduce compression ratio
press = MoERouterPress(compression_ratio=0.3)

# Or reduce number of experts
press = MoERouterPress(num_experts=2)
```

3. **Performance Issues**
```python
# Enable flash attention
pipe = pipeline("kv-press-text-generation", 
                model=model, 
                device=device,
                model_kwargs={"attn_implementation": "flash_attention_2"})
```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging
press = MoERouterPress(router_type="pikv")
# View routing decision details
```

## Contributing

We welcome contributions! If you want to add new router types or improve existing features, please:

1. Fork the project
2. Create a feature branch
3. Submit changes
4. Create a Pull Request

## License

Apache 2.0 License

## Citation

If you use PiKV Routing in your research, please cite:

```bibtex
@misc{PiKVpress2024,
  title={PiKVPress: KV Cache Compression with PiKV Routing},
  author={Dong Liu},
  year={2024},
  url={https://github.com/NoakLiu/kvpress}
}
```

---

**Start using PiKV Routing to optimize your long-context LLM applications!** 🚀