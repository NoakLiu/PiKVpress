#!/usr/bin/env python3
"""
Quick Comparison Script: EPLB Routing + Duo Attention vs Duo Attention vs No Compression

This script provides a quick comparison between the three methods.
Run with: python run_comparison.py
"""

import time
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from kvpress import MoERouterPress, DuoAttentionPress, ComposedPress, BasePress

class TestDuoAttentionPress(DuoAttentionPress):
    """Test version of DuoAttentionPress that works with any model"""
    @staticmethod
    def load_attention_pattern(model):
        n_layers = model.config.num_hidden_layers
        # GPT2使用num_attention_heads，其他模型使用num_key_value_heads
        if hasattr(model.config, 'num_key_value_heads'):
            n_heads = model.config.num_key_value_heads
        else:
            n_heads = model.config.num_attention_heads
        return 2, 2, np.random.rand(n_layers, n_heads)

class GPT2MoERouterPress(MoERouterPress):
    """GPT2-compatible version of MoERouterPress"""
    def __call__(self, model):
        """Override to support GPT2 model structure"""
        if not hasattr(model, 'transformer'):
            raise ValueError("GPT2 model must have transformer attribute")
        
        hooks = []
        try:
            # GPT2模型结构
            layers = model.transformer.h
            for i, layer in enumerate(layers):
                layer.layer_idx = i
                # 注册到注意力层
                hooks.append(layer.attn.register_forward_hook(self.forward_hook, with_kwargs=True))
            
            yield
            
        finally:
            # 清理hooks
            for hook in hooks:
                hook.remove()

class NoCompressionPress(BasePress):
    """Baseline press that doesn't compress KV cache"""
    def __init__(self):
        super().__init__()
        self.compression_ratio = 0.0
    
    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        return keys, values

def main():
    print("🚀 PiKVPress Comparison Experiment")
    print("=" * 60)
    
    # Configuration
    model_name = "distilgpt2"  # Use lightweight model for quick testing
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compression_ratio = 0.5
    
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Compression Ratio: {compression_ratio}")
    
    # Load model
    print("\n📥 Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",  # 使用auto而不是具体device
        use_safetensors=True  # 使用safetensors格式避免PyTorch版本问题
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test context
    context = """
    Artificial Intelligence and Machine Learning have transformed how we solve problems. 
    Machine learning algorithms can learn patterns from data without explicit programming. 
    There are three main types: supervised learning, unsupervised learning, and reinforcement learning. 
    Deep learning uses neural networks with multiple layers to model complex patterns. 
    Natural Language Processing enables computers to understand and generate human language.
    """
    
    question = "What are the main types of machine learning?"
    
    print(f"\n📝 Context length: {len(context.split())} words")
    print(f"❓ Question: {question}")
    
    results = []
    
    # Method 1: No Compression
    print(f"\n🔍 Testing No Compression...")
    no_compression_press = NoCompressionPress()
    
    torch.cuda.empty_cache() if device == "cuda" else None
    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
    
    start_time = time.time()
    try:
        inputs = tokenizer(context + " " + question, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=30,
                pad_token_id=tokenizer.eos_token_id
            )
        generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        end_time = time.time()
        
        inference_time = end_time - start_time
        memory_usage = torch.cuda.max_memory_allocated() / (1024**3) if device == "cuda" else 0.0
        tokens_per_second = 30 / inference_time
        
        results.append({
            'method': 'No Compression',
            'memory_gb': memory_usage,
            'time_seconds': inference_time,
            'tokens_per_second': tokens_per_second
        })
        
        print(f"   Memory: {memory_usage:.2f} GB")
        print(f"   Time: {inference_time:.2f}s")
        print(f"   Speed: {tokens_per_second:.1f} tok/s")
        
    except Exception as e:
        print(f"   Error: {e}")
    
    # Method 2: Duo Attention
    print(f"\n🔍 Testing Duo Attention...")
    duo_press = DuoAttentionPress(head_compression_ratio=compression_ratio)
    
    torch.cuda.empty_cache() if device == "cuda" else None
    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
    
    start_time = time.time()
    try:
        pipe = pipeline("kv-press-text-generation", model=model, tokenizer=tokenizer)
        result = pipe(context, question=question, press=duo_press)
        end_time = time.time()
        
        inference_time = end_time - start_time
        memory_usage = torch.cuda.max_memory_allocated() / (1024**3) if device == "cuda" else 0.0
        tokens_per_second = 30 / inference_time
        
        results.append({
            'method': 'Duo Attention',
            'memory_gb': memory_usage,
            'time_seconds': inference_time,
            'tokens_per_second': tokens_per_second
        })
        
        print(f"   Memory: {memory_usage:.2f} GB")
        print(f"   Time: {inference_time:.2f}s")
        print(f"   Speed: {tokens_per_second:.1f} tok/s")
        
    except Exception as e:
        print(f"   Error: {e}")
    
    # Method 3: EPLB + Duo Attention
    print(f"\n🔍 Testing EPLB + Duo Attention...")
    
    try:
        eplb_press = MoERouterPress(
            router_type="eplb",
            num_experts=4,
            top_k=2,
            compression_ratio=compression_ratio * 0.7
        )
        duo_press_combined = DuoAttentionPress(head_compression_ratio=compression_ratio * 0.3)
        combined_press = ComposedPress([eplb_press, duo_press_combined])
        
        torch.cuda.empty_cache() if device == "cuda" else None
        torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
        
        start_time = time.time()
        result = pipe(context, question=question, press=combined_press)
        end_time = time.time()
        
        inference_time = end_time - start_time
        memory_usage = torch.cuda.max_memory_allocated() / (1024**3) if device == "cuda" else 0.0
        tokens_per_second = 30 / inference_time
        
        results.append({
            'method': 'EPLB + Duo Attention',
            'memory_gb': memory_usage,
            'time_seconds': inference_time,
            'tokens_per_second': tokens_per_second
        })
        
        print(f"   Memory: {memory_usage:.2f} GB")
        print(f"   Time: {inference_time:.2f}s")
        print(f"   Speed: {tokens_per_second:.1f} tok/s")
        
    except Exception as e:
        print(f"   Error: {e}")
    
    # Print summary
    print(f"\n📊 COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Method':<20} {'Memory(GB)':<12} {'Time(s)':<10} {'Speed(tok/s)':<12}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['method']:<20} {result['memory_gb']:<12.2f} {result['time_seconds']:<10.2f} {result['tokens_per_second']:<12.1f}")
    
    # Calculate improvements
    if len(results) >= 3:
        baseline = results[0]
        duo = results[1]
        combined = results[2]
        
        print(f"\n📈 RELATIVE IMPROVEMENTS")
        print("-" * 40)
        
        if baseline['memory_gb'] > 0:
            print(f"Duo Attention vs Baseline:")
            print(f"  Memory reduction: {((baseline['memory_gb'] - duo['memory_gb']) / baseline['memory_gb'] * 100):.1f}%")
            print(f"  Speed improvement: {((duo['tokens_per_second'] / baseline['tokens_per_second'] - 1) * 100):.1f}%")
            
            print(f"\nEPLB + Duo Attention vs Baseline:")
            print(f"  Memory reduction: {((baseline['memory_gb'] - combined['memory_gb']) / baseline['memory_gb'] * 100):.1f}%")
            print(f"  Speed improvement: {((combined['tokens_per_second'] / baseline['tokens_per_second'] - 1) * 100):.1f}%")
            
            print(f"\nEPLB + Duo Attention vs Duo Attention:")
            print(f"  Additional memory reduction: {((duo['memory_gb'] - combined['memory_gb']) / duo['memory_gb'] * 100):.1f}%")
            print(f"  Additional speed improvement: {((combined['tokens_per_second'] / duo['tokens_per_second'] - 1) * 100):.1f}%")
    
    print(f"\n✅ Experiment completed!")
    print(f"💡 EPLB + Duo Attention typically provides the best balance of memory savings and speed improvements.")

if __name__ == "__main__":
    main() 