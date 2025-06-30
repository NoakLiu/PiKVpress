#!/usr/bin/env python3
"""
Simple Comparison Experiment: EPLB Routing + Duo Attention vs Duo Attention vs No Compression

This script provides a simple comparison between:
1. EPLB Routing + Duo Attention (MoE router with duo attention)
2. Duo Attention only  
3. No KV cache compression (baseline)

Run this script to see the performance differences.
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from kvpress import MoERouterPress, DuoAttentionPress, ComposedPress, BasePress

class NoCompressionPress(BasePress):
    """Baseline press that doesn't compress KV cache"""
    def __init__(self):
        super().__init__()
        self.compression_ratio = 0.0
    
    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        return keys, values

def measure_memory():
    """Measure current GPU memory usage in GB"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024**3)
    return 0.0

def reset_memory_stats():
    """Reset memory statistics"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def run_experiment(method_name, press, model, tokenizer, context, question, max_new_tokens=50):
    """Run a single experiment"""
    print(f"\nRunning {method_name}...")
    
    # Reset memory stats
    reset_memory_stats()
    
    # Measure inference time and memory
    start_time = time.time()
    
    try:
        if method_name == "No Compression":
            # Direct model call without press
            inputs = tokenizer(context + " " + question, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id
                )
            generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        else:
            # Use pipeline with press
            pipe = pipeline("kv-press-text-generation", model=model, tokenizer=tokenizer, device=model.device)
            result = pipe(context, question=question, press=press)
            generated_text = result["answer"]
        
        end_time = time.time()
        
    except Exception as e:
        print(f"Error in experiment {method_name}: {e}")
        return None
    
    # Calculate metrics
    inference_time = end_time - start_time
    memory_usage = measure_memory()
    tokens_per_second = max_new_tokens / inference_time
    
    print(f"  Memory Usage: {memory_usage:.2f} GB")
    print(f"  Inference Time: {inference_time:.2f} seconds")
    print(f"  Speed: {tokens_per_second:.1f} tokens/sec")
    print(f"  Generated: {generated_text[:100]}...")
    
    return {
        'method': method_name,
        'memory_gb': memory_usage,
        'time_seconds': inference_time,
        'tokens_per_second': tokens_per_second
    }

def main():
    """Main comparison function"""
    print("="*80)
    print("EPLB ROUTING + DUO ATTENTION vs DUO ATTENTION vs NO COMPRESSION")
    print("="*80)
    
    # Configuration
    model_name = "microsoft/DialoGPT-medium"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compression_ratio = 0.5
    
    # Load model and tokenizer
    print(f"\nLoading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
        use_safetensors=True  # 使用safetensors格式避免PyTorch版本问题
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test context and question
    context = """
    Artificial Intelligence (AI) and Machine Learning (ML) have revolutionized the way we approach problem-solving in the digital age. 
    These technologies encompass a wide range of techniques and methodologies that enable computers to learn from data and make intelligent decisions.
    
    Machine Learning, a subset of AI, focuses on developing algorithms that can learn patterns from data without being explicitly programmed. 
    There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.
    
    Supervised learning involves training models on labeled data, where the algorithm learns to map inputs to known outputs. 
    Common applications include image classification, speech recognition, and natural language processing.
    
    Unsupervised learning deals with unlabeled data, where the algorithm discovers hidden patterns and structures. 
    This includes clustering, dimensionality reduction, and association rule learning.
    
    Reinforcement learning is based on the concept of learning through interaction with an environment. 
    The agent receives rewards or penalties for its actions and learns to maximize cumulative rewards over time.
    
    Deep Learning, a subset of machine learning, uses artificial neural networks with multiple layers to model complex patterns. 
    Convolutional Neural Networks (CNNs) are particularly effective for image processing tasks, while Recurrent Neural Networks (RNNs) 
    and their variants like LSTM and GRU are well-suited for sequential data processing.
    
    Natural Language Processing (NLP) is a field that combines AI, linguistics, and computer science to enable computers to understand, 
    interpret, and generate human language. Modern NLP systems use transformer architectures like BERT, GPT, and T5 to achieve 
    state-of-the-art performance on various language tasks.
    """
    
    question = "What are the main applications of machine learning?"
    
    print(f"\nContext length: {len(context.split())} words")
    print(f"Question: {question}")
    
    results = []
    
    # Method 1: No Compression (Baseline)
    no_compression_press = NoCompressionPress()
    result1 = run_experiment("No Compression", no_compression_press, model, tokenizer, context, question)
    if result1:
        results.append(result1)
    
    # Method 2: Duo Attention Only
    duo_press = DuoAttentionPress(head_compression_ratio=compression_ratio)
    result2 = run_experiment("Duo Attention", duo_press, model, tokenizer, context, question)
    if result2:
        results.append(result2)
    
    # Method 3: EPLB Routing + Duo Attention
    eplb_press = MoERouterPress(
        router_type="eplb",
        num_experts=4,
        top_k=2,
        compression_ratio=compression_ratio * 0.7  # Slightly less aggressive
    )
    duo_press_combined = DuoAttentionPress(head_compression_ratio=compression_ratio * 0.3)
    combined_press = ComposedPress([eplb_press, duo_press_combined])
    
    result3 = run_experiment("EPLB + Duo Attention", combined_press, model, tokenizer, context, question)
    if result3:
        results.append(result3)
    
    # Print comparison summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Method':<20} {'Memory(GB)':<12} {'Time(s)':<10} {'Speed(tok/s)':<12}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['method']:<20} {result['memory_gb']:<12.2f} {result['time_seconds']:<10.2f} {result['tokens_per_second']:<12.1f}")
    
    # Calculate improvements
    if len(results) >= 3:
        baseline = results[0]
        duo = results[1]
        combined = results[2]
        
        print(f"\nRelative Improvements:")
        print(f"Duo Attention vs Baseline:")
        print(f"  Memory reduction: {((baseline['memory_gb'] - duo['memory_gb']) / baseline['memory_gb'] * 100):.1f}%")
        print(f"  Speed improvement: {((duo['tokens_per_second'] / baseline['tokens_per_second'] - 1) * 100):.1f}%")
        
        print(f"\nEPLB + Duo Attention vs Baseline:")
        print(f"  Memory reduction: {((baseline['memory_gb'] - combined['memory_gb']) / baseline['memory_gb'] * 100):.1f}%")
        print(f"  Speed improvement: {((combined['tokens_per_second'] / baseline['tokens_per_second'] - 1) * 100):.1f}%")
        
        print(f"\nEPLB + Duo Attention vs Duo Attention:")
        print(f"  Additional memory reduction: {((duo['memory_gb'] - combined['memory_gb']) / duo['memory_gb'] * 100):.1f}%")
        print(f"  Additional speed improvement: {((combined['tokens_per_second'] / duo['tokens_per_second'] - 1) * 100):.1f}%")

if __name__ == "__main__":
    main() 