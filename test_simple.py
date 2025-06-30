#!/usr/bin/env python3
"""
Simple test script to verify the fixes
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
        n_layers, n_heads = model.config.num_hidden_layers, model.config.num_key_value_heads
        return 2, 2, np.random.rand(n_layers, n_heads)

class NoCompressionPress(BasePress):
    """Baseline press that doesn't compress KV cache"""
    def __init__(self):
        super().__init__()
        self.compression_ratio = 0.0
    
    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        return keys, values

def main():
    print("🚀 Testing PiKVPress with fixes")
    print("=" * 50)
    
    # Configuration
    model_name = "distilgpt2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    
    # Load model
    print("\n📥 Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
        use_safetensors=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test context
    context = "Artificial Intelligence and Machine Learning have transformed how we solve problems."
    question = "What is machine learning?"
    
    print(f"\n📝 Context: {context}")
    print(f"❓ Question: {question}")
    
    # Test 1: No Compression
    print(f"\n🔍 Testing No Compression...")
    try:
        inputs = tokenizer(context + " " + question, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=20,
                pad_token_id=tokenizer.eos_token_id
            )
        generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"   ✅ Success: {generated_text[:50]}...")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: Duo Attention
    print(f"\n🔍 Testing Duo Attention...")
    try:
        duo_press = TestDuoAttentionPress(head_compression_ratio=0.5)
        pipe = pipeline("kv-press-text-generation", model=model, tokenizer=tokenizer)
        result = pipe(context, question=question, press=duo_press)
        print(f"   ✅ Success: {result['answer'][:50]}...")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: EPLB + Duo Attention
    print(f"\n🔍 Testing EPLB + Duo Attention...")
    try:
        eplb_press = MoERouterPress(
            router_type="eplb",
            num_experts=4,
            top_k=2,
            compression_ratio=0.35
        )
        duo_press_combined = TestDuoAttentionPress(head_compression_ratio=0.15)
        combined_press = ComposedPress([eplb_press, duo_press_combined])
        
        pipe = pipeline("kv-press-text-generation", model=model, tokenizer=tokenizer)
        result = pipe(context, question=question, press=combined_press)
        print(f"   ✅ Success: {result['answer'][:50]}...")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print(f"\n✅ Test completed!")

if __name__ == "__main__":
    main() 