#!/usr/bin/env python3
"""
Quick test to verify the fixes work
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    print("🚀 Quick Test")
    print("=" * 30)
    
    # Load model
    model_name = "distilgpt2"
    print(f"Loading {model_name}...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        use_safetensors=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test basic generation
    print("Testing basic generation...")
    context = "Hello world"
    inputs = tokenizer(context, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=10,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"Generated: {generated_text}")
    
    # Test model structure
    print(f"\nModel type: {type(model)}")
    print(f"Has transformer: {hasattr(model, 'transformer')}")
    if hasattr(model, 'transformer'):
        print(f"Number of layers: {len(model.transformer.h)}")
        print(f"Config num_attention_heads: {model.config.num_attention_heads}")
        print(f"Config num_key_value_heads: {getattr(model.config, 'num_key_value_heads', 'Not available')}")
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    main() 