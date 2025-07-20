#!/usr/bin/env python3
"""
Quick Test cho Continued Model
"""

import torch
from unsloth import FastLanguageModel

def main():
    print("🚀 Quick Test Continued Model")
    print("=" * 40)
    
    # Load model
    print("📥 Loading...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        "./gemma_multimodal_finetuned_continued",
        max_seq_length=512,
        dtype=None,
        load_in_4bit=True,
    )
    
    FastLanguageModel.for_inference(model)
    print("✅ Loaded!")
    
    # Handle tokenizer
    if hasattr(tokenizer, 'tokenizer'):
        actual_tokenizer = tokenizer.tokenizer
    else:
        actual_tokenizer = tokenizer
    
    # Test one question
    question = "Tính 2 + 3 = ?"
    print(f"\n❓ Question: {question}")
    
    # Format
    prompt = f"<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"
    
    # Generate
    inputs = actual_tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            temperature=0.1,
            do_sample=True,
            pad_token_id=actual_tokenizer.eos_token_id,
        )
    
    # Decode
    response = actual_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract answer
    if "<start_of_turn>model\n" in response:
        answer = response.split("<start_of_turn>model\n")[-1]
    else:
        answer = response.replace(prompt, "")
    
    print(f"🤖 Answer: {answer}")
    print("\n🎉 Done!")

if __name__ == "__main__":
    main() 