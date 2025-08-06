#!/usr/bin/env python3
"""
Quick test để verify fixes hoạt động
"""

import torch
from datasets import load_dataset  
from unsloth import FastVisionModel
from fixed_data_collator import FixedGemma3nDataCollator

def quick_test():
    """Quick test với 1 sample"""
    print("⚡ Quick Test - Fixed Gemma3n Collator")
    print("="*50)
    
    # Load model and processor
    print("📥 Loading model...")
    model, processor = FastVisionModel.from_pretrained(
        "unsloth/gemma-3n-E4B",
        load_in_4bit=True,
    )
    
    # Disable gradient checkpointing
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_disable()
        print("🔧 Gradient checkpointing disabled")
    
    # Load 1 sample
    print("📥 Loading dataset sample...")
    dataset = load_dataset("ngohongthai/exam-sixth_grade-instruct-dataset", split="train")
    
    # Process sample to conversation format
    sample = dataset[0]
    
    # Simple conversion to conversation format
    conversation = [
        {
            "role": "user",
            "content": [{"type": "text", "text": sample["question"]}]
        },
        {
            "role": "assistant", 
            "content": [{"type": "text", "text": sample["solution"]}]
        }
    ]
    
    test_example = {"conversations": conversation}
    
    print("🧪 Testing fixed collator...")
    
    # Test fixed collator
    collator = FixedGemma3nDataCollator(processor, handle_text_only=True)
    
    try:
        batch = collator([test_example])
        print("✅ SUCCESS! Batch created:")
        print(f"  Keys: {batch.keys()}")
        print(f"  Input IDs shape: {batch['input_ids'].shape}")
        print(f"  Pixel values shape: {batch['pixel_values'].shape}")
        
        # Test forward pass
        print("\n🧪 Testing forward pass...")
        model.eval()
        with torch.no_grad():
            outputs = model(**batch)
            print("✅ Forward pass successful!")
            if outputs.loss is not None:
                print(f"  Loss: {outputs.loss.item():.4f}")
            else:
                print("  No loss (inference mode)")
                
        print("\n🎉 ALL TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_test()
    if success:
        print("\n✅ Ready for full training!")
    else:
        print("\n❌ Need more debugging...")