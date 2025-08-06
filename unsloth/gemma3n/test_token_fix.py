#!/usr/bin/env python3
"""
Test token fix với correct <image_soft_token>
"""

import torch
from datasets import load_dataset  
from unsloth import FastVisionModel
from fixed_data_collator import FixedGemma3nDataCollator

def test_token_fix():
    """Test với <image_soft_token> thay vì <image>"""
    print("🧪 Testing Token Fix - <image_soft_token>")
    print("="*60)
    
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
    
    # Test tokenization
    print("\n🔍 Testing tokenization:")
    test_texts = [
        "<image>",  # Wrong token
        "<image_soft_token>",  # Correct token
        "test <image_soft_token> test"
    ]
    
    for text in test_texts:
        tokens = processor.tokenizer.tokenize(text)
        token_ids = processor.tokenizer.encode(text, add_special_tokens=False)
        print(f"Text: {repr(text)}")
        print(f"  Tokens: {tokens}")
        print(f"  Token IDs: {token_ids}")
        print(f"  Count: {len(token_ids)} tokens")
        print()
    
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
    
    print("🧪 Testing fixed collator with correct token...")
    
    # Test fixed collator
    collator = FixedGemma3nDataCollator(processor, handle_text_only=True)
    
    try:
        batch = collator([test_example])
        print("✅ SUCCESS! Batch created:")
        print(f"  Keys: {batch.keys()}")
        print(f"  Input IDs shape: {batch['input_ids'].shape}")
        print(f"  Pixel values shape: {batch['pixel_values'].shape}")
        
        # Decode input_ids to see what tokens were actually used
        print("\n🔍 Decoded input:")
        decoded_text = processor.tokenizer.decode(batch['input_ids'][0], skip_special_tokens=False)
        print(f"  Decoded: {repr(decoded_text[:300])}...")
        
        # Count image tokens in decoded text
        soft_token_count = decoded_text.count('<image_soft_token>')
        print(f"  <image_soft_token> count: {soft_token_count}")
        
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
                
        print(f"\n🎉 ALL TESTS PASSED! No more token mismatch!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_token_fix()
    if success:
        print("\n✅ Token fix successful! Ready for training!")
    else:
        print("\n❌ Need more debugging...")