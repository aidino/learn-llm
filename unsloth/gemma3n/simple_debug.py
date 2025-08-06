#!/usr/bin/env python3
"""
Simple debug script với proper memory management
"""

import torch
from datasets import load_dataset  
from unsloth import FastVisionModel

def simple_debug():
    """Simple debug với memory-efficient approach"""
    print("🔍 SIMPLE DEBUG - Memory Efficient")
    print("="*50)
    
    # Load model with better memory management
    print("📥 Loading model with device mapping...")
    try:
        model, processor = FastVisionModel.from_pretrained(
            "unsloth/gemma-3n-E4B",
            load_in_4bit=True,
            device_map="auto",  # Let it handle device mapping
            llm_int8_enable_fp32_cpu_offload=True,  # Enable CPU offload
            max_memory={0: "13GB"}  # Limit GPU memory usage
        )
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False
    
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_disable()
        print("🔧 Gradient checkpointing disabled")
    
    # Test tokenizer only first (no model forward)
    print("\n🧪 TOKENIZER ANALYSIS:")
    print("-" * 30)
    
    # Test specific tokens
    tokens_to_test = [
        "<image>",
        "<image_soft_token>", 
        "test <image_soft_token> test"
    ]
    
    for token in tokens_to_test:
        try:
            # Encode
            token_ids = processor.tokenizer.encode(token, add_special_tokens=False)
            tokens = processor.tokenizer.tokenize(token)
            
            # Decode
            decoded = processor.tokenizer.decode(token_ids, skip_special_tokens=False)
            
            print(f"Input: {repr(token)}")
            print(f"  Tokens: {tokens}")
            print(f"  IDs: {token_ids} (length: {len(token_ids)})")
            print(f"  Decoded: {repr(decoded)}")
            print()
            
        except Exception as e:
            print(f"❌ Tokenization failed for {repr(token)}: {e}")
    
    # Test conversation template without image first
    print("🧪 CONVERSATION TEMPLATE TEST:")
    print("-" * 30)
    
    # Simple conversation
    simple_conv = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "What is 2+2?"}]
        },
        {
            "role": "assistant", 
            "content": [{"type": "text", "text": "2+2 equals 4."}]
        }
    ]
    
    try:
        text_template = processor.apply_chat_template(
            simple_conv, tokenize=False, add_generation_prompt=False
        )
        print(f"✅ Template without image:")
        print(f"  Result: {repr(text_template)}")
        print(f"  Length: {len(text_template)}")
        print()
        
        # Add image manually
        text_with_token = "<image_soft_token>\n" + text_template
        print(f"✅ Template with manual token:")
        print(f"  Result: {repr(text_with_token[:100])}...")
        print()
        
        # Test tokenization of the full text
        final_tokens = processor.tokenizer.tokenize(text_with_token)
        final_ids = processor.tokenizer.encode(text_with_token, add_special_tokens=False)
        
        print(f"📊 FINAL TOKENIZATION:")
        print(f"  Total tokens: {len(final_ids)}")
        print(f"  First 10 tokens: {final_tokens[:10]}")
        
        # Count image tokens
        image_soft_token_id = processor.tokenizer.convert_tokens_to_ids('<image_soft_token>')
        image_count = final_ids.count(image_soft_token_id)
        
        print(f"  <image_soft_token> ID: {image_soft_token_id}")
        print(f"  <image_soft_token> count: {image_count}")
        
        if image_count == 1:
            print("✅ Perfect! Exactly 1 image token as expected")
        else:
            print(f"❌ Problem! Expected 1, got {image_count}")
            
    except Exception as e:
        print(f"❌ Template test failed: {e}")
        return False
    
    # Test with placeholder image (processor test without model forward)
    print("\n🧪 PROCESSOR TEST (no model forward):")
    print("-" * 30)
    
    try:
        from PIL import Image
        placeholder = Image.new('RGB', (224, 224), color=(245, 245, 245))
        
        # Test processor
        batch = processor(
            text=[text_with_token],
            images=[[placeholder]],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512  # Keep short
        )
        
        print(f"✅ Processor successful!")
        print(f"  Batch keys: {batch.keys()}")
        print(f"  Input IDs shape: {batch['input_ids'].shape}")
        print(f"  Pixel values shape: {batch['pixel_values'].shape}")
        
        # Decode what processor actually created
        decoded_batch = processor.tokenizer.decode(batch['input_ids'][0], skip_special_tokens=False)
        print(f"  Processed text: {repr(decoded_batch[:150])}...")
        
        # Count tokens in final batch
        batch_image_count = decoded_batch.count('<image_soft_token>')
        print(f"  <image_soft_token> in batch: {batch_image_count}")
        
        # Check special image tokens that model might be looking for
        input_ids = batch['input_ids'][0]
        special_tokens = {
            '<image_soft_token>': processor.tokenizer.convert_tokens_to_ids('<image_soft_token>'),
            '<start_of_image>': processor.tokenizer.convert_tokens_to_ids('<start_of_image>'),
            '<end_of_image>': processor.tokenizer.convert_tokens_to_ids('<end_of_image>')
        }
        
        print(f"\n📊 SPECIAL TOKEN ANALYSIS:")
        for token_name, token_id in special_tokens.items():
            count = (input_ids == token_id).sum().item() if token_id is not None else 0
            print(f"  {token_name} (ID: {token_id}): {count} occurrences")
        
        return True
        
    except Exception as e:
        print(f"❌ Processor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting simple debug...")
    success = simple_debug()
    if success:
        print("\n✅ Simple debug completed! Check the token counts above.")
    else:
        print("\n❌ Debug failed.")