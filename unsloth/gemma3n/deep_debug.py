#!/usr/bin/env python3
"""
Deep debug để tìm ra exactly what's happening với tokenization
"""

import torch
from datasets import load_dataset  
from unsloth import FastVisionModel
from fixed_data_collator import FixedGemma3nDataCollator

def deep_debug():
    """Deep debug tokenization và model forward"""
    print("🔍 DEEP DEBUG - Tokenization Analysis")
    print("="*60)
    
    # Load model and processor
    print("📥 Loading model...")
    model, processor = FastVisionModel.from_pretrained(
        "unsloth/gemma-3n-E4B",
        load_in_4bit=True,
    )
    
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_disable()
        print("🔧 Gradient checkpointing disabled")
    
    # Test each individual token
    print("\n🧪 INDIVIDUAL TOKEN ANALYSIS:")
    print("-" * 40)
    
    tokens_to_test = [
        "<image>",
        "<image_soft_token>", 
        "<start_of_image>",
        "<end_of_image>"
    ]
    
    for token in tokens_to_test:
        token_ids = processor.tokenizer.encode(token, add_special_tokens=False)
        decoded = processor.tokenizer.decode(token_ids, skip_special_tokens=False)
        
        print(f"Token: {repr(token)}")
        print(f"  Encoded IDs: {token_ids}")
        print(f"  Length: {len(token_ids)}")
        print(f"  Decoded back: {repr(decoded)}")
        print(f"  Is special token: {token in processor.tokenizer.special_tokens_map.values()}")
        print()
    
    # Load dataset sample
    print("📥 Loading dataset sample...")
    dataset = load_dataset("ngohongthai/exam-sixth_grade-instruct-dataset", split="train")
    sample = dataset[0]
    
    conversation = [
        {
            "role": "user",
            "content": [{"type": "text", "text": sample["question"][:100]}]  # Shorter for debug
        },
        {
            "role": "assistant", 
            "content": [{"type": "text", "text": sample["solution"][:100]}]
        }
    ]
    
    test_example = {"conversations": conversation}
    
    print("🧪 TESTING COLLATOR STEP BY STEP:")
    print("-" * 40)
    
    collator = FixedGemma3nDataCollator(processor, handle_text_only=True)
    
    # Step 1: Test conversation modification
    conv = test_example["conversations"]
    modified_conv = collator._create_conversation_with_placeholder(conv)
    
    print("1️⃣ Modified conversation:")
    for i, msg in enumerate(modified_conv):
        print(f"  Message {i}: {msg['role']}")
        for j, content in enumerate(msg['content']):
            if content['type'] == 'image':
                print(f"    Content {j}: IMAGE ({type(content['image'])})")
            else:
                print(f"    Content {j}: TEXT '{content['text'][:50]}...'")
    
    # Step 2: Test template application
    text = processor.apply_chat_template(
        modified_conv, tokenize=False, add_generation_prompt=False
    )
    
    print(f"\n2️⃣ Template result:")
    print(f"  Text length: {len(text)}")
    print(f"  Text preview: {repr(text[:200])}...")
    print(f"  Contains <image_soft_token>: {'<image_soft_token>' in text}")
    print(f"  Contains <image>: {'<image>' in text}")
    
    # Step 3: Force token insertion if needed
    if '<image_soft_token>' not in text:
        print(f"\n3️⃣ Forcing token insertion...")
        text = collator._force_image_token_insertion(text, num_tokens=1)
        print(f"  After insertion: {repr(text[:200])}...")
        print(f"  Contains <image_soft_token>: {'<image_soft_token>' in text}")
    
    # Step 4: Test tokenization
    print(f"\n4️⃣ TOKENIZATION ANALYSIS:")
    print("-" * 40)
    
    # Tokenize the final text
    token_ids = processor.tokenizer.encode(text, add_special_tokens=False)
    tokens = processor.tokenizer.tokenize(text)
    
    print(f"  Total tokens: {len(token_ids)}")
    print(f"  First 20 tokens: {tokens[:20]}")
    print(f"  First 20 IDs: {token_ids[:20]}")
    
    # Count specific tokens
    image_soft_token_id = processor.tokenizer.convert_tokens_to_ids('<image_soft_token>')
    image_soft_count = token_ids.count(image_soft_token_id)
    
    print(f"  <image_soft_token> ID: {image_soft_token_id}")
    print(f"  <image_soft_token> count in tokenized: {image_soft_count}")
    
    # Find where image tokens are
    image_positions = [i for i, tid in enumerate(token_ids) if tid == image_soft_token_id]
    print(f"  <image_soft_token> positions: {image_positions}")
    
    # Step 5: Test with processor
    print(f"\n5️⃣ PROCESSOR TEST:")
    print("-" * 40)
    
    from PIL import Image
    placeholder = Image.new('RGB', (224, 224), color=(245, 245, 245))
    
    try:
        batch = processor(
            text=[text],
            images=[[placeholder]],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512  # Shorter for debug
        )
        
        print(f"  ✅ Processor success!")
        print(f"  Batch keys: {batch.keys()}")
        print(f"  Input IDs shape: {batch['input_ids'].shape}")
        print(f"  Pixel values shape: {batch['pixel_values'].shape}")
        
        # Decode processed input_ids
        decoded_processed = processor.tokenizer.decode(batch['input_ids'][0], skip_special_tokens=False)
        print(f"  Processed text: {repr(decoded_processed[:200])}...")
        
        # Count tokens in processed
        processed_soft_count = decoded_processed.count('<image_soft_token>')
        print(f"  <image_soft_token> in processed: {processed_soft_count}")
        
        # Step 6: Test model forward (this is where the error occurs)
        print(f"\n6️⃣ MODEL FORWARD TEST:")
        print("-" * 40)
        
        model.eval()
        with torch.no_grad():
            try:
                outputs = model(**batch)
                print(f"  ✅ Model forward success!")
                if outputs.loss is not None:
                    print(f"  Loss: {outputs.loss.item():.4f}")
            except Exception as model_error:
                print(f"  ❌ Model forward failed: {model_error}")
                
                # Additional debug for model error
                print(f"\n🔍 Model Error Debug:")
                print(f"  Input IDs shape: {batch['input_ids'].shape}")
                print(f"  Pixel values shape: {batch['pixel_values'].shape}")
                print(f"  Input IDs content: {batch['input_ids'][0][:50]}...")
                
                # Check for special image mask
                input_ids = batch['input_ids'][0]
                special_token_ids = [
                    processor.tokenizer.convert_tokens_to_ids('<image_soft_token>'),
                    processor.tokenizer.convert_tokens_to_ids('<start_of_image>'),
                    processor.tokenizer.convert_tokens_to_ids('<end_of_image>')
                ]
                
                for token_name, token_id in zip(['<image_soft_token>', '<start_of_image>', '<end_of_image>'], special_token_ids):
                    count = (input_ids == token_id).sum().item()
                    print(f"  {token_name} (ID {token_id}) count: {count}")
                
                return False
        
    except Exception as processor_error:
        print(f"  ❌ Processor failed: {processor_error}")
        return False
    
    return True

if __name__ == "__main__":
    success = deep_debug()
    if success:
        print("\n✅ Deep debug completed successfully!")
    else:
        print("\n❌ Found issues in deep debug.")