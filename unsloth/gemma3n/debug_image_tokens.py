#!/usr/bin/env python3
"""
Debug script để tìm image token format chính xác cho Gemma3n
"""

import sys
import os
from datasets import load_dataset
from unsloth import FastVisionModel

def debug_gemma3n_image_tokens():
    """Debug Gemma3n image token format"""
    print("🔍 Debugging Gemma3n Image Token Format...")
    
    # Load model and processor
    try:
        model, processor = FastVisionModel.from_pretrained(
            "unsloth/gemma-3n-E4B",
            load_in_4bit=True,
        )
        print("✅ Model and processor loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Check tokenizer special tokens
    print("\n🔍 CHECKING TOKENIZER SPECIAL TOKENS:")
    print("=" * 60)
    print(f"Special tokens map: {processor.tokenizer.special_tokens_map}")
    
    # Check for BOI/EOI tokens
    if hasattr(processor.tokenizer, 'boi_token'):
        print(f"📷 BOI token: {processor.tokenizer.boi_token}")
    else:
        print("❌ No BOI token found")
        
    if hasattr(processor.tokenizer, 'eoi_token'):
        print(f"📷 EOI token: {processor.tokenizer.eoi_token}")
    else:
        print("❌ No EOI token found")
    
    # Check if there are image-related special tokens
    print("\n🔍 CHECKING ALL SPECIAL TOKENS:")
    print("=" * 60)
    for name, token in processor.tokenizer.special_tokens_map.items():
        print(f"{name}: {repr(token)}")
        
    # Check additional image tokens
    if hasattr(processor.tokenizer, 'additional_special_tokens'):
        print(f"\nAdditional special tokens: {processor.tokenizer.additional_special_tokens}")
    
    # Check vocabulary for image-related tokens
    print("\n🔍 SEARCHING VOCABULARY FOR IMAGE TOKENS:")
    print("=" * 60)
    vocab = processor.tokenizer.get_vocab()
    image_related_tokens = []
    
    for token, token_id in vocab.items():
        if any(keyword in token.lower() for keyword in ['image', 'img', 'boi', 'eoi', 'vision', 'visual']):
            image_related_tokens.append((token, token_id))
    
    print(f"Found {len(image_related_tokens)} image-related tokens:")
    for token, token_id in sorted(image_related_tokens, key=lambda x: x[1]):
        print(f"  {token_id}: {repr(token)}")
    
    # Test conversation with images
    print("\n🧪 TESTING CONVERSATION WITH IMAGES:")
    print("=" * 60)
    
    # Create conversation with image
    from PIL import Image
    test_image = Image.new('RGB', (224, 224), color='red')
    
    conversation_with_image = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": test_image},
                {"type": "text", "text": "What is in this image?"}
            ]
        },
        {
            "role": "assistant", 
            "content": [{"type": "text", "text": "This is a test image."}]
        }
    ]
    
    try:
        # Test template with image
        formatted_with_image = processor.apply_chat_template(
            conversation_with_image,
            tokenize=False,
            add_generation_prompt=False
        )
        print("✅ CONVERSATION WITH IMAGE:")
        print(repr(formatted_with_image))
        print("\nFormatted:")
        print(formatted_with_image)
        
        # Check what tokens are used
        print(f"\nContains '<image>': {'<image>' in formatted_with_image}")
        if hasattr(processor.tokenizer, 'boi_token') and processor.tokenizer.boi_token:
            print(f"Contains BOI token '{processor.tokenizer.boi_token}': {processor.tokenizer.boi_token in formatted_with_image}")
        if hasattr(processor.tokenizer, 'eoi_token') and processor.tokenizer.eoi_token:
            print(f"Contains EOI token '{processor.tokenizer.eoi_token}': {processor.tokenizer.eoi_token in formatted_with_image}")
            
    except Exception as e:
        print(f"❌ Failed to format conversation with image: {e}")
    
    # Test conversation without images
    print("\n🧪 TESTING CONVERSATION WITHOUT IMAGES:")
    print("=" * 60)
    
    conversation_no_image = [
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
        formatted_no_image = processor.apply_chat_template(
            conversation_no_image,
            tokenize=False,
            add_generation_prompt=False
        )
        print("✅ CONVERSATION WITHOUT IMAGE:")
        print(repr(formatted_no_image))
        print("\nFormatted:")
        print(formatted_no_image)
        
    except Exception as e:
        print(f"❌ Failed to format conversation without image: {e}")
    
    # Test tokenization
    print("\n🧪 TESTING TOKENIZATION:")
    print("=" * 60)
    
    test_texts = [
        "<image>",
        "test <image> test",
    ]
    
    # Add BOI/EOI if they exist
    if hasattr(processor.tokenizer, 'boi_token') and processor.tokenizer.boi_token:
        test_texts.append(processor.tokenizer.boi_token)
        test_texts.append(f"test {processor.tokenizer.boi_token} test")
    
    if hasattr(processor.tokenizer, 'eoi_token') and processor.tokenizer.eoi_token:
        test_texts.append(processor.tokenizer.eoi_token)
        test_texts.append(f"test {processor.tokenizer.eoi_token} test")
    
    for text in test_texts:
        try:
            tokens = processor.tokenizer.tokenize(text)
            token_ids = processor.tokenizer.encode(text, add_special_tokens=False)
            print(f"Text: {repr(text)}")
            print(f"  Tokens: {tokens}")
            print(f"  Token IDs: {token_ids}")
            print()
        except Exception as e:
            print(f"Failed to tokenize {repr(text)}: {e}")

if __name__ == "__main__":
    debug_gemma3n_image_tokens()