#!/usr/bin/env python3
"""
Tokenizer-only debug để tìm ra vấn đề tokenization
"""

from transformers import AutoTokenizer

def tokenizer_debug():
    """Debug chỉ tokenizer, không load model"""
    print("🔍 TOKENIZER-ONLY DEBUG")
    print("="*40)
    
    # Load tokenizer only
    print("📥 Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-3n-E4B")
        print("✅ Tokenizer loaded successfully")
    except Exception as e:
        print(f"❌ Tokenizer loading failed: {e}")
        return False
    
    # Check special tokens
    print(f"\n🔍 SPECIAL TOKENS MAP:")
    print("-" * 30)
    for name, token in tokenizer.special_tokens_map.items():
        print(f"{name}: {repr(token)}")
    
    # Test image-related tokens
    print(f"\n🧪 IMAGE TOKEN TESTS:")
    print("-" * 30)
    
    test_cases = [
        # Test wrong token
        "<image>",
        # Test correct tokens  
        "<image_soft_token>",
        "<start_of_image>",
        "<end_of_image>",
        # Test in context
        "Hello <image_soft_token> world",
        "Text before <image_soft_token> text after"
    ]
    
    for text in test_cases:
        print(f"\nInput: {repr(text)}")
        
        # Tokenize
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        
        print(f"  Tokens: {tokens}")
        print(f"  Token IDs: {token_ids}")
        print(f"  Length: {len(token_ids)}")
        
        # Decode back
        decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
        print(f"  Decoded: {repr(decoded)}")
        
        # Count specific image tokens
        image_soft_count = text.count('<image_soft_token>')
        decoded_soft_count = decoded.count('<image_soft_token>')
        
        if image_soft_count > 0:
            print(f"  <image_soft_token> original: {image_soft_count}")
            print(f"  <image_soft_token> after decode: {decoded_soft_count}")
            if image_soft_count == decoded_soft_count:
                print(f"  ✅ Token preserved correctly")
            else:
                print(f"  ❌ Token count changed!")
    
    # Test conversation template format
    print(f"\n🧪 CONVERSATION TEMPLATE TEST:")
    print("-" * 30)
    
    # Create simple conversation
    conversation = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "What is 2+2?"}]
        },
        {
            "role": "assistant", 
            "content": [{"type": "text", "text": "4"}]
        }
    ]
    
    try:
        # Test if tokenizer has apply_chat_template
        if hasattr(tokenizer, 'apply_chat_template'):
            template_result = tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=False
            )
            print(f"✅ Chat template result:")
            print(f"  {repr(template_result)}")
            
            # Test adding image token manually
            with_image = "<image_soft_token>\n" + template_result
            print(f"\n✅ With manual image token:")
            print(f"  {repr(with_image[:100])}...")
            
            # Tokenize the result
            final_tokens = tokenizer.tokenize(with_image)
            final_ids = tokenizer.encode(with_image, add_special_tokens=False)
            
            print(f"\n📊 FINAL ANALYSIS:")
            print(f"  Total tokens: {len(final_ids)}")
            print(f"  Sample tokens: {final_tokens[:15]}...")
            
            # Find image token ID and count
            try:
                image_token_id = tokenizer.convert_tokens_to_ids('<image_soft_token>')
                image_count = final_ids.count(image_token_id)
                
                print(f"  <image_soft_token> ID: {image_token_id}")
                print(f"  <image_soft_token> count: {image_count}")
                
                if image_count == 1:
                    print(f"  ✅ Perfect! Exactly 1 image token")
                    return True
                else:
                    print(f"  ❌ Problem! Expected 1, got {image_count}")
                    
                    # Debug: find where image tokens are
                    positions = [i for i, tid in enumerate(final_ids) if tid == image_token_id]
                    print(f"  Image token positions: {positions}")
                    
                    return False
                    
            except Exception as e:
                print(f"  ❌ Token analysis failed: {e}")
                return False
                
        else:
            print("❌ Tokenizer doesn't have apply_chat_template method")
            return False
            
    except Exception as e:
        print(f"❌ Template test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting tokenizer-only debug...")
    success = tokenizer_debug()
    if success:
        print("\n✅ Tokenizer debug successful! Issue is likely in model forward.")
    else:
        print("\n❌ Found tokenization issue!")
        print("\n🔍 Next steps:")
        print("1. Check if we're using the right image token")
        print("2. Verify token insertion logic")
        print("3. Test with processor instead of tokenizer")