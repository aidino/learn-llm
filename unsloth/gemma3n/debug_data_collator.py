#!/usr/bin/env python3
"""
Debug script để kiểm tra data collator và dataset issues

Usage:
    python debug_data_collator.py
"""

import os
import sys
from PIL import Image
import torch
from datasets import load_dataset

# Thêm path để import từ main script
sys.path.append('.')

def debug_raw_dataset():
    """Debug raw dataset trước khi processing."""
    print("🔍 Debugging raw dataset...")
    
    try:
        # Load raw dataset
        raw_dataset = load_dataset("ngohongthai/exam-sixth_grade-instruct-dataset", split="train")
        print(f"Raw dataset size: {len(raw_dataset)}")
        
        # Check first few samples
        for i in range(min(3, len(raw_dataset))):
            sample = raw_dataset[i]
            print(f"\nSample {i}:")
            print(f"  Keys: {sample.keys()}")
            print(f"  Question: {sample['question'][:100]}...")
            print(f"  Solution: {sample['solution'][:100]}...")
            print(f"  Question has images: {'![' in sample['question']}")
            print(f"  Solution has images: {'![' in sample['solution']}")
            
    except Exception as e:
        print(f"Error loading raw dataset: {e}")
        return False
    
    return True

def debug_processed_dataset():
    """Debug processed dataset."""
    print("\n🔍 Debugging processed dataset...")
    
    try:
        from gemma3n_math_finetuning import prepare_dataset
        
        # Process small subset
        raw_dataset = load_dataset("ngohongthai/exam-sixth_grade-instruct-dataset", split="train[:5]")
        
        processed_data = []
        for i, sample in enumerate(raw_dataset):
            try:
                from gemma3n_math_finetuning import process_math_sample
                processed_sample = process_math_sample(sample)
                processed_data.append(processed_sample)
                
                print(f"\nProcessed sample {i}:")
                print(f"  Conversations: {len(processed_sample['conversations'])}")
                
                for j, conv in enumerate(processed_sample['conversations']):
                    print(f"  Conv {j}: role={conv['role']}, content_items={len(conv['content'])}")
                    for k, content in enumerate(conv['content']):
                        if content['type'] == 'image':
                            img = content.get('image')
                            print(f"    Content {k}: image, type={type(img)}, size={getattr(img, 'size', 'unknown')}")
                        else:
                            print(f"    Content {k}: text, length={len(content.get('text', ''))}")
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                import traceback
                traceback.print_exc()
        
        return len(processed_data) > 0
        
    except Exception as e:
        print(f"Error in processed dataset debug: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_processor():
    """Debug processor setup."""
    print("\n🔍 Debugging processor...")
    
    try:
        from gemma3n_math_finetuning import setup_model_and_processor, CONFIG
        
        model, processor = setup_model_and_processor(CONFIG)
        
        print("✅ Model and processor loaded successfully")
        print(f"Processor type: {type(processor)}")
        print(f"Tokenizer type: {type(processor.tokenizer)}")
        print(f"Has image processor: {hasattr(processor, 'image_processor')}")
        
        # Test simple processing
        test_text = "Hello world"
        test_image = Image.new('RGB', (224, 224), color='white')
        
        try:
            batch = processor(
                text=[test_text],
                images=[[test_image]],
                return_tensors="pt",
                padding=True
            )
            print("✅ Processor test successful")
            print(f"Batch keys: {batch.keys()}")
            
        except Exception as e:
            print(f"❌ Processor test failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"Error in processor debug: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_data_collator():
    """Debug data collator với real data."""
    print("\n🔍 Debugging data collator...")
    
    try:
        from gemma3n_math_finetuning import (
            setup_model_and_processor, 
            HybridVisionDataCollator,
            process_math_sample,
            CONFIG
        )
        
        # Setup processor
        model, processor = setup_model_and_processor(CONFIG)
        
        # Create collator
        collator = HybridVisionDataCollator(processor, handle_text_only=True)
        
        # Load và process một sample
        raw_dataset = load_dataset("ngohongthai/exam-sixth_grade-instruct-dataset", split="train[:3]")
        
        processed_samples = []
        for sample in raw_dataset:
            try:
                processed = process_math_sample(sample)
                processed_samples.append(processed)
            except Exception as e:
                print(f"Error processing sample: {e}")
                # Tạo fallback sample
                fallback_sample = {
                    "conversations": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": "Test question"}]
                        },
                        {
                            "role": "assistant", 
                            "content": [{"type": "text", "text": "Test answer"}]
                        }
                    ]
                }
                processed_samples.append(fallback_sample)
        
        print(f"Processed {len(processed_samples)} samples")
        
        # Test collator
        try:
            batch = collator(processed_samples)
            print("✅ Data collator test successful!")
            print(f"Batch keys: {batch.keys()}")
            for key, value in batch.items():
                if hasattr(value, 'shape'):
                    print(f"  {key}: {value.shape}")
            
            return True
            
        except Exception as e:
            print(f"❌ Data collator failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"Error in data collator debug: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main debug function."""
    print("🐛 Data Collator Debug Script")
    print("=" * 50)
    
    success = True
    
    # 1. Debug raw dataset
    success &= debug_raw_dataset()
    
    # 2. Debug processed dataset
    success &= debug_processed_dataset()
    
    # 3. Debug processor
    success &= debug_processor()
    
    # 4. Debug data collator
    success &= debug_data_collator()
    
    if success:
        print("\n✅ All debug tests passed!")
        print("The data collator should work with your dataset.")
    else:
        print("\n❌ Some debug tests failed!")
        print("Please check the errors above and fix them.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)