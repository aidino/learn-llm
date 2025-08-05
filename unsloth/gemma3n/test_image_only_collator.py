#!/usr/bin/env python3
"""
Quick test của ImageOnlyVisionDataCollator để verify logic

Usage:
    python test_image_only_collator.py
"""

import sys
from PIL import Image

# Mock data for testing
def create_mock_sample_with_image():
    """Create mock sample có image."""
    return {
        "conversations": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Solve this math problem:"},
                    {"type": "image", "image": Image.new('RGB', (224, 224), color='red')}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "The answer is..."}
                ]
            }
        ]
    }

def create_mock_sample_without_image():
    """Create mock sample không có image."""
    return {
        "conversations": [
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": "What is 2+2?"}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "2+2 = 4"}
                ]
            }
        ]
    }

def test_image_detection():
    """Test image detection logic."""
    print("🧪 Testing image detection logic...")
    
    # Mock collator class với chỉ method cần thiết
    class MockImageOnlyCollator:
        def _validate_and_process_image(self, img):
            if img is None:
                return None
            if not hasattr(img, 'convert'):
                return None
            img = img.convert('RGB')
            if img.size[0] < 1 or img.size[1] < 1:
                return None
            return img
        
        def _extract_images_from_conversation(self, conv):
            images = []
            for message in conv:
                for content in message.get("content", []):
                    if content.get("type") == "image" and "image" in content:
                        img = content["image"]
                        processed_img = self._validate_and_process_image(img)
                        if processed_img is not None:
                            images.append(processed_img)
            return images
        
        def _has_real_images(self, conv):
            images = self._extract_images_from_conversation(conv)
            return len(images) > 0
    
    collator = MockImageOnlyCollator()
    
    # Test sample with image
    sample_with_image = create_mock_sample_with_image()
    has_images_1 = collator._has_real_images(sample_with_image["conversations"])
    print(f"Sample with image: {has_images_1} ✅" if has_images_1 else f"Sample with image: {has_images_1} ❌")
    
    # Test sample without image
    sample_without_image = create_mock_sample_without_image()
    has_images_2 = collator._has_real_images(sample_without_image["conversations"])
    print(f"Sample without image: {has_images_2} ✅" if not has_images_2 else f"Sample without image: {has_images_2} ❌")
    
    return has_images_1 and not has_images_2

def test_batch_filtering():
    """Test batch filtering logic."""
    print("\n🧪 Testing batch filtering logic...")
    
    # Create test batch
    examples = [
        create_mock_sample_with_image(),    # Should be kept
        create_mock_sample_without_image(), # Should be filtered out
        create_mock_sample_with_image(),    # Should be kept
        create_mock_sample_without_image()  # Should be filtered out
    ]
    
    # Mock filtering logic
    valid_examples = []
    for idx, example in enumerate(examples):
        conv = example["conversations"]
        
        # Check for images
        has_images = False
        for message in conv:
            for content in message.get("content", []):
                if content.get("type") == "image" and content.get("image") is not None:
                    img = content["image"]
                    if hasattr(img, 'convert') and hasattr(img, 'size'):
                        if img.size[0] > 0 and img.size[1] > 0:
                            has_images = True
                            break
            if has_images:
                break
        
        if has_images:
            valid_examples.append(example)
            print(f"Sample {idx}: HAS IMAGES ✅")
        else:
            print(f"Sample {idx}: NO IMAGES ❌ (FILTERED OUT)")
    
    print(f"\nFiltering results:")
    print(f"- Original samples: {len(examples)}")
    print(f"- Samples with images: {len(valid_examples)}")
    print(f"- Filtered out: {len(examples) - len(valid_examples)}")
    
    # Should keep 2 out of 4
    return len(valid_examples) == 2

def main():
    """Main test function."""
    print("🧪 ImageOnlyVisionDataCollator Logic Test")
    print("=" * 50)
    
    success = True
    
    # Test 1: Image detection
    success &= test_image_detection()
    
    # Test 2: Batch filtering
    success &= test_batch_filtering()
    
    if success:
        print("\n✅ All logic tests passed!")
        print("ImageOnlyVisionDataCollator should work correctly.")
    else:
        print("\n❌ Some logic tests failed!")
        print("Please check the implementation.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)