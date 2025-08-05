#!/usr/bin/env python3
"""
Script để kiểm tra số lượng samples có images trong dataset

Usage:
    python check_dataset_images.py
"""

import sys
from datasets import load_dataset
from PIL import Image

def check_dataset_images():
    """Check dataset để xem có bao nhiều samples có real images."""
    print("🔍 Checking dataset for samples with images...")
    
    try:
        # Load dataset
        dataset = load_dataset("ngohongthai/exam-sixth_grade-instruct-dataset", split="train")
        print(f"Total samples: {len(dataset)}")
        
        samples_with_images = 0
        total_images = 0
        
        # Check first 100 samples for quick analysis
        check_count = min(100, len(dataset))
        print(f"Checking first {check_count} samples...")
        
        for i in range(check_count):
            sample = dataset[i]
            
            # Count images in question
            question_images = sample["question"].count("![")
            
            # Count images in solution  
            solution_images = sample["solution"].count("![")
            
            sample_image_count = question_images + solution_images
            
            if sample_image_count > 0:
                samples_with_images += 1
                total_images += sample_image_count
                
                if i < 10:  # Show details for first 10
                    print(f"Sample {i}: {sample_image_count} images (Q:{question_images}, S:{solution_images})")
                    print(f"  Question: {sample['question'][:100]}...")
                    print(f"  Solution: {sample['solution'][:100]}...")
                    print()
        
        print(f"\n📊 Results for first {check_count} samples:")
        print(f"- Samples with images: {samples_with_images}")
        print(f"- Text-only samples: {check_count - samples_with_images}")
        print(f"- Total images found: {total_images}")
        print(f"- Percentage with images: {samples_with_images/check_count*100:.1f}%")
        
        if samples_with_images > 0:
            print(f"- Average images per sample: {total_images/samples_with_images:.1f}")
        
        # Estimate for full dataset
        if check_count < len(dataset):
            estimated_samples_with_images = int(samples_with_images * len(dataset) / check_count)
            print(f"\n🔮 Estimated for full dataset ({len(dataset)} samples):")
            print(f"- Estimated samples with images: {estimated_samples_with_images}")
            print(f"- Estimated text-only samples: {len(dataset) - estimated_samples_with_images}")
        
        return samples_with_images > 0
        
    except Exception as e:
        print(f"Error checking dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    print("🖼️  Dataset Image Checker")
    print("=" * 50)
    
    success = check_dataset_images()
    
    if success:
        print("\n✅ Dataset contains samples with images!")
        print("You can proceed with vision fine-tuning.")
    else:
        print("\n❌ No samples with images found!")
        print("This dataset might not be suitable for vision fine-tuning.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)