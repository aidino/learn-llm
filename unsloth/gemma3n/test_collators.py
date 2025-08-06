#!/usr/bin/env python3
"""
Test script để verify cả 2 data collator approaches
"""

import sys
import os
import torch
from datasets import load_dataset
from unsloth import FastVisionModel
from gemma3n_math_finetuning import HybridVisionDataCollator, prepare_dataset, CONFIG
from fixed_data_collator import FixedGemma3nDataCollator

def test_collators():
    """Test both collators với real data"""
    print("🧪 Testing Data Collators...")
    
    # Load model and processor
    try:
        model, processor = FastVisionModel.from_pretrained(
            CONFIG["model_name"],
            load_in_4bit=CONFIG["load_in_4bit"],
        )
        print("✅ Model and processor loaded")
        
        # Important: Disable gradient checkpointing for vision models
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_disable()
            print("🔧 Gradient checkpointing disabled")
            
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Load dataset
    try:
        dataset = prepare_dataset(CONFIG["dataset_name"], CONFIG["train_split"])
        # Take small sample for testing
        test_samples = [dataset[i] for i in range(3)]
        print(f"✅ Dataset loaded, testing with {len(test_samples)} samples")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return
    
    print("\n" + "="*80)
    print("🔧 TESTING FIXED GEMMA3N DATA COLLATOR")
    print("="*80)
    
    try:
        fixed_collator = FixedGemma3nDataCollator(
            processor, 
            handle_text_only=True
        )
        
        print("🧪 Testing fixed collator...")
        fixed_batch = fixed_collator(test_samples)
        
        print("✅ Fixed collator successful!")
        print(f"  Batch keys: {fixed_batch.keys()}")
        print(f"  Input shape: {fixed_batch['input_ids'].shape}")
        if 'pixel_values' in fixed_batch:
            print(f"  Pixel values shape: {fixed_batch['pixel_values'].shape}")
        
        # Test with model
        print("🧪 Testing forward pass...")
        model.eval()
        with torch.no_grad():
            outputs = model(**fixed_batch)
            print("✅ Forward pass successful!")
            print(f"  Loss: {outputs.loss.item() if outputs.loss is not None else 'None'}")
        
    except Exception as e:
        print(f"❌ Fixed collator failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("📦 TESTING HYBRID VISION DATA COLLATOR")
    print("="*80)
    
    try:
        hybrid_collator = HybridVisionDataCollator(
            processor,
            handle_text_only=True,
            debug_mode=True,
            force_image_tokens=True
        )
        
        print("🧪 Testing hybrid collator...")
        hybrid_batch = hybrid_collator(test_samples)
        
        print("✅ Hybrid collator successful!")
        print(f"  Batch keys: {hybrid_batch.keys()}")
        print(f"  Input shape: {hybrid_batch['input_ids'].shape}")
        if 'pixel_values' in hybrid_batch:
            print(f"  Pixel values shape: {hybrid_batch['pixel_values'].shape}")
        
        # Test with model
        print("🧪 Testing forward pass...")
        model.eval()
        with torch.no_grad():
            outputs = model(**hybrid_batch)
            print("✅ Forward pass successful!")
            print(f"  Loss: {outputs.loss.item() if outputs.loss is not None else 'None'}")
        
    except Exception as e:
        print(f"❌ Hybrid collator failed: {e}")
        import traceback
        traceback.print_exc()

def test_image_token_debug():
    """Run image token debug"""
    print("\n" + "="*80)
    print("🔍 DEBUGGING IMAGE TOKENS")
    print("="*80)
    
    try:
        from debug_image_tokens import debug_gemma3n_image_tokens
        debug_gemma3n_image_tokens()
    except Exception as e:
        print(f"❌ Image token debug failed: {e}")

if __name__ == "__main__":
    print("🚀 Starting Comprehensive Data Collator Testing...")
    
    # Test 1: Basic collator functionality
    test_collators()
    
    # Test 2: Debug image tokens
    test_image_token_debug()
    
    print("\n🎯 Testing completed!")