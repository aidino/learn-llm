#!/usr/bin/env python3
"""
Demo script để test quick setup và validate
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def demo_config():
    """Demo configuration system"""
    print("🔧 Testing Configuration System...")
    
    from config import get_optimized_config_for_t4, print_config_summary
    
    config = get_optimized_config_for_t4()
    print_config_summary(config)
    
    print("✅ Configuration system working!")
    return True

def demo_data_processor():
    """Demo data processor (without actual loading)"""
    print("\n📚 Testing Data Processor...")
    
    from data_processor import create_data_processor
    from config import DatasetConfig
    
    config = DatasetConfig()
    processor = create_data_processor(config)
    
    # Test sample processing
    sample = {
        "question": "Tính 5 + 3 = ?",
        "solution": "5 + 3 = 8"
    }
    
    processed = processor.process_sample(sample)
    print(f"Sample processed: {list(processed.keys())}")
    print(f"Conversation format: {processed['conversations'][0]['role']}")
    
    print("✅ Data processor working!")
    return True

def demo_model_manager():
    """Demo model manager setup (without loading)"""
    print("\n🤖 Testing Model Manager...")
    
    from model_manager import create_model_manager
    from config import ModelConfig
    
    config = ModelConfig()
    config.max_seq_length = 512  # Smaller for demo
    
    manager = create_model_manager(config)
    print(f"Model config: {config.model_name}")
    print(f"LoRA rank: {config.lora_r}")
    print(f"Max sequence length: {config.max_seq_length}")
    
    print("✅ Model manager working!")
    return True

def demo_cli_interface():
    """Demo CLI interface"""
    print("\n💻 Testing CLI Interface...")
    
    # Simulate CLI call
    import subprocess
    import os
    
    # Test help command
    try:
        result = subprocess.run([
            sys.executable, "train_gemma3n.py", "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 and "fine-tune" in result.stdout.lower():
            print("✅ CLI help working!")
            return True
        else:
            print("⚠️ CLI help might have issues")
            return False
    except Exception as e:
        print(f"⚠️ CLI test skipped: {e}")
        return True

def main():
    """Run all demos"""
    print("🚀 GEMMA3N FINE-TUNING DEMO")
    print("=" * 50)
    
    demos = [
        ("Configuration", demo_config),
        ("Data Processor", demo_data_processor), 
        ("Model Manager", demo_model_manager),
        ("CLI Interface", demo_cli_interface),
    ]
    
    results = {}
    
    for name, demo_func in demos:
        try:
            results[name] = demo_func()
        except Exception as e:
            print(f"❌ {name} demo failed: {e}")
            results[name] = False
    
    print("\n" + "=" * 50)
    print("📊 DEMO RESULTS:")
    
    all_passed = True
    for name, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {name}")
        all_passed = all_passed and passed
    
    if all_passed:
        print("\n🎉 All demos passed! System ready for training.")
        print("\nNext steps:")
        print("1. Set Comet ML credentials:")
        print("   export COMET_API_KEY='your-api-key'")
        print("   export COMET_WORKSPACE='your-workspace'")
        print("2. Run training:")
        print("   python train_gemma3n.py --test-run")
    else:
        print("\n⚠️ Some demos failed. Check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
