#!/usr/bin/env python3
"""
Example usage script for Gemma3N fine-tuning
Demonstrates different use cases and configurations
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from config import (
    ExperimentConfig,
    get_optimized_config_for_t4,
    get_config_for_larger_gpu,
    print_config_summary
)
from trainer_wrapper import run_complete_training_pipeline


def example_basic_training():
    """
    Example 1: Basic training với default configuration
    """
    print("=" * 60)
    print("📚 EXAMPLE 1: Basic Training")
    print("=" * 60)
    
    # Get optimized config for T4
    config = get_optimized_config_for_t4()
    
    # Quick test settings
    config.training.num_train_epochs = 1
    config.training.max_steps = 10
    config.dataset.max_samples = 20
    config.comet.experiment_name = "example_basic_training"
    
    print_config_summary(config)
    
    # Uncomment để chạy thực tế
    # results = run_complete_training_pipeline(config)
    # print(f"Results: {results}")


def example_custom_hyperparameters():
    """
    Example 2: Custom hyperparameters cho model tốt hơn
    """
    print("=" * 60)
    print("🔧 EXAMPLE 2: Custom Hyperparameters")
    print("=" * 60)
    
    config = get_optimized_config_for_t4()
    
    # Custom model settings
    config.model.lora_r = 32                    # Higher rank
    config.model.lora_alpha = 32               # Match với r
    config.model.max_seq_length = 2048         # Longer sequences
    
    # Custom training settings
    config.training.learning_rate = 1e-4       # Lower learning rate
    config.training.num_train_epochs = 3       # More epochs
    config.training.warmup_ratio = 0.05        # Less warmup
    config.training.weight_decay = 0.02        # More regularization
    
    # Custom experiment
    config.comet.experiment_name = "example_custom_hyperparams"
    config.comet.tags.append("custom")
    config.comet.tags.append("high-quality")
    
    # Enable early stopping
    config.training.early_stopping_patience = 2
    config.training.eval_strategy = "steps"
    config.training.eval_steps = 25
    
    print_config_summary(config)


def example_large_gpu_config():
    """
    Example 3: Configuration cho GPU lớn (A100, V100)
    """
    print("=" * 60)
    print("🚀 EXAMPLE 3: Large GPU Configuration")
    print("=" * 60)
    
    config = get_config_for_larger_gpu()
    
    # Aggressive settings cho GPU lớn
    config.training.per_device_train_batch_size = 4
    config.training.gradient_accumulation_steps = 4    # Effective batch = 16
    config.training.num_train_epochs = 5
    config.model.max_seq_length = 4096                 # Very long sequences
    
    # Use BF16 instead of FP16
    config.training.fp16 = False
    config.training.bf16 = True
    
    # More workers
    config.training.dataloader_num_workers = 8
    
    config.comet.experiment_name = "example_large_gpu"
    
    print_config_summary(config)


def example_memory_efficient():
    """
    Example 4: Memory-efficient config cho GPU nhỏ
    """
    print("=" * 60)
    print("💾 EXAMPLE 4: Memory Efficient")
    print("=" * 60)
    
    config = get_optimized_config_for_t4()
    
    # Extreme memory optimization
    config.model.max_seq_length = 1024         # Shorter sequences
    config.training.per_device_train_batch_size = 1
    config.training.gradient_accumulation_steps = 16   # Larger accumulation
    config.training.dataloader_num_workers = 0         # No parallel loading
    config.training.dataloader_pin_memory = False      # Reduce memory
    
    # More aggressive quantization
    config.model.load_in_4bit = True
    
    # Reduce LoRA complexity
    config.model.lora_r = 8
    config.model.lora_alpha = 16
    
    config.comet.experiment_name = "example_memory_efficient"
    
    print_config_summary(config)


def example_production_config():
    """
    Example 5: Production-ready configuration
    """
    print("=" * 60)
    print("🏭 EXAMPLE 5: Production Configuration")
    print("=" * 60)
    
    config = get_optimized_config_for_t4()
    
    # Production settings
    config.training.num_train_epochs = 3
    config.training.save_strategy = "epoch"
    config.training.eval_strategy = "epoch"
    config.training.load_best_model_at_end = True
    config.training.save_total_limit = 5
    
    # Robust training
    config.training.early_stopping_patience = 3
    config.training.warmup_ratio = 0.1
    config.training.weight_decay = 0.01
    config.training.max_grad_norm = 1.0
    
    # Full dataset
    config.dataset.max_samples = None  # Use all data
    
    # Production experiment naming
    config.comet.experiment_name = "production_v1.0"
    config.comet.tags = [
        "production", "stable", "gemma3n", 
        "vietnamese", "math-tutor", "v1.0"
    ]
    
    # Model registry
    config.comet.model_registry_name = "gemma3n-math-tutor-production"
    
    config.description = "Production model for Vietnamese 6th grade math tutoring"
    config.version = "1.0.0"
    
    print_config_summary(config)


def example_hyperparameter_sweep():
    """
    Example 6: Hyperparameter sweep configuration
    """
    print("=" * 60)
    print("🔍 EXAMPLE 6: Hyperparameter Sweep")
    print("=" * 60)
    
    # Different learning rates to test
    learning_rates = [5e-5, 1e-4, 2e-4, 5e-4]
    lora_ranks = [8, 16, 32]
    
    configs = []
    
    for lr in learning_rates:
        for rank in lora_ranks:
            config = get_optimized_config_for_t4()
            
            # Quick evaluation settings
            config.training.num_train_epochs = 1
            config.training.max_steps = 50
            config.dataset.max_samples = 100
            
            # Hyperparameters to sweep
            config.training.learning_rate = lr
            config.model.lora_r = rank
            config.model.lora_alpha = rank
            
            # Unique experiment name
            config.comet.experiment_name = f"sweep_lr{lr}_r{rank}"
            config.comet.tags.append("hyperparameter_sweep")
            
            configs.append(config)
            
            print(f"Config: LR={lr}, LoRA_R={rank}")
    
    print(f"\nTotal configurations: {len(configs)}")
    print("To run sweep, uncomment the training loop below")
    
    # Uncomment để chạy sweep
    # for i, config in enumerate(configs):
    #     print(f"Running config {i+1}/{len(configs)}")
    #     try:
    #         results = run_complete_training_pipeline(config)
    #         print(f"Config {i+1} completed: {results['training_stats']['eval_loss']}")
    #     except Exception as e:
    #         print(f"Config {i+1} failed: {e}")


def example_resume_training():
    """
    Example 7: Resume training từ checkpoint
    """
    print("=" * 60)
    print("🔄 EXAMPLE 7: Resume Training")
    print("=" * 60)
    
    config = get_optimized_config_for_t4()
    
    # Setup cho resume
    config.training.resume_from_checkpoint = "outputs/checkpoint-100"  # Path to checkpoint
    config.training.num_train_epochs = 5  # Continue training
    
    # Có thể adjust learning rate khi resume
    config.training.learning_rate = 1e-5  # Lower LR for fine-tuning
    
    config.comet.experiment_name = "example_resume_training"
    
    print_config_summary(config)
    print(f"Resume from: {config.training.resume_from_checkpoint}")


def example_save_and_load_config():
    """
    Example 8: Save và load configuration
    """
    print("=" * 60)
    print("💾 EXAMPLE 8: Save and Load Config")
    print("=" * 60)
    
    # Create custom config
    config = get_optimized_config_for_t4()
    config.comet.experiment_name = "example_saved_config"
    config.training.learning_rate = 3e-4
    config.model.lora_r = 24
    
    # Save config
    config_path = "example_config.json"
    config.save_to_file(config_path)
    print(f"✅ Config saved to: {config_path}")
    
    # Load config
    loaded_config = ExperimentConfig.load_from_file(config_path)
    print(f"✅ Config loaded from: {config_path}")
    
    # Verify they're the same
    assert loaded_config.training.learning_rate == config.training.learning_rate
    assert loaded_config.model.lora_r == config.model.lora_r
    print("✅ Config verification passed")
    
    # Clean up
    os.remove(config_path)
    print(f"🧹 Cleaned up: {config_path}")


def example_inference_only():
    """
    Example 9: Inference-only mode
    """
    print("=" * 60)
    print("🧠 EXAMPLE 9: Inference Only")
    print("=" * 60)
    
    from model_manager import create_model_manager
    from config import ModelConfig, InferenceConfig
    
    # Create model config
    model_config = ModelConfig()
    model_config.max_seq_length = 1024  # Smaller for inference
    
    # Create manager
    manager = create_model_manager(model_config)
    
    print("Setting up model for inference...")
    
    # Uncomment để chạy inference
    # model, tokenizer = manager.load_base_model()
    # model = manager.apply_lora()
    # 
    # # Load trained adapter if available
    # # manager.load_adapter("path/to/trained/adapter")
    # 
    # # Test questions
    # questions = [
    #     "Tính 15 + 27 = ?",
    #     "Một hình chữ nhật có chiều dài 6m và chiều rộng 4m. Tính chu vi?",
    #     "Tìm x: 2x + 5 = 15"
    # ]
    # 
    # inference_config = InferenceConfig(
    #     max_new_tokens=256,
    #     temperature=0.7,
    #     top_p=0.9
    # )
    # 
    # for i, question in enumerate(questions, 1):
    #     print(f"\nQuestion {i}: {question}")
    #     response = manager.generate_response(question, inference_config)
    #     print(f"Answer: {response}")


def main():
    """
    Chạy tất cả examples
    """
    print("🚀 GEMMA3N FINE-TUNING EXAMPLES")
    print("=" * 80)
    
    examples = [
        example_basic_training,
        example_custom_hyperparameters,
        example_large_gpu_config,
        example_memory_efficient,
        example_production_config,
        example_hyperparameter_sweep,
        example_resume_training,
        example_save_and_load_config,
        example_inference_only,
    ]
    
    for i, example_func in enumerate(examples, 1):
        try:
            print(f"\n{'='*20} EXAMPLE {i} {'='*20}")
            example_func()
            print("✅ Example completed successfully")
        except Exception as e:
            print(f"❌ Example failed: {e}")
        
        input("\nPress Enter to continue to next example...")
    
    print("\n🎉 All examples completed!")
    print("\nTo run actual training, uncomment the training calls in the examples.")
    print("Remember to set your Comet ML credentials first:")
    print("  export COMET_API_KEY='your-api-key'")
    print("  export COMET_WORKSPACE='your-workspace'")
    print("  export COMET_PROJECT='mathpal-gemma3n'")


if __name__ == "__main__":
    main()
