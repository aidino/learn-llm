#!/usr/bin/env python3
"""
Main training script for Gemma3N fine-tuning with Unsloth and Comet ML
Optimized for T4 GPU with 1000 samples dataset

Usage:
    python train_gemma3n.py [options]

Examples:
    # Basic training with default config
    python train_gemma3n.py

    # Training with custom config file
    python train_gemma3n.py --config custom_config.json

    # Quick test with limited samples
    python train_gemma3n.py --test-run --max-samples 50

    # Resume from checkpoint
    python train_gemma3n.py --resume-from-checkpoint outputs/checkpoint-100

    # Push to HuggingFace Hub after training
    python train_gemma3n.py --push-to-hub your-username/model-name

Author: Generated for Vietnamese 6th grade math tutoring
"""

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Optional, List
import json

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Import our modules
from config import (
    ExperimentConfig, 
    get_optimized_config_for_t4,
    get_config_for_larger_gpu,
    validate_environment,
    print_config_summary
)
from trainer_wrapper import create_trainer_wrapper, run_complete_training_pipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

# Suppress some warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Fine-tune Gemma3N for Vietnamese 6th grade math tutoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Basic training
  %(prog)s --config my_config.json           # Custom config  
  %(prog)s --test-run --max-samples 50       # Quick test
  %(prog)s --gpu-type a100                   # For larger GPU
  %(prog)s --push-to-hub user/model          # Push to Hub
  %(prog)s --resume-from-checkpoint path     # Resume training
        """
    )
    
    # Configuration options
    parser.add_argument(
        "--config", 
        type=str,
        help="Path to custom configuration JSON file"
    )
    
    parser.add_argument(
        "--gpu-type",
        choices=["t4", "a100", "v100", "auto"],
        default="auto",
        help="GPU type for optimized configuration (default: auto)"
    )
    
    # Training options
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for model and checkpoints"
    )
    
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="Name for the experiment"
    )
    
    parser.add_argument(
        "--num-epochs",
        type=int,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate for training"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Per-device training batch size"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to use (for testing)"
    )
    
    # Model options
    parser.add_argument(
        "--model-name",
        type=str,
        help="Base model name to fine-tune"
    )
    
    parser.add_argument(
        "--max-seq-length",
        type=int,
        help="Maximum sequence length"
    )
    
    parser.add_argument(
        "--lora-r",
        type=int,
        help="LoRA rank"
    )
    
    parser.add_argument(
        "--lora-alpha",
        type=int,
        help="LoRA alpha parameter"
    )
    
    # Dataset options
    parser.add_argument(
        "--dataset-name",
        type=str,
        help="Dataset name or path"
    )
    
    # Comet ML options
    parser.add_argument(
        "--comet-workspace",
        type=str,
        help="Comet ML workspace"
    )
    
    parser.add_argument(
        "--comet-project",
        type=str,
        help="Comet ML project"
    )
    
    parser.add_argument(
        "--comet-api-key",
        type=str,
        help="Comet ML API key"
    )
    
    # Training control
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        help="Path to checkpoint to resume training from"
    )
    
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Run a quick test with limited samples and epochs"
    )
    
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training, only setup and validate"
    )
    
    parser.add_argument(
        "--inference-only",
        action="store_true",
        help="Run inference test only (requires existing model)"
    )
    
    # Saving and sharing
    parser.add_argument(
        "--save-formats",
        nargs="+",
        choices=["lora", "merged_16bit", "merged_4bit", "gguf_q8_0", "gguf_q4_k_m"],
        default=["lora"],
        help="Model save formats"
    )
    
    parser.add_argument(
        "--push-to-hub",
        type=str,
        help="Push model to HuggingFace Hub (format: username/model-name)"
    )
    
    parser.add_argument(
        "--hf-token",
        type=str,
        help="HuggingFace token for pushing to Hub"
    )
    
    parser.add_argument(
        "--private-repo",
        action="store_true",
        help="Make HuggingFace repository private"
    )
    
    # Debugging and logging
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--save-config",
        type=str,
        help="Save the final configuration to specified file"
    )
    
    return parser.parse_args()


def detect_gpu_type():
    """Detect GPU type for automatic configuration"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0).lower()
            if "t4" in gpu_name:
                return "t4"
            elif "a100" in gpu_name:
                return "a100"
            elif "v100" in gpu_name:
                return "v100"
            else:
                # Default to T4 config for unknown GPUs
                logger.info(f"Unknown GPU: {gpu_name}, using T4 configuration")
                return "t4"
        else:
            logger.warning("CUDA not available, using CPU configuration")
            return "t4"  # Conservative choice
    except Exception as e:
        logger.warning(f"Could not detect GPU: {e}, using T4 configuration")
        return "t4"


def load_config(args) -> ExperimentConfig:
    """Load and create configuration based on arguments"""
    
    # Start with base configuration
    if args.config:
        # Load from file
        logger.info(f"Loading configuration from: {args.config}")
        config = ExperimentConfig.load_from_file(args.config)
    else:
        # Use optimized configuration based on GPU type
        gpu_type = args.gpu_type if args.gpu_type != "auto" else detect_gpu_type()
        logger.info(f"Using optimized configuration for GPU type: {gpu_type}")
        
        if gpu_type == "t4":
            config = get_optimized_config_for_t4()
        else:
            config = get_config_for_larger_gpu()
    
    # Override with command line arguments
    if args.output_dir:
        config.training.output_dir = args.output_dir
    
    if args.experiment_name:
        config.comet.experiment_name = args.experiment_name
    
    if args.num_epochs:
        config.training.num_train_epochs = args.num_epochs
    
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    
    if args.batch_size:
        config.training.per_device_train_batch_size = args.batch_size
    
    if args.max_samples:
        config.dataset.max_samples = args.max_samples
    
    if args.model_name:
        config.model.model_name = args.model_name
    
    if args.max_seq_length:
        config.model.max_seq_length = args.max_seq_length
    
    if args.lora_r:
        config.model.lora_r = args.lora_r
    
    if args.lora_alpha:
        config.model.lora_alpha = args.lora_alpha
    
    if args.dataset_name:
        config.dataset.dataset_name = args.dataset_name
    
    # Comet ML configuration
    if args.comet_workspace:
        config.comet.workspace = args.comet_workspace
    
    if args.comet_project:
        config.comet.project = args.comet_project
    
    if args.comet_api_key:
        config.comet.api_key = args.comet_api_key
    
    # Test run modifications
    if args.test_run:
        logger.info("🧪 Test run mode enabled")
        config.training.num_train_epochs = 1
        config.training.max_steps = 10
        config.training.eval_steps = 5
        config.training.save_steps = 5
        config.training.logging_steps = 1
        config.dataset.max_samples = min(config.dataset.max_samples or 100, 50)
        config.comet.experiment_name = f"test_{config.comet.experiment_name}"
    
    # Resume from checkpoint
    if args.resume_from_checkpoint:
        config.training.resume_from_checkpoint = args.resume_from_checkpoint
    
    return config


def setup_environment(args):
    """Setup environment variables and logging"""
    
    # Set logging level
    log_level = getattr(logging, args.log_level.upper())
    logging.getLogger().setLevel(log_level)
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set environment variables if provided
    if args.comet_api_key:
        os.environ["COMET_API_KEY"] = args.comet_api_key
    
    if args.comet_workspace:
        os.environ["COMET_WORKSPACE"] = args.comet_workspace
    
    if args.comet_project:
        os.environ["COMET_PROJECT"] = args.comet_project
    
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
    
    # Validate environment
    try:
        validate_environment()
    except ValueError as e:
        logger.warning(f"Environment validation warning: {e}")


def run_inference_test(config: ExperimentConfig, model_path: Optional[str] = None):
    """Run inference test với existing model"""
    logger.info("🧪 Running inference test...")
    
    # Create wrapper
    wrapper = create_trainer_wrapper(config)
    
    try:
        # Setup components
        wrapper.setup_components()
        
        # Load model
        if model_path:
            # Load from specific path
            logger.info(f"Loading model from: {model_path}")
            # TODO: Implement loading from specific path
        else:
            # Load base model and apply LoRA
            model, tokenizer = wrapper.model_manager.load_base_model()
            model = wrapper.model_manager.apply_lora()
        
        # Run inference test
        results = wrapper.run_inference_test()
        
        # Print results
        logger.info("📊 Inference Test Results:")
        for i, result in enumerate(results, 1):
            logger.info(f"Test {i}:")
            logger.info(f"  Question: {result['question']}")
            if result['success']:
                logger.info(f"  Response: {result['response']}")
            else:
                logger.error(f"  Error: {result['error']}")
        
        return results
        
    finally:
        wrapper.cleanup()


def main():
    """Main training function"""
    args = parse_arguments()
    
    # Setup environment
    setup_environment(args)
    
    # Load configuration
    try:
        config = load_config(args)
    except Exception as e:
        logger.error(f"❌ Failed to load configuration: {e}")
        return 1
    
    # Save configuration if requested
    if args.save_config:
        config.save_to_file(args.save_config)
        logger.info(f"💾 Configuration saved to: {args.save_config}")
    
    # Print configuration summary
    print_config_summary(config)
    
    # Inference only mode
    if args.inference_only:
        try:
            results = run_inference_test(config, args.resume_from_checkpoint)
            logger.info("✅ Inference test completed successfully")
            return 0
        except Exception as e:
            logger.error(f"❌ Inference test failed: {e}")
            return 1
    
    # Skip training mode
    if args.skip_training:
        logger.info("⏭️  Skipping training as requested")
        return 0
    
    # Run training pipeline
    try:
        logger.info("🚀 Starting Gemma3N fine-tuning pipeline...")
        
        # Run complete training
        results = run_complete_training_pipeline(config)
        
        # Push to Hub if requested
        if args.push_to_hub:
            logger.info(f"☁️  Pushing model to HuggingFace Hub: {args.push_to_hub}")
            
            wrapper = create_trainer_wrapper(config)
            wrapper.setup_components()
            
            # Load trained model
            model, tokenizer = wrapper.model_manager.load_base_model()
            model = wrapper.model_manager.apply_lora()
            
            # Load best checkpoint if available
            if results.get("best_checkpoint"):
                logger.info(f"Loading best checkpoint: {results['best_checkpoint']}")
                # TODO: Load checkpoint
            
            # Push to hub
            wrapper.model_manager.push_to_hub(
                repo_id=args.push_to_hub,
                save_formats=args.save_formats,
                token=args.hf_token,
                private=args.private_repo
            )
            
            wrapper.cleanup()
        
        # Print final summary
        logger.info("🎉 Training pipeline completed successfully!")
        logger.info("📊 Final Results Summary:")
        logger.info(f"  Training Loss: {results['training_stats'].get('train_loss', 'N/A')}")
        logger.info(f"  Eval Loss: {results['training_stats'].get('eval_loss', 'N/A')}")
        logger.info(f"  Best Checkpoint: {results.get('best_checkpoint', 'N/A')}")
        logger.info(f"  Saved Formats: {list(results['saved_paths'].keys())}")
        
        if results.get('experiment_url'):
            logger.info(f"  Comet ML Experiment: {results['experiment_url']}")
        
        # Print inference test results
        if results.get('inference_results'):
            logger.info("  Inference Test:")
            for i, result in enumerate(results['inference_results'], 1):
                if result['success']:
                    logger.info(f"    Test {i}: ✅ {result['response'][:50]}...")
                else:
                    logger.info(f"    Test {i}: ❌ {result.get('error', 'Unknown error')}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("⚠️  Training interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"❌ Training pipeline failed: {e}")
        logger.exception("Full error traceback:")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
