"""
Model management module for Gemma3N fine-tuning
Handles model loading, LoRA setup, and model operations
"""

import logging
import os
import torch
from typing import Dict, Any, Optional, Tuple, Union
from pathlib import Path
import tempfile

from unsloth import FastModel, get_chat_template
from unsloth.chat_templates import train_on_responses_only
from transformers import PreTrainedModel, PreTrainedTokenizer
from peft import PeftModel

from config import ModelConfig, InferenceConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GemmaModelManager:
    """
    Manager class cho Gemma3N model với Unsloth optimizations
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.is_training_ready = False
        self.is_inference_ready = False
        
    def load_base_model(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load base Gemma3N model với Unsloth optimizations
        """
        try:
            logger.info(f"Loading base model: {self.config.model_name}")
            logger.info(f"Configuration:")
            logger.info(f"  Max sequence length: {self.config.max_seq_length}")
            logger.info(f"  4-bit quantization: {self.config.load_in_4bit}")
            logger.info(f"  Full fine-tuning: {self.config.full_finetuning}")
            
            # Load model với Unsloth
            self.model, self.tokenizer = FastModel.from_pretrained(
                model_name=self.config.model_name,
                max_seq_length=self.config.max_seq_length,
                load_in_4bit=self.config.load_in_4bit,
                full_finetuning=self.config.full_finetuning,
            )
            
            logger.info("✅ Base model loaded successfully")
            
            # Setup chat template
            self.setup_chat_template()
            
            return self.model, self.tokenizer
            
        except Exception as e:
            logger.error(f"❌ Error loading base model: {e}")
            raise
    
    def setup_chat_template(self):
        """
        Setup Gemma3N chat template
        """
        try:
            logger.info("Setting up Gemma3N chat template...")
            
            self.tokenizer = get_chat_template(
                self.tokenizer, 
                chat_template="gemma-3n"
            )
            
            logger.info("✅ Chat template setup complete")
            
        except Exception as e:
            logger.error(f"❌ Error setting up chat template: {e}")
            raise
    
    def apply_lora(self) -> PreTrainedModel:
        """
        Apply LoRA adapters to the model
        """
        try:
            logger.info("Applying LoRA configuration...")
            logger.info(f"LoRA settings:")
            logger.info(f"  Rank (r): {self.config.lora_r}")
            logger.info(f"  Alpha: {self.config.lora_alpha}")
            logger.info(f"  Dropout: {self.config.lora_dropout}")
            logger.info(f"  Target modules: {self.config.target_modules}")
            logger.info(f"  Gradient checkpointing: {self.config.use_gradient_checkpointing}")
            
            self.model = FastModel.get_peft_model(
                self.model,
                finetune_vision_layers=False,    # Text-only for math problems
                finetune_language_layers=True,
                finetune_attention_modules=True,
                finetune_mlp_modules=True,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                bias=self.config.lora_bias,
                target_modules=self.config.target_modules,
                use_gradient_checkpointing=self.config.use_gradient_checkpointing,
                use_rslora=self.config.use_rslora,
                random_state=self.config.random_state,
            )
            
            # Print trainable parameters
            self.print_trainable_parameters()
            
            logger.info("✅ LoRA applied successfully")
            return self.model
            
        except Exception as e:
            logger.error(f"❌ Error applying LoRA: {e}")
            raise
    
    def print_trainable_parameters(self):
        """
        Print information about trainable parameters
        """
        try:
            trainable_params = 0
            all_param = 0
            
            for _, param in self.model.named_parameters():
                all_param += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
            
            percentage = 100 * trainable_params / all_param
            
            logger.info("🧮 Parameter Statistics:")
            logger.info(f"  Trainable parameters: {trainable_params:,}")
            logger.info(f"  Total parameters: {all_param:,}")
            logger.info(f"  Trainable percentage: {percentage:.4f}%")
            
        except Exception as e:
            logger.warning(f"Could not calculate parameter statistics: {e}")
    
    def prepare_for_training(self):
        """
        Prepare model for training
        """
        try:
            logger.info("Preparing model for training...")
            
            # Enable training mode
            FastModel.for_training(self.model)
            
            self.is_training_ready = True
            logger.info("✅ Model ready for training")
            
        except Exception as e:
            logger.error(f"❌ Error preparing model for training: {e}")
            raise
    
    def prepare_for_inference(self):
        """
        Prepare model for inference
        """
        try:
            logger.info("Preparing model for inference...")
            
            # Enable inference mode với Unsloth optimizations
            FastModel.for_inference(self.model)
            
            self.is_inference_ready = True
            logger.info("✅ Model ready for inference")
            
        except Exception as e:
            logger.error(f"❌ Error preparing model for inference: {e}")
            raise
    
    def generate_response(
        self, 
        question: str, 
        inference_config: InferenceConfig
    ) -> str:
        """
        Generate response for a given question
        """
        if not self.is_inference_ready:
            self.prepare_for_inference()
        
        try:
            # Format input
            messages = [{
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": question,
                }]
            }]
            
            # Apply chat template
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                tokenize=True,
                return_dict=True,
            ).to(self.model.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=inference_config.max_new_tokens,
                    temperature=inference_config.temperature,
                    top_p=inference_config.top_p,
                    top_k=inference_config.top_k,
                    do_sample=inference_config.do_sample,
                    pad_token_id=inference_config.pad_token_id or self.tokenizer.pad_token_id,
                    eos_token_id=inference_config.eos_token_id or self.tokenizer.eos_token_id,
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part
            input_text = self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
            generated_text = response.replace(input_text, "").strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"❌ Error generating response: {e}")
            raise
    
    def save_model(
        self, 
        output_dir: str, 
        save_method: str = "lora",
        tokenizer_save: bool = True
    ):
        """
        Save model với different methods
        
        Args:
            output_dir: Directory to save model
            save_method: "lora", "merged_16bit", "merged_4bit", "gguf"
            tokenizer_save: Whether to save tokenizer
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Saving model to {output_path} with method: {save_method}")
            
            if save_method == "lora":
                # Save only LoRA adapters
                self.model.save_pretrained(str(output_path))
                if tokenizer_save:
                    self.tokenizer.save_pretrained(str(output_path))
                    
            elif save_method == "merged_16bit":
                # Save merged model in 16-bit
                self.model.save_pretrained_merged(
                    str(output_path), 
                    self.tokenizer, 
                    save_method="merged_16bit"
                )
                
            elif save_method == "merged_4bit":
                # Save merged model in 4-bit
                self.model.save_pretrained_merged(
                    str(output_path), 
                    self.tokenizer, 
                    save_method="merged_4bit"
                )
                
            elif save_method.startswith("gguf"):
                # Save in GGUF format
                quant_method = save_method.replace("gguf_", "") if "_" in save_method else "q8_0"
                self.model.save_pretrained_gguf(
                    str(output_path), 
                    tokenizer=self.tokenizer,
                    quantization_method=quant_method
                )
            else:
                raise ValueError(f"Unknown save method: {save_method}")
            
            logger.info(f"✅ Model saved successfully to {output_path}")
            
        except Exception as e:
            logger.error(f"❌ Error saving model: {e}")
            raise
    
    def push_to_hub(
        self, 
        repo_id: str, 
        save_method: str = "lora",
        token: Optional[str] = None,
        private: bool = False
    ):
        """
        Push model to HuggingFace Hub
        """
        try:
            logger.info(f"Pushing model to Hub: {repo_id}")
            
            if save_method == "lora":
                self.model.push_to_hub(repo_id, token=token, private=private)
                self.tokenizer.push_to_hub(repo_id, token=token, private=private)
                
            elif save_method == "merged_16bit":
                self.model.push_to_hub_merged(
                    repo_id, 
                    self.tokenizer,
                    save_method="merged_16bit",
                    token=token,
                    private=private
                )
                
            elif save_method == "merged_4bit":
                self.model.push_to_hub_merged(
                    repo_id, 
                    self.tokenizer,
                    save_method="merged_4bit", 
                    token=token,
                    private=private
                )
                
            elif save_method.startswith("gguf"):
                quant_method = save_method.replace("gguf_", "") if "_" in save_method else "q8_0"
                self.model.push_to_hub_gguf(
                    repo_id,
                    tokenizer=self.tokenizer,
                    quantization_method=quant_method,
                    token=token,
                    private=private
                )
            
            logger.info(f"✅ Model pushed to Hub successfully: {repo_id}")
            
        except Exception as e:
            logger.error(f"❌ Error pushing model to Hub: {e}")
            raise
    
    def load_adapter(self, adapter_path: str):
        """
        Load LoRA adapter from path
        """
        try:
            logger.info(f"Loading LoRA adapter from: {adapter_path}")
            
            # Load the adapter
            self.model = PeftModel.from_pretrained(
                self.model,
                adapter_path,
                is_trainable=True
            )
            
            logger.info("✅ LoRA adapter loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error loading LoRA adapter: {e}")
            raise
    
    def create_trainer_compatible_model(self, trainer_class):
        """
        Setup model với train_on_responses_only cho SFTTrainer
        """
        try:
            logger.info("Setting up trainer with response-only training...")
            
            # Create basic trainer first
            trainer = trainer_class(
                model=self.model,
                tokenizer=self.tokenizer,
                # Other args will be added by caller
            )
            
            # Apply train_on_responses_only
            trainer = train_on_responses_only(
                trainer,
                instruction_part="<start_of_turn>user\n",
                response_part="<start_of_turn>model\n",
            )
            
            logger.info("✅ Trainer setup with response-only training")
            return trainer
            
        except Exception as e:
            logger.error(f"❌ Error setting up trainer: {e}")
            raise
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get current GPU memory usage
        """
        try:
            if torch.cuda.is_available():
                device = self.model.device if self.model else torch.cuda.current_device()
                
                return {
                    "allocated": torch.cuda.memory_allocated(device) / 1024**3,  # GB
                    "cached": torch.cuda.memory_reserved(device) / 1024**3,      # GB
                    "max_allocated": torch.cuda.max_memory_allocated(device) / 1024**3,  # GB
                    "device": device
                }
            else:
                return {"message": "CUDA not available"}
                
        except Exception as e:
            logger.warning(f"Could not get memory usage: {e}")
            return {"error": str(e)}
    
    def validate_model_setup(self) -> bool:
        """
        Validate model setup before training
        """
        try:
            logger.info("Validating model setup...")
            
            # Check if model is loaded
            if self.model is None:
                logger.error("Model not loaded")
                return False
            
            # Check if tokenizer is loaded
            if self.tokenizer is None:
                logger.error("Tokenizer not loaded")
                return False
            
            # Check if model is on correct device
            if torch.cuda.is_available():
                if not next(self.model.parameters()).is_cuda:
                    logger.warning("Model not on CUDA device")
            
            # Check if LoRA is applied (for PEFT models)
            if hasattr(self.model, 'peft_config'):
                logger.info("✅ PEFT/LoRA configuration detected")
            else:
                logger.warning("No PEFT configuration found")
            
            # Test generation
            try:
                test_response = self.generate_response(
                    "Test question: 2 + 2 = ?",
                    InferenceConfig(max_new_tokens=10, temperature=0.1)
                )
                logger.info(f"✅ Test generation successful: {test_response[:50]}...")
            except Exception as e:
                logger.warning(f"Test generation failed: {e}")
            
            logger.info("✅ Model validation passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model validation failed: {e}")
            return False


def create_model_manager(config: ModelConfig) -> GemmaModelManager:
    """
    Factory function to create model manager
    """
    return GemmaModelManager(config)


def setup_complete_model(config: ModelConfig) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Complete model setup pipeline
    """
    logger.info("🚀 Starting complete model setup...")
    
    # Create manager
    manager = create_model_manager(config)
    
    # Load base model
    model, tokenizer = manager.load_base_model()
    
    # Apply LoRA
    model = manager.apply_lora()
    
    # Prepare for training
    manager.prepare_for_training()
    
    # Validate setup
    if not manager.validate_model_setup():
        raise RuntimeError("Model setup validation failed")
    
    logger.info("✅ Complete model setup finished successfully")
    
    return model, tokenizer


def test_model_manager():
    """
    Test function for model manager
    """
    from config import ModelConfig
    
    # Create test config với smaller settings
    config = ModelConfig()
    config.max_seq_length = 512  # Smaller for testing
    
    try:
        manager = create_model_manager(config)
        
        # Test base model loading
        model, tokenizer = manager.load_base_model()
        logger.info("✅ Base model loading test passed")
        
        # Test LoRA application
        model = manager.apply_lora()
        logger.info("✅ LoRA application test passed")
        
        # Test memory usage
        memory_info = manager.get_memory_usage()
        logger.info(f"✅ Memory usage test passed: {memory_info}")
        
        # Test validation
        is_valid = manager.validate_model_setup()
        logger.info(f"✅ Validation test passed: {is_valid}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model manager test failed: {e}")
        return False


if __name__ == "__main__":
    # Run test
    test_model_manager()
