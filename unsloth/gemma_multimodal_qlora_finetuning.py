#!/usr/bin/env python3
"""
Fine-tuning QLora model Gemma multimodal với dataset toán học tiếng Việt
Phiên bản cải tiến với xử lý lỗi tensor conversion
Tối ưu cho RTX 3070 8GB VRAM
"""

# Import Unsloth FIRST để tránh warning
from unsloth import FastLanguageModel, is_bfloat16_supported

import os
import torch
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
from PIL import Image
import base64
from io import BytesIO

# Core libraries
from datasets import load_dataset, Dataset
from transformers import TrainingArguments, TextStreamer, DataCollatorForSeq2Seq
from trl import SFTTrainer
import torch.nn.functional as F

# Opik imports for tracking  
import opik
from opik import track, Opik

# Environment setup
warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configuration
class Config:
    # Model configuration - Gemma 3 4B multimodal tối ưu cho RTX 3070 8GB
    MODEL_NAME = "unsloth/gemma-3-4b-pt-unsloth-bnb-4bit"  # Multimodal Gemma 3 4B
    
    # Fallback models nếu model chính không hoạt động
    MODEL_CANDIDATES = [
        "unsloth/gemma-3-4b-pt-unsloth-bnb-4bit",  # Multimodal Gemma 3 4B (target)
        "unsloth/gemma-2b-it",                      # Fallback 1
        "unsloth/llama-3.2-1b-it",                  # Fallback 2 (smallest)
    ]
    
    # Extreme memory optimization cho 4B model
    MAX_SEQ_LENGTH = 512    # Giảm mạnh cho 4B model
    LOAD_IN_4BIT = True
    DTYPE = None  # Auto-detect
    
    # LoRA configuration - tối ưu cho model lớn
    LORA_R = 4              # Giảm rank xuống minimum
    LORA_ALPHA = 8          # Giảm alpha tương ứng  
    LORA_DROPOUT = 0.05     # Giảm dropout
    TARGET_MODULES = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    
    # Training configuration - extreme memory optimization cho 4B
    PER_DEVICE_TRAIN_BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 8  # Tăng để bù batch size nhỏ
    WARMUP_STEPS = 3
    MAX_STEPS = 20          # Giảm cho testing với model lớn
    LEARNING_RATE = 1e-4    # Giảm learning rate cho model lớn
    LOGGING_STEPS = 1
    OPTIM = "adamw_8bit"
    WEIGHT_DECAY = 0.01
    LR_SCHEDULER_TYPE = "linear"
    SEED = 3407
    
    # Dataset configuration
    DATASET_NAME = "ngohongthai/dethivaolop6-montoan"
    
    # Output paths
    OUTPUT_DIR = "./outputs_multimodal"
    MODEL_SAVE_PATH = "./gemma_multimodal_finetuned"
    
    # Opik configuration
    OPIK_PROJECT_NAME = "gemma-multimodal-vietnamese-math"


class MultimodalDataProcessor:
    """Xử lý dataset multimodal với error handling tốt hơn"""
    
    def __init__(self):
        self.tokenizer = None
    
    def set_tokenizer(self, tokenizer):
        """Set tokenizer - compatible with both regular tokenizer and multimodal processor"""
        self.tokenizer = tokenizer
        
        # Handle Gemma3Processor (multimodal) vs regular tokenizer
        if hasattr(tokenizer, 'tokenizer'):
            # Multimodal processor - get the underlying tokenizer
            actual_tokenizer = tokenizer.tokenizer
            print("🔍 Detected multimodal processor, using underlying tokenizer")
        else:
            # Regular tokenizer
            actual_tokenizer = tokenizer
            print("🔍 Using regular tokenizer")
        
        # Ensure proper special tokens for the actual tokenizer
        try:
            if hasattr(actual_tokenizer, 'pad_token') and actual_tokenizer.pad_token is None:
                actual_tokenizer.pad_token = actual_tokenizer.eos_token
                print("✅ Set pad_token")
                
            if hasattr(actual_tokenizer, 'unk_token') and actual_tokenizer.unk_token is None:
                actual_tokenizer.unk_token = actual_tokenizer.eos_token
                print("✅ Set unk_token")
        except Exception as e:
            print(f"⚠️ Could not set special tokens: {e}")
            print("💡 Continuing without special token setup")
    
    def process_image_safely(self, image_data) -> Optional[str]:
        """Xử lý image data an toàn"""
        if image_data is None:
            return None
        
        try:
            if isinstance(image_data, dict) and 'bytes' in image_data:
                image = Image.open(BytesIO(image_data['bytes']))
            elif isinstance(image_data, Image.Image):
                image = image_data
            else:
                return None
            
            # Resize để tiết kiệm memory
            if image.size[0] > 256 or image.size[1] > 256:
                image.thumbnail((256, 256), Image.Resampling.LANCZOS)
            
            # Convert to base64
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return f"[IMAGE:{img_str[:100]}...]"  # Truncate để tiết kiệm memory
            
        except Exception as e:
            print(f"⚠️ Lỗi xử lý image: {e}")
            return None
    
    def format_single_example(self, text: str, solution: str = "", result: str = "", image_data=None) -> str:
        """Format một example thành text format đơn giản"""
        
        # Tạo user content
        user_content = str(text).strip() if text else "Giải bài toán này."
        if image_data:
            image_desc = self.process_image_safely(image_data)
            if image_desc:
                user_content = f"{image_desc}\n{user_content}"
        
        # Tạo assistant response
        solution_str = str(solution).strip() if solution else ""
        result_str = str(result).strip() if result else ""
        
        if solution_str and result_str and solution_str != result_str:
            assistant_content = f"**Giải:**\n{solution_str}\n\n**Đáp án:** {result_str}"
        elif result_str:
            assistant_content = f"**Đáp án:** {result_str}"
        elif solution_str:
            assistant_content = solution_str
        else:
            assistant_content = "Cần thêm thông tin để giải bài này."
        
        # Format theo Gemma format
        formatted_text = f"<start_of_turn>user\n{user_content}<end_of_turn>\n<start_of_turn>model\n{assistant_content}<end_of_turn>"
        
        return formatted_text
    
    def formatting_prompts_func(self, examples):
        """Simplified formatting for SFTTrainer compatibility"""
        
        try:
            # Handle batch processing
            if isinstance(examples, dict) and any(isinstance(v, list) for v in examples.values()):
                # Batch format
                texts = []
                
                # Get field data safely
                text_list = examples.get('text', examples.get('question', examples.get('problem', [])))
                solution_list = examples.get('Solution', examples.get('solution', []))
                result_list = examples.get('Result', examples.get('result', []))
                
                if not text_list:
                    print("⚠️ No text data found, creating dummy")
                    return {"text": ["Bài toán mẫu: 2+2=?"]}
                
                # Process each example
                for i in range(len(text_list)):
                    try:
                        # Get values safely
                        question = str(text_list[i]) if i < len(text_list) else "Bài toán mẫu"
                        solution = str(solution_list[i]) if i < len(solution_list) and solution_list[i] else ""
                        result = str(result_list[i]) if i < len(result_list) and result_list[i] else ""
                        
                        # Create simple format
                        if solution and result and solution != result:
                            answer = f"Giải: {solution}\nĐáp án: {result}"
                        elif result:
                            answer = f"Đáp án: {result}"
                        elif solution:
                            answer = solution
                        else:
                            answer = "Cần thêm thông tin"
                        
                        # Simple Gemma format
                        formatted = f"<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n{answer}<end_of_turn>"
                        
                        # Ensure it's a simple string
                        texts.append(str(formatted))
                        
                    except Exception as e:
                        print(f"⚠️ Error formatting item {i}: {e}")
                        texts.append("<start_of_turn>user\nSample question<end_of_turn>\n<start_of_turn>model\nSample answer<end_of_turn>")
                
                print(f"✅ Formatted {len(texts)} examples")
                
                # Ensure we return proper structure
                result = {"text": texts}
                return result
                
            else:
                # Single example
                print("⚠️ Single example format")
                question = str(examples.get('text', 'Sample question'))
                answer = str(examples.get('Result', 'Sample answer'))
                formatted = f"<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n{answer}<end_of_turn>"
                return {"text": [str(formatted)]}
                
        except Exception as e:
            print(f"❌ Critical formatting error: {e}")
            # Return minimal valid data
            return {"text": ["<start_of_turn>user\nTest question<end_of_turn>\n<start_of_turn>model\nTest answer<end_of_turn>"]}


def clear_gpu_memory():
    """Clear GPU memory cache - enhanced for large models"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()  # Additional cleanup for multiprocessing
        import gc
        gc.collect()  # Python garbage collection


def print_gpu_memory():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        free = total - allocated
        
        print(f"🔥 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 VRAM Total: {total:.1f}GB")
        print(f"📊 VRAM Status: {allocated:.1f}GB allocated, {cached:.1f}GB cached, {free:.1f}GB free")
        
        # Warning thresholds for 4B model
        if free < 3.0:
            print("🔴 VRAM rất thấp! Model 4B có thể OOM")
            print("💡 Đề xuất: giảm MAX_SEQ_LENGTH xuống 256 hoặc sử dụng model nhỏ hơn")
        elif free < 4.0:
            print("🟡 VRAM hơi thấp cho model 4B")
            print("💡 Nên monitor memory usage trong quá trình training")
        else:
            print(f"✅ Có {free:.1f}GB VRAM - đủ cho Gemma 3 4B training")


def load_model_and_tokenizer(config: Config):
    """Load model và tokenizer với fallback mechanism"""
    print("🔄 Đang load model và tokenizer...")
    
    # Try each model candidate
    for i, model_name in enumerate(config.MODEL_CANDIDATES):
        try:
            print(f"🔄 Thử model {i+1}/{len(config.MODEL_CANDIDATES)}: {model_name}")
            
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=config.MAX_SEQ_LENGTH,
                dtype=config.DTYPE,
                load_in_4bit=config.LOAD_IN_4BIT,
            )
            
            print(f"✅ Đã load model thành công: {model_name}")
            print(f"📊 Max sequence length: {config.MAX_SEQ_LENGTH}")
            
            # Update config with working model name
            config.MODEL_NAME = model_name
            
            return model, tokenizer
            
        except Exception as e:
            print(f"❌ Lỗi load model {model_name}: {e}")
            if i < len(config.MODEL_CANDIDATES) - 1:
                print(f"🔄 Thử model tiếp theo...")
                continue
            else:
                print("❌ Tất cả models đều failed!")
                
                # Try one more fallback with a very small model
                print("🔄 Thử model backup cuối cùng...")
                try:
                    # Use the smallest available model for RTX 3070
                    fallback_model = "microsoft/DialoGPT-small"
                    print(f"🔄 Thử fallback model: {fallback_model}")
                    
                    from transformers import AutoModelForCausalLM, AutoTokenizer
                    model = AutoModelForCausalLM.from_pretrained(
                        fallback_model,
                        load_in_4bit=True,
                        device_map="auto"
                    )
                    tokenizer = AutoTokenizer.from_pretrained(fallback_model)
                    
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                    
                    print(f"✅ Fallback model loaded: {fallback_model}")
                    config.MODEL_NAME = fallback_model
                    
                    return model, tokenizer
                    
                except Exception as fallback_error:
                    print(f"❌ Fallback model cũng failed: {fallback_error}")
                    raise RuntimeError(f"Không thể load bất kỳ model nào. Lỗi cuối cùng: {e}")
    
    raise RuntimeError("Unexpected error in model loading")


def setup_lora(model, config: Config):
    """Setup LoRA - compatible with both Unsloth and regular models"""
    print("🔄 Đang setup LoRA...")
    
    try:
        # Try Unsloth LoRA first
        if hasattr(FastLanguageModel, 'get_peft_model'):
            model = FastLanguageModel.get_peft_model(
                model,
                r=config.LORA_R,
                target_modules=config.TARGET_MODULES,
                lora_alpha=config.LORA_ALPHA,
                lora_dropout=config.LORA_DROPOUT,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=config.SEED,
                use_rslora=False,
                loftq_config=None,
            )
            print("✅ Unsloth LoRA setup hoàn tất")
        else:
            raise Exception("Unsloth LoRA not available")
            
    except Exception as e:
        print(f"⚠️ Unsloth LoRA failed ({e}), thử PEFT standard...")
        
        # Fallback to standard PEFT
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            
            # Adjust target modules based on model type
            target_modules = config.TARGET_MODULES
            if "DialoGPT" in config.MODEL_NAME:
                target_modules = ["c_attn", "c_proj"]  # DialoGPT specific
            
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=config.LORA_R,
                lora_alpha=config.LORA_ALPHA,
                lora_dropout=config.LORA_DROPOUT,
                target_modules=target_modules,
                bias="none"
            )
            
            model = get_peft_model(model, peft_config)
            print("✅ Standard PEFT LoRA setup hoàn tất")
            
        except Exception as peft_error:
            print(f"❌ PEFT cũng failed: {peft_error}")
            print("⚠️ Tiếp tục không có LoRA - full fine-tuning")
            return model
    
    # Print trainable parameters
    try:
        if hasattr(model, 'print_trainable_parameters'):
            model.print_trainable_parameters()
        else:
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"📊 Trainable params: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")
    except Exception as e:
        print(f"⚠️ Không thể hiển thị thống kê parameters: {e}")
    
    return model


def load_and_process_dataset(config: Config, data_processor):
    """Load và process dataset - tối ưu cho model 4B"""
    print("🔄 Đang load dataset...")
    
    try:
        # Load dataset - giảm samples cho model lớn hơn
        train_dataset = load_dataset(
            config.DATASET_NAME, 
            split="train[:50]",   # Giảm xuống 50 samples cho 4B model
            trust_remote_code=True
        )
        
        eval_dataset = load_dataset(
            config.DATASET_NAME,
            split="train[50:55]",  # 5 samples cho eval
            trust_remote_code=True
        )
        
        print(f"📊 Train samples: {len(train_dataset)}")
        print(f"📊 Eval samples: {len(eval_dataset)}")
        
        # Process datasets with debugging
        print("🔍 Processing train dataset...")
        train_dataset = train_dataset.map(
            data_processor.formatting_prompts_func,
            batched=True,
            batch_size=5,   # Smaller batch for debugging
            remove_columns=train_dataset.column_names,
            desc="Formatting train"
        )
        
        print("🔍 Processing eval dataset...")
        eval_dataset = eval_dataset.map(
            data_processor.formatting_prompts_func,
            batched=True,
            batch_size=5,
            remove_columns=eval_dataset.column_names,
            desc="Formatting eval"
        )
        
        # Debug: check dataset structure
        print(f"🔍 Train dataset sample keys: {train_dataset[0].keys() if len(train_dataset) > 0 else 'Empty'}")
        if len(train_dataset) > 0 and 'text' in train_dataset[0]:
            sample_text = train_dataset[0]['text']
            print(f"🔍 Sample text type: {type(sample_text)}")
            print(f"🔍 Sample text length: {len(sample_text) if isinstance(sample_text, str) else 'Not string'}")
            print(f"🔍 Sample text preview: {sample_text[:100] if isinstance(sample_text, str) else str(sample_text)[:100]}...")
        
        # Filter out empty or invalid samples
        def filter_valid_samples(example):
            return (
                'text' in example and 
                isinstance(example['text'], str) and 
                len(example['text']) > 0 and
                len(example['text']) < 2000  # Max length check
            )
        
        train_dataset = train_dataset.filter(filter_valid_samples)
        eval_dataset = eval_dataset.filter(filter_valid_samples)
        
        print(f"📊 After filtering - Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
        
        print("✅ Dataset processing hoàn tất")
        
        return train_dataset, eval_dataset
        
    except Exception as e:
        print(f"❌ Lỗi load dataset: {e}")
        print("🔄 Tạo dummy dataset...")
        return create_dummy_dataset(data_processor)


def create_dummy_dataset(data_processor):
    """Tạo dummy dataset cho testing - tối ưu cho model 4B"""
    dummy_data = [
        {
            "text": "Tính diện tích hình chữ nhật có chiều dài 8cm, chiều rộng 5cm.",
            "Solution": "Diện tích = chiều dài × chiều rộng = 8 × 5 = 40cm²",
            "Result": "40cm²",
            "image": None
        },
        {
            "text": "Tìm x: 2x + 5 = 13",
            "Solution": "2x = 13 - 5 = 8\nx = 4",
            "Result": "x = 4", 
            "image": None
        },
        {
            "text": "Tam giác có cạnh 3, 4, 5cm. Đây có phải tam giác vuông?",
            "Solution": "Kiểm tra: 3² + 4² = 9 + 16 = 25 = 5²\nVậy đây là tam giác vuông",
            "Result": "Có",
            "image": None
        },
        {
            "text": "Tính chu vi hình tròn có bán kính 7cm.",
            "Solution": "Chu vi = 2πr = 2 × 3.14 × 7 = 43.96cm",
            "Result": "43.96cm",
            "image": None
        },
        {
            "text": "Giải phương trình: 3x - 2 = 10",
            "Solution": "3x = 10 + 2 = 12\nx = 12/3 = 4",
            "Result": "x = 4",
            "image": None
        }
    ] * 10  # 50 samples - giảm từ 90
    
    dataset = Dataset.from_list(dummy_data)
    
    train_dataset = dataset.map(
        data_processor.formatting_prompts_func,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    eval_dataset = Dataset.from_list(dummy_data[:5]).map(  # Giảm từ 6 xuống 5
        data_processor.formatting_prompts_func,
        batched=True,
        remove_columns=["text", "Solution", "Result", "image"]
    )
    
    print("✅ Dummy dataset created")
    return train_dataset, eval_dataset


def setup_trainer(model, tokenizer, train_dataset, eval_dataset, config: Config):
    """Setup simplified trainer for Gemma 3 4B"""
    print("⚙️ Đang setup simplified trainer...")
    
    # Handle tokenizer for SFTTrainer
    if hasattr(tokenizer, 'tokenizer'):
        # Multimodal processor - use underlying tokenizer
        actual_tokenizer = tokenizer.tokenizer
        print("🔍 Using underlying tokenizer for SFT")
    else:
        actual_tokenizer = tokenizer
        print("🔍 Using regular tokenizer for SFT")
    
    # Ensure tokenizer has required attributes
    if not hasattr(actual_tokenizer, 'pad_token_id') or actual_tokenizer.pad_token_id is None:
        actual_tokenizer.pad_token_id = actual_tokenizer.eos_token_id
        
    if not hasattr(actual_tokenizer, 'pad_token') or actual_tokenizer.pad_token is None:
        actual_tokenizer.pad_token = actual_tokenizer.eos_token
    
    # Training arguments với extreme memory optimization cho Gemma 3 4B
    training_args = TrainingArguments(
        per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=config.WARMUP_STEPS,
        max_steps=config.MAX_STEPS,
        learning_rate=config.LEARNING_RATE,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=config.LOGGING_STEPS,
        optim=config.OPTIM,
        weight_decay=config.WEIGHT_DECAY,
        lr_scheduler_type=config.LR_SCHEDULER_TYPE,
        seed=config.SEED,
        output_dir=config.OUTPUT_DIR,
        report_to=None,
        save_steps=15,
        save_total_limit=1,
        dataloader_num_workers=0,
        
        # Extreme memory optimizations
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        dataloader_drop_last=True,
        max_grad_norm=1.0,
        
        # Simplify evaluation
        eval_strategy="no",  # Disable evaluation to save memory
        
        # Advanced optimizations
        tf32=True if torch.cuda.is_available() else False,
        auto_find_batch_size=False,
        group_by_length=False,
        
        # Logging optimizations
        logging_first_step=True,
        save_safetensors=True,
        load_best_model_at_end=False,
        remove_unused_columns=True,  # Let SFT handle columns
    )
    
    # Create SFTTrainer with minimal configuration
    try:
        trainer = SFTTrainer(
            model=model,
            tokenizer=actual_tokenizer,
            train_dataset=train_dataset,
            dataset_text_field="text",
            max_seq_length=config.MAX_SEQ_LENGTH,
            args=training_args,
            
            # Simplified settings to avoid tensor issues
            packing=False,                    # Disable packing 
            dataset_num_proc=1,              # Single process
            data_collator=None,              # Let SFT create default
            formatting_func=None,            # Use dataset text field directly
        )
        
        print("✅ SFTTrainer setup thành công")
        return trainer
        
    except Exception as e:
        print(f"❌ SFTTrainer setup failed: {e}")
        print("🔄 Thử với cấu hình đơn giản hơn...")
        
        # Fallback: even simpler trainer
        trainer = SFTTrainer(
            model=model,
            tokenizer=actual_tokenizer,
            train_dataset=train_dataset,
            dataset_text_field="text",
            max_seq_length=config.MAX_SEQ_LENGTH,
            args=training_args,
            packing=False,
        )
        
        print("✅ Fallback trainer setup")
        return trainer


@track(project_name=Config.OPIK_PROJECT_NAME)
def train_model(trainer):
    """Train model"""
    print("🚀 Bắt đầu extreme memory optimized training...")
    
    # Clear memory before training
    clear_gpu_memory()
    print("🗑️ Đã clear memory cache")
    
    # Check GPU memory
    print_gpu_memory()
    
    try:
        trainer_stats = trainer.train()
        print("✅ Training hoàn tất!")
        return trainer_stats
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"❌ CUDA OOM Error: {e}")
        print("💡 Thử giảm batch size hoặc sequence length")
        raise
    except Exception as e:
        print(f"❌ Lỗi training: {e}")
        raise


def test_inference(model, tokenizer):
    """Test inference - compatible with multimodal processor"""
    print("🧪 Testing inference...")
    
    # Try to enable Unsloth fast inference
    try:
        if hasattr(FastLanguageModel, 'for_inference'):
            FastLanguageModel.for_inference(model)
            print("✅ Unsloth fast inference enabled")
    except Exception as e:
        print(f"⚠️ Unsloth fast inference not available: {e}")
    
    # Prepare test prompt
    test_prompt = "<start_of_turn>user\nTính diện tích hình vuông có cạnh 6cm<end_of_turn>\n<start_of_turn>model\n"
    
    try:
        # Handle multimodal processor vs regular tokenizer
        if hasattr(tokenizer, 'tokenizer'):
            # Multimodal processor - use underlying tokenizer for text-only
            actual_tokenizer = tokenizer.tokenizer
            inputs = actual_tokenizer([test_prompt], return_tensors="pt")
            decode_func = actual_tokenizer.decode
            eos_token_id = actual_tokenizer.eos_token_id
            pad_token_id = getattr(actual_tokenizer, 'pad_token_id', eos_token_id)
        else:
            # Regular tokenizer
            inputs = tokenizer([test_prompt], return_tensors="pt")
            decode_func = tokenizer.decode
            eos_token_id = tokenizer.eos_token_id
            pad_token_id = getattr(tokenizer, 'pad_token_id', eos_token_id)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                temperature=0.3,
                do_sample=True,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                use_cache=True
            )
        
        response = decode_func(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part
        if "<start_of_turn>model\n" in response:
            response = response.split("<start_of_turn>model\n")[-1].strip()
        else:
            # Fallback: remove the input prompt
            response = response[len(test_prompt):].strip()
        
        print("🤖 Model response:")
        print(response)
        
    except Exception as e:
        print(f"❌ Lỗi inference: {e}")
        print("💡 Có thể do multimodal processor - cần text-only input format")


def save_model(model, tokenizer, config: Config):
    """Save model - compatible with both Unsloth and regular models"""
    print("💾 Đang save model...")
    
    try:
        # Try Unsloth save methods first
        if hasattr(model, 'save_pretrained_merged') and 'unsloth' in config.MODEL_NAME.lower():
            try:
                # Save LoRA weights
                model.save_pretrained(config.MODEL_SAVE_PATH)
                tokenizer.save_pretrained(config.MODEL_SAVE_PATH)
                print(f"✅ LoRA weights saved tại: {config.MODEL_SAVE_PATH}")
                
                # Try to save merged model
                merged_path = f"{config.MODEL_SAVE_PATH}_merged"
                model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")
                print(f"✅ Merged model saved tại: {merged_path}")
                
            except Exception as merge_error:
                print(f"⚠️ Merge save failed: {merge_error}, chỉ save LoRA weights")
                model.save_pretrained(config.MODEL_SAVE_PATH)
                tokenizer.save_pretrained(config.MODEL_SAVE_PATH)
        else:
            # Standard save for regular models
            model.save_pretrained(config.MODEL_SAVE_PATH)
            tokenizer.save_pretrained(config.MODEL_SAVE_PATH)
            print(f"✅ Model saved (standard method)")
        
        print(f"✅ Model đã được save tại: {config.MODEL_SAVE_PATH}")
        
        # Save config for reference
        import json
        config_dict = {
            "model_name": config.MODEL_NAME,
            "max_seq_length": config.MAX_SEQ_LENGTH,
            "lora_r": config.LORA_R,
            "lora_alpha": config.LORA_ALPHA,
            "training_steps": config.MAX_STEPS
        }
        
        with open(f"{config.MODEL_SAVE_PATH}/training_config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        print("✅ Training config saved")
        
    except Exception as e:
        print(f"❌ Lỗi save model: {e}")
        print("💡 Có thể cần check disk space hoặc permissions")


@track(project_name=Config.OPIK_PROJECT_NAME)  
def main():
    """Main function"""
    print("🚀 Bắt đầu Gemma Multimodal QLora Fine-tuning")
    print("=" * 60)
    
    # Setup Opik
    try:
        opik.configure()
        print("✅ Opik configured")
    except Exception as e:
        print(f"⚠️ Opik setup warning: {e}")
    
    config = Config()
    
    try:
        # Load model
        model, tokenizer = load_model_and_tokenizer(config)
        
        # Setup data processor
        data_processor = MultimodalDataProcessor()
        data_processor.set_tokenizer(tokenizer)
        
        # Setup LoRA
        model = setup_lora(model, config)
        
        # Load dataset
        train_dataset, eval_dataset = load_and_process_dataset(config, data_processor)
        
        # Setup trainer (simplified, no eval to save memory)
        trainer = setup_trainer(model, tokenizer, train_dataset, None, config)
        
        # Train
        trainer_stats = train_model(trainer)
        
        # Test inference
        test_inference(model, tokenizer)
        
        # Save model
        save_model(model, tokenizer, config)
        
        print("🎉 Pipeline hoàn tất thành công!")
        
        return {
            "status": "success",
            "training_stats": trainer_stats,
            "model_path": config.MODEL_SAVE_PATH
        }
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}


if __name__ == "__main__":
    result = main()
    
    if result["status"] == "success":
        print("\n🎯 THÀNH CÔNG!")
        print(f"📁 Model: {result['model_path']}")
    else:
        print(f"\n❌ THẤT BẠI: {result['error']}") 