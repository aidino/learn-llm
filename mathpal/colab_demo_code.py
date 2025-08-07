# 🚀 Gemma3N Fine-tuning Demo Code for Google Colab
# Copy từng section này vào các cell riêng biệt trong Colab

# =============================================================================
# CELL 1: Installation (Code cell)
# =============================================================================
"""
%%capture
# Install dependencies for Google Colab
import os

print("🔧 Installing for Google Colab...")

# Install Unsloth for Colab
!pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl triton cut_cross_entropy unsloth_zoo
!pip install sentencepiece protobuf "datasets>=3.4.1,<4.0.0" "huggingface_hub>=0.34.0" hf_transfer
!pip install --no-deps unsloth

# Install transformers (specific version for Unsloth compatibility)
!pip install --no-deps --upgrade timm
!pip install "transformers>=4.52.4,<4.53.0"

# Install Comet ML
!pip install comet-ml

# Other dependencies
!pip install pandas numpy tqdm psutil packaging Pillow

print("✅ Installation completed!")
"""

# =============================================================================
# CELL 2: Import Libraries (Code cell)
# =============================================================================
import os
import re
import io
import json
import time
import warnings
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field

# ML libraries
import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

# Transformers & Training
from transformers import TrainingArguments, EarlyStoppingCallback
from datasets import Dataset, load_dataset
from trl import SFTTrainer, SFTConfig

# Unsloth
from unsloth import FastModel, get_chat_template
from unsloth.chat_templates import train_on_responses_only

# Comet ML
import comet_ml

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

print("🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.")
print("📊 Comet ML: Experiment tracking ready.")
print("✅ All libraries imported successfully!")

# =============================================================================
# CELL 3: Configuration Classes (Code cell)
# =============================================================================
@dataclass
class CometConfig:
    """Comet ML configuration với tính năng model registry"""
    # API Keys - sẽ được set từ Colab secrets
    api_key: Optional[str] = None
    workspace: Optional[str] = None
    project: Optional[str] = None
    
    # Experiment settings
    experiment_name: str = "gemma3n-math-tutor-colab-demo"
    tags: List[str] = field(default_factory=lambda: [
        "gemma3n", "math-tutor", "vietnamese", "sixth-grade", 
        "fine-tuning", "unsloth", "colab-demo"
    ])
    
    # Logging settings
    auto_metric_logging: bool = True
    auto_param_logging: bool = True
    auto_histogram_weight_logging: bool = True
    auto_histogram_gradient_logging: bool = True
    auto_histogram_activation_logging: bool = False
    
    # Model Registry
    log_model: bool = True
    model_registry_name: str = "gemma3n-math-tutor-colab"

@dataclass
class ModelConfig:
    """Model configuration tối ưu cho Colab T4"""
    # Base model
    model_name: str = "unsloth/gemma-3n-E4B-it"
    max_seq_length: int = 1536  # Optimized for T4
    load_in_4bit: bool = True
    full_finetuning: bool = False
    
    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_bias: str = "none"
    
    # Target modules
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Memory optimization
    use_gradient_checkpointing: str = "unsloth"
    random_state: int = 42

@dataclass
class TrainingConfig:
    """Training configuration cho Colab demo"""
    # Output
    output_dir: str = "outputs/gemma3n-math-colab"
    run_name: str = "gemma3n-colab-demo"
    
    # Training schedule
    num_train_epochs: int = 2
    max_steps: int = -1
    eval_strategy: str = "steps"
    eval_steps: int = 25
    
    # Batch settings - optimized for T4
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    
    # Learning rate
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    
    # Optimization
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Precision
    fp16: bool = True
    bf16: bool = False
    
    # Logging & Saving
    logging_steps: int = 5
    save_strategy: str = "steps"
    save_steps: int = 50
    save_total_limit: int = 2
    
    # Early stopping
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    early_stopping_patience: int = 3
    
    # Memory
    remove_unused_columns: bool = True
    dataloader_num_workers: int = 2
    dataloader_pin_memory: bool = True
    
    # Reproducibility
    seed: int = 42
    data_seed: int = 42
    
    # Report
    report_to: str = "comet_ml"

@dataclass
class DatasetConfig:
    """Dataset configuration"""
    dataset_name: str = "ngohongthai/exam-sixth_grade-instruct-dataset"
    train_split: str = "train"
    test_split: str = "test"
    max_samples: Optional[int] = 200  # Limit for demo
    dataset_text_field: str = "text"
    dataset_num_proc: int = 2

@dataclass
class DemoConfig:
    """Complete demo configuration"""
    comet: CometConfig = field(default_factory=CometConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

print("✅ Configuration classes defined!")

# =============================================================================
# CELL 4: Setup Comet ML Credentials (Code cell)
# =============================================================================
# Setup Comet ML credentials từ Colab secrets
try:
    from google.colab import userdata
    
    # Load từ Colab secrets
    COMET_API_KEY = userdata.get('COMET_API_KEY')
    COMET_WORKSPACE = userdata.get('COMET_WORKSPACE') 
    COMET_PROJECT = userdata.get('COMET_PROJECT', 'mathpal-gemma3n-demo')
    
    # Set environment variables
    os.environ['COMET_API_KEY'] = COMET_API_KEY
    os.environ['COMET_WORKSPACE'] = COMET_WORKSPACE
    os.environ['COMET_PROJECT'] = COMET_PROJECT
    
    print("✅ Comet ML credentials loaded from Colab secrets")
    print(f"   Workspace: {COMET_WORKSPACE}")
    print(f"   Project: {COMET_PROJECT}")
    
except Exception as e:
    print("⚠️ Could not load Comet ML credentials from secrets")
    print("   Please add COMET_API_KEY, COMET_WORKSPACE to Colab secrets")
    print("   Or set them manually below:")
    
    # Manual setup (uncomment and fill in)
    # os.environ['COMET_API_KEY'] = "your-api-key-here"
    # os.environ['COMET_WORKSPACE'] = "your-workspace-here"
    # os.environ['COMET_PROJECT'] = "mathpal-gemma3n-demo"
    
    COMET_API_KEY = os.getenv('COMET_API_KEY')
    COMET_WORKSPACE = os.getenv('COMET_WORKSPACE')
    COMET_PROJECT = os.getenv('COMET_PROJECT', 'mathpal-gemma3n-demo')

# Optional: HuggingFace token for model upload
try:
    from google.colab import userdata
    HF_TOKEN = userdata.get('HF_TOKEN')
    if HF_TOKEN:
        os.environ['HF_TOKEN'] = HF_TOKEN
        print("✅ HuggingFace token loaded")
except:
    print("ℹ️ HuggingFace token not found (optional)")

# =============================================================================
# CELL 5: Create Configuration (Code cell)
# =============================================================================
# Create configuration
config = DemoConfig()

# Update Comet ML settings
config.comet.api_key = COMET_API_KEY
config.comet.workspace = COMET_WORKSPACE
config.comet.project = COMET_PROJECT

# Create output directory
os.makedirs(config.training.output_dir, exist_ok=True)

def print_config_summary(config: DemoConfig):
    """Print configuration summary"""
    print("=" * 60)
    print("🔧 GEMMA3N DEMO CONFIGURATION")
    print("=" * 60)
    
    print(f"📊 Experiment: {config.comet.experiment_name}")
    print(f"🤖 Model: {config.model.model_name}")
    print(f"📚 Dataset: {config.dataset.dataset_name}")
    print(f"💾 Output: {config.training.output_dir}")
    
    print(f"\n⚙️ Training Settings:")
    print(f"   Epochs: {config.training.num_train_epochs}")
    print(f"   Batch Size: {config.training.per_device_train_batch_size}")
    print(f"   Gradient Accumulation: {config.training.gradient_accumulation_steps}")
    print(f"   Effective Batch Size: {config.training.per_device_train_batch_size * config.training.gradient_accumulation_steps}")
    print(f"   Learning Rate: {config.training.learning_rate}")
    print(f"   Max Samples: {config.dataset.max_samples}")
    
    print(f"\n🧬 LoRA Settings:")
    print(f"   Rank (r): {config.model.lora_r}")
    print(f"   Alpha: {config.model.lora_alpha}")
    print(f"   Dropout: {config.model.lora_dropout}")
    
    print(f"\n☁️ Comet ML:")
    print(f"   Workspace: {config.comet.workspace}")
    print(f"   Project: {config.comet.project}")
    print(f"   API Key: {'✅ Set' if config.comet.api_key else '❌ Not set'}")
    
    print("=" * 60)

print_config_summary(config)

# =============================================================================
# CELL 6: Helper Functions (Code cell)
# =============================================================================
def process_sample(sample: Dict[str, str]) -> Dict[str, List[Dict[str, Any]]]:
    """Convert sample to Gemma3N conversation format"""
    conversations = [
        {
            "role": "user",
            "content": [{"type": "text", "text": sample["question"]}]
        },
        {
            "role": "assistant", 
            "content": [{"type": "text", "text": sample["solution"]}]
        }
    ]
    return {"conversations": conversations}

def setup_comet_experiment(config: DemoConfig):
    """Setup Comet ML experiment"""
    if not config.comet.api_key:
        print("⚠️ Comet ML API key not set, skipping experiment tracking")
        return None
    
    try:
        print("📊 Setting up Comet ML experiment...")
        
        # Initialize experiment
        experiment = comet_ml.Experiment(
            api_key=config.comet.api_key,
            workspace=config.comet.workspace,
            project_name=config.comet.project,
            experiment_name=config.comet.experiment_name,
            auto_metric_logging=config.comet.auto_metric_logging,
            auto_param_logging=config.comet.auto_param_logging,
        )
        
        # Add tags
        for tag in config.comet.tags:
            experiment.add_tag(tag)
        
        # Log configuration
        experiment.log_parameter("model_name", config.model.model_name)
        experiment.log_parameter("max_seq_length", config.model.max_seq_length)
        experiment.log_parameter("lora_r", config.model.lora_r)
        experiment.log_parameter("learning_rate", config.training.learning_rate)
        experiment.log_parameter("batch_size", config.training.per_device_train_batch_size)
        experiment.log_parameter("num_epochs", config.training.num_train_epochs)
        experiment.log_parameter("max_samples", config.dataset.max_samples)
        
        # Set environment variables for transformers integration
        os.environ["COMET_PROJECT_NAME"] = config.comet.project
        if config.comet.workspace:
            os.environ["COMET_WORKSPACE"] = config.comet.workspace
        
        print(f"✅ Comet ML experiment initialized")
        print(f"🔗 Experiment URL: {experiment.url}")
        
        return experiment
        
    except Exception as e:
        print(f"❌ Failed to setup Comet ML: {e}")
        return None

def generate_response(model, tokenizer, question: str, max_new_tokens: int = 256):
    """Generate response for a question"""
    # Format input
    messages = [{
        "role": "user",
        "content": [{"type": "text", "text": question}]
    }]
    
    # Apply chat template
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
        return_dict=True,
    ).to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    input_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    generated_text = response.replace(input_text, "").strip()
    
    return generated_text

print("✅ Helper functions defined!")

# =============================================================================
# CELL 7: Main Training Pipeline (Code cell)
# =============================================================================
def run_complete_pipeline(config: DemoConfig):
    """Run complete training pipeline"""
    print("🎯 Starting Complete Gemma3N Fine-tuning Pipeline")
    print("=" * 80)
    
    # Step 1: Setup Comet ML
    experiment = setup_comet_experiment(config)
    
    # Step 2: Setup Model
    print("\n🤖 STEP 2: Setting up model...")
    model, tokenizer = FastModel.from_pretrained(
        model_name=config.model.model_name,
        max_seq_length=config.model.max_seq_length,
        load_in_4bit=config.model.load_in_4bit,
        full_finetuning=config.model.full_finetuning,
    )
    
    # Setup chat template
    tokenizer = get_chat_template(tokenizer, "gemma-3n")
    
    # Apply LoRA
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=config.model.lora_r,
        lora_alpha=config.model.lora_alpha,
        lora_dropout=config.model.lora_dropout,
        bias=config.model.lora_bias,
        target_modules=config.model.target_modules,
        use_gradient_checkpointing=config.model.use_gradient_checkpointing,
        random_state=config.model.random_state,
    )
    
    # Prepare for training
    FastModel.for_training(model)
    
    print("✅ Model setup complete")
    
    # Step 3: Prepare Data
    print("\n📚 STEP 3: Preparing datasets...")
    raw_dataset = load_dataset(config.dataset.dataset_name)
    
    # Limit samples for demo
    train_dataset = raw_dataset[config.dataset.train_split]
    if config.dataset.max_samples and len(train_dataset) > config.dataset.max_samples:
        train_dataset = train_dataset.select(range(config.dataset.max_samples))
    
    # Process dataset
    train_dataset = train_dataset.map(process_sample, desc="Converting to conversations")
    
    # Apply chat template
    def format_conversations(examples):
        convos = examples["conversations"]
        texts = []
        for convo in convos:
            try:
                formatted_text = tokenizer.apply_chat_template(
                    convo, tokenize=False, add_generation_prompt=False
                ).removeprefix('<bos>')
                texts.append(formatted_text)
            except:
                # Fallback format
                user_text = convo[0]["content"][0]["text"]
                assistant_text = convo[1]["content"][0]["text"]
                fallback = f"<start_of_turn>user\n{user_text}<end_of_turn>\n<start_of_turn>model\n{assistant_text}<end_of_turn>"
                texts.append(fallback)
        return {"text": texts}
    
    train_dataset = train_dataset.map(format_conversations, batched=True, desc="Formatting conversations")
    
    # Create eval split
    train_test = train_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_test["train"]
    eval_dataset = train_test["test"]
    
    print(f"✅ Datasets prepared: Train={len(train_dataset)}, Eval={len(eval_dataset)}")
    
    # Step 4: Create Trainer
    print("\n🏋️ STEP 4: Creating trainer...")
    
    training_args = SFTConfig(
        dataset_text_field="text",
        output_dir=config.training.output_dir,
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        warmup_ratio=config.training.warmup_ratio,
        optim=config.training.optim,
        fp16=config.training.fp16,
        eval_strategy=config.training.eval_strategy,
        eval_steps=config.training.eval_steps,
        save_strategy=config.training.save_strategy,
        save_steps=config.training.save_steps,
        logging_steps=config.training.logging_steps,
        report_to=config.training.report_to,
        load_best_model_at_end=config.training.load_best_model_at_end,
        metric_for_best_model=config.training.metric_for_best_model,
        seed=config.training.seed,
        max_seq_length=config.model.max_seq_length,
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )
    
    # Apply train_on_responses_only
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
    )
    
    # Add early stopping
    early_stopping = EarlyStoppingCallback(early_stopping_patience=3)
    trainer.add_callback(early_stopping)
    
    print("✅ Trainer created")
    
    # Step 5: Training
    print("\n🚀 STEP 5: Running training...")
    start_time = time.time()
    
    train_result = trainer.train()
    
    training_time = time.time() - start_time
    print(f"✅ Training completed in {training_time:.1f} seconds")
    print(f"📈 Final train loss: {train_result.metrics.get('train_loss', 'N/A')}")
    print(f"📈 Final eval loss: {train_result.metrics.get('eval_loss', 'N/A')}")
    
    # Step 6: Save Model
    print("\n💾 STEP 6: Saving model...")
    save_path = Path(config.training.output_dir) / "model_lora"
    model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    print(f"✅ Model saved to {save_path}")
    
    # Log to Comet ML Model Registry
    if experiment:
        try:
            experiment.log_model(
                name=config.comet.model_registry_name,
                file_or_folder=str(save_path),
                metadata={
                    "train_loss": train_result.metrics.get('train_loss'),
                    "eval_loss": train_result.metrics.get('eval_loss'),
                    "training_time": training_time,
                }
            )
            print("✅ Model logged to Comet ML registry")
        except Exception as e:
            print(f"⚠️ Failed to log to registry: {e}")
    
    # Step 7: Test Inference
    print("\n🧠 STEP 7: Testing inference...")
    FastModel.for_inference(model)
    
    test_questions = [
        "Tính 15 + 27 = ?",
        "Một hình chữ nhật có chiều dài 8m và chiều rộng 5m. Tính diện tích?",
        "Tìm x biết: 2x + 5 = 15",
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n--- Test {i} ---")
        print(f"❓ {question}")
        try:
            answer = generate_response(model, tokenizer, question)
            print(f"🤖 {answer}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # End experiment
    if experiment:
        experiment.end()
        print("\n🏁 Comet ML experiment ended")
    
    return {
        "model": model,
        "tokenizer": tokenizer,
        "train_result": train_result,
        "save_path": str(save_path),
        "experiment_url": experiment.url if experiment else None
    }

print("✅ Pipeline function defined!")

# =============================================================================
# CELL 8: RUN TRAINING! (Code cell)
# =============================================================================
# Check GPU
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"🔥 GPU: {gpu_name}")
    print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
    print(f"✅ Ready for training!")
else:
    print("⚠️ No GPU detected. Please enable GPU runtime.")
    print("Go to Runtime > Change runtime type > GPU")

# Run the complete pipeline
if torch.cuda.is_available():
    print("\n" + "=" * 80)
    print("🎯 STARTING GEMMA3N FINE-TUNING DEMO")
    print("=" * 80)
    
    try:
        results = run_complete_pipeline(config)
        
        print("\n🎊 SUCCESS! Training completed successfully!")
        print(f"📈 Model saved to: {results['save_path']}")
        if results['experiment_url']:
            print(f"🔗 View results: {results['experiment_url']}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\n⚠️ GPU not available. Please enable GPU runtime to run training.")

# =============================================================================
# CELL 9: Custom Inference Testing (Code cell)
# =============================================================================
# Test với câu hỏi của riêng bạn
if 'results' in locals() and 'model' in results:
    model = results['model']
    tokenizer = results['tokenizer']
    
    # Ensure model is in inference mode
    FastModel.for_inference(model)
    
    # Your custom questions
    custom_questions = [
        "Tính 25 × 4 = ?",
        "Một lớp học có 35 học sinh. Có 15 học sinh nam. Hỏi có bao nhiêu học sinh nữ?",
        "Tìm x biết: x + 12 = 20",
        # Add your own questions here
    ]
    
    print("🧠 Testing với custom questions:")
    
    for i, question in enumerate(custom_questions, 1):
        print(f"\n--- Custom Test {i} ---")
        print(f"❓ {question}")
        
        try:
            answer = generate_response(model, tokenizer, question, max_new_tokens=200)
            print(f"🤖 {answer}")
        except Exception as e:
            print(f"❌ Error: {e}")
else:
    print("⚠️ No trained model found. Run training first.")

# =============================================================================
# END OF CODE
# =============================================================================
