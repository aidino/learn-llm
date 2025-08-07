"""
Configuration module for Gemma3N fine-tuning with Unsloth and Comet ML
Optimized for T4 GPU with 1000 samples dataset
"""

import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class CometConfig:
    """Comet ML configuration với tính năng model registry"""
    # API Keys - REQUIRED
    api_key: Optional[str] = None
    workspace: Optional[str] = None
    project: Optional[str] = None
    
    # Experiment settings
    experiment_name: str = "gemma3n-math-tutor-baseline"
    tags: List[str] = field(default_factory=lambda: [
        "gemma3n",
        "math-tutor", 
        "vietnamese",
        "sixth-grade",
        "fine-tuning",
        "unsloth"
    ])
    
    # Logging settings
    auto_metric_logging: bool = True
    auto_param_logging: bool = True
    auto_histogram_weight_logging: bool = True
    auto_histogram_gradient_logging: bool = True
    auto_histogram_activation_logging: bool = False  # Tiết kiệm memory
    auto_output_logging: str = "default"
    
    # Model Registry settings
    log_model: bool = True
    log_graph: bool = False  # Có thể chậm với LLM lớn
    log_code: bool = True
    log_git_metadata: bool = True
    model_registry_name: str = "gemma3n-math-tutor"
    
    def __post_init__(self):
        """Load từ environment variables nếu không set"""
        if self.api_key is None:
            self.api_key = os.getenv("COMET_API_KEY")
        if self.workspace is None:
            self.workspace = os.getenv("COMET_WORKSPACE")
        if self.project is None:
            self.project = os.getenv("COMET_PROJECT", "mathpal-gemma3n")


@dataclass
class ModelConfig:
    """Cấu hình model tối ưu cho T4 GPU"""
    # Base model settings
    model_name: str = "unsloth/gemma-3n-E4B-it"
    max_seq_length: int = 2048  # Phù hợp với T4 memory
    load_in_4bit: bool = True   # Quan trọng cho T4
    full_finetuning: bool = False
    
    # LoRA settings - tối ưu cho hiệu suất
    lora_r: int = 16            # Tăng từ 8 để better performance
    lora_alpha: int = 16        # Matched với lora_r  
    lora_dropout: float = 0.0   # Optimized cho Unsloth
    lora_bias: str = "none"     # Optimized cho Unsloth
    
    # Target modules - comprehensive coverage
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Memory optimization
    use_gradient_checkpointing: str = "unsloth"  # Unsloth optimized
    use_rslora: bool = False
    random_state: int = 42


@dataclass  
class DatasetConfig:
    """Cấu hình dataset"""
    dataset_name: str = "ngohongthai/exam-sixth_grade-instruct-dataset"
    train_split: str = "train"
    test_split: str = "test"
    
    # Processing settings
    dataset_text_field: str = "text"
    dataset_num_proc: int = 2
    max_samples: Optional[int] = None  # Limit cho testing
    
    # Chat template
    chat_template: str = "gemma-3n"


@dataclass
class TrainingConfig:
    """Cấu hình training tối ưu cho T4 GPU và 1000 samples"""
    # Output settings
    output_dir: str = "outputs/gemma3n-math-tutor"
    run_name: str = "gemma3n-math-baseline"
    
    # Training schedule - optimized cho 1000 samples
    num_train_epochs: int = 3           # Đủ cho 1000 samples
    max_steps: int = -1                 # Use epochs instead
    eval_strategy: str = "epoch"        # Evaluate sau mỗi epoch
    
    # Batch size settings - optimized cho T4 
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16  # Effective batch = 16
    
    # Learning rate settings
    learning_rate: float = 2e-4         # Standard cho LoRA
    warmup_ratio: float = 0.1           # 10% warmup
    lr_scheduler_type: str = "cosine"
    
    # Regularization
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Optimization
    optim: str = "adamw_8bit"           # Memory efficient
    fp16: bool = True                   # T4 supports FP16
    bf16: bool = False                  # T4 không support BF16
    dataloader_pin_memory: bool = True
    
    # Logging và saving
    logging_steps: int = 10
    save_strategy: str = "epoch"
    save_steps: int = 500
    save_total_limit: int = 3           # Giữ 3 checkpoints
    
    # Evaluation settings
    eval_accumulation_steps: int = 4
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Early stopping
    early_stopping_patience: int = 2
    early_stopping_threshold: float = 0.001
    
    # Reproducibility
    seed: int = 42
    data_seed: int = 42
    
    # Memory optimization
    remove_unused_columns: bool = True
    dataloader_num_workers: int = 2     # Phù hợp với T4
    
    # Report settings
    report_to: str = "comet_ml"
    
    def __post_init__(self):
        """Validate và adjust parameters"""
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Adjust max_steps based on dataset size if needed
        if self.max_steps == -1 and hasattr(self, '_dataset_size'):
            steps_per_epoch = self._dataset_size // (
                self.per_device_train_batch_size * 
                self.gradient_accumulation_steps
            )
            self.max_steps = steps_per_epoch * self.num_train_epochs


@dataclass
class InferenceConfig:
    """Cấu hình cho inference"""
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None


@dataclass
class ExperimentConfig:
    """Cấu hình tổng thể cho experiment"""
    comet: CometConfig = field(default_factory=CometConfig)
    model: ModelConfig = field(default_factory=ModelConfig) 
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    # Experiment metadata
    description: str = "Fine-tuning Gemma3N for Vietnamese 6th grade math tutoring"
    version: str = "1.0.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging"""
        def dataclass_to_dict(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {
                    field_name: dataclass_to_dict(getattr(obj, field_name))
                    for field_name in obj.__dataclass_fields__
                }
            elif isinstance(obj, (list, tuple)):
                return [dataclass_to_dict(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: dataclass_to_dict(v) for k, v in obj.items()}
            else:
                return obj
        
        return dataclass_to_dict(self)
    
    def save_to_file(self, filepath: str):
        """Save configuration to JSON file"""
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'ExperimentConfig':
        """Load configuration from JSON file"""
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Recursive reconstruction of dataclass
        def dict_to_dataclass(data_dict, target_class):
            field_types = {f.name: f.type for f in target_class.__dataclass_fields__.values()}
            kwargs = {}
            
            for field_name, field_type in field_types.items():
                if field_name in data_dict:
                    value = data_dict[field_name]
                    if hasattr(field_type, '__dataclass_fields__'):
                        kwargs[field_name] = dict_to_dataclass(value, field_type)
                    else:
                        kwargs[field_name] = value
            
            return target_class(**kwargs)
        
        return dict_to_dataclass(data, cls)


def get_optimized_config_for_t4() -> ExperimentConfig:
    """
    Trả về cấu hình tối ưu cho GPU T4 với 1000 samples
    """
    config = ExperimentConfig()
    
    # Optimize cho T4 memory constraints
    config.model.max_seq_length = 1536      # Giảm để fit memory
    config.training.per_device_train_batch_size = 1
    config.training.gradient_accumulation_steps = 8  # Effective batch = 8
    config.training.fp16 = True
    config.training.optim = "adamw_8bit"
    
    # Optimize cho 1000 samples
    config.training.num_train_epochs = 2     # Đủ cho small dataset
    config.training.eval_strategy = "steps"
    config.training.eval_steps = 50
    config.training.save_steps = 100
    config.training.logging_steps = 5
    
    # Early stopping để tránh overfitting
    config.training.early_stopping_patience = 3
    config.training.load_best_model_at_end = True
    
    return config


def get_config_for_larger_gpu() -> ExperimentConfig:
    """
    Cấu hình cho GPU lớn hơn (A100, V100, etc.)
    """
    config = ExperimentConfig()
    
    # Có thể dùng settings cao hơn
    config.model.max_seq_length = 2048
    config.training.per_device_train_batch_size = 2
    config.training.gradient_accumulation_steps = 8
    config.training.bf16 = True  # Better cho A100
    config.training.optim = "adamw_torch_fused"
    
    return config


# Helper functions
def validate_environment():
    """Validate required environment variables"""
    required_vars = ["COMET_API_KEY", "COMET_WORKSPACE"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise ValueError(
            f"Missing required environment variables: {missing_vars}. "
            "Please set them in your environment or .env file."
        )


def print_config_summary(config: ExperimentConfig):
    """In ra tóm tắt cấu hình"""
    print("=" * 60)
    print("🔧 GEMMA3N FINE-TUNING CONFIGURATION SUMMARY")
    print("=" * 60)
    
    print(f"📊 Experiment: {config.comet.experiment_name}")
    print(f"🤖 Model: {config.model.model_name}")
    print(f"📚 Dataset: {config.dataset.dataset_name}")
    print(f"💾 Output: {config.training.output_dir}")
    
    print(f"\n⚙️  Training Settings:")
    print(f"   Epochs: {config.training.num_train_epochs}")
    print(f"   Batch Size: {config.training.per_device_train_batch_size}")
    print(f"   Gradient Accumulation: {config.training.gradient_accumulation_steps}")
    print(f"   Effective Batch Size: {config.training.per_device_train_batch_size * config.training.gradient_accumulation_steps}")
    print(f"   Learning Rate: {config.training.learning_rate}")
    print(f"   Max Seq Length: {config.model.max_seq_length}")
    
    print(f"\n🧬 LoRA Settings:")
    print(f"   Rank (r): {config.model.lora_r}")
    print(f"   Alpha: {config.model.lora_alpha}")
    print(f"   Dropout: {config.model.lora_dropout}")
    print(f"   Target Modules: {', '.join(config.model.target_modules)}")
    
    print(f"\n☁️  Comet ML:")
    print(f"   Workspace: {config.comet.workspace}")
    print(f"   Project: {config.comet.project}")
    print(f"   Model Registry: {config.comet.model_registry_name}")
    
    print("=" * 60)
