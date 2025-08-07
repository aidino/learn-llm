# Gemma3N Fine-tuning for Vietnamese 6th Grade Math Tutoring

Dự án fine-tuning mô hình Gemma3N để tạo ra một gia sư toán học tiếng Việt cho học sinh lớp 6. Sử dụng Unsloth để tối ưu hóa tốc độ training và Comet ML để theo dõi experiments.

## ✨ Tính năng chính

- 🚀 **Tối ưu hóa cho GPU T4**: Cấu hình đặc biệt cho GPU T4 với 1000 samples
- 🧠 **Unsloth Integration**: Tăng tốc 2x so với training thông thường
- 📊 **Comet ML Tracking**: Theo dõi đầy đủ experiments và model registry
- 🔧 **Highly Configurable**: Dễ dàng thay đổi hyperparameters
- 📚 **LoRA Fine-tuning**: Memory-efficient với Parameter-Efficient Fine-Tuning
- 💾 **Multiple Save Formats**: Hỗ trợ LoRA, merged 16bit/4bit, GGUF
- ☁️ **HuggingFace Hub**: Tự động push models lên Hub

## 🛠️ Cài đặt

### 1. Clone repository
```bash
git clone <your-repo-url>
cd mathpal
```

### 2. Cài đặt dependencies
```bash
# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc venv\\Scripts\\activate  # Windows

# Cài đặt packages
pip install -r requirements.txt
```

### 3. Setup Comet ML
```bash
# Set environment variables
export COMET_API_KEY="your-api-key"
export COMET_WORKSPACE="your-workspace" 
export COMET_PROJECT="mathpal-gemma3n"

# Hoặc tạo file .env
echo "COMET_API_KEY=your-api-key" > .env
echo "COMET_WORKSPACE=your-workspace" >> .env
echo "COMET_PROJECT=mathpal-gemma3n" >> .env
```

### 4. Setup HuggingFace (optional)
```bash
export HF_TOKEN="your-hf-token"
```

## 🚀 Sử dụng

### Training cơ bản
```bash
python train_gemma3n.py
```

### Training với custom config
```bash
python train_gemma3n.py --config custom_config.json
```

### Test nhanh với ít samples
```bash
python train_gemma3n.py --test-run --max-samples 50
```

### Training cho GPU lớn hơn
```bash
python train_gemma3n.py --gpu-type a100
```

### Push model lên HuggingFace Hub
```bash
python train_gemma3n.py --push-to-hub your-username/gemma3n-math-tutor
```

### Resume từ checkpoint
```bash
python train_gemma3n.py --resume-from-checkpoint outputs/checkpoint-100
```

## ⚙️ Cấu hình

### Cấu hình mặc định cho T4 GPU

Được tối ưu hóa cho:
- GPU T4 (16GB VRAM)
- Dataset ~1000 samples  
- Training time ~2-3 giờ

```python
# Batch settings
per_device_train_batch_size = 1
gradient_accumulation_steps = 8  # Effective batch = 8

# Model settings
max_seq_length = 1536
load_in_4bit = True

# LoRA settings
lora_r = 16
lora_alpha = 16
lora_dropout = 0.0

# Training settings
num_train_epochs = 2
learning_rate = 2e-4
warmup_ratio = 0.1
```

### Customization

Tạo file config JSON:
```json
{
  "model": {
    "model_name": "unsloth/gemma-3n-E4B-it",
    "max_seq_length": 2048,
    "lora_r": 32
  },
  "training": {
    "num_train_epochs": 3,
    "learning_rate": 1e-4,
    "per_device_train_batch_size": 2
  },
  "comet": {
    "experiment_name": "my-experiment",
    "tags": ["custom", "experiment"]
  }
}
```

## 📊 Dataset

Sử dụng dataset `ngohongthai/exam-sixth_grade-instruct-dataset` với format:

```json
{
  "question": "Tính 15 + 23 = ?",
  "solution": "15 + 23 = 38"
}
```

Dataset được convert tự động sang định dạng Gemma3N conversation:
```json
{
  "conversations": [
    {
      "role": "user", 
      "content": [{"type": "text", "text": "Tính 15 + 23 = ?"}]
    },
    {
      "role": "assistant",
      "content": [{"type": "text", "text": "15 + 23 = 38"}]
    }
  ]
}
```

## 📈 Monitoring với Comet ML

### Experiment Tracking
- Training/validation loss
- Learning rate schedule  
- GPU memory usage
- Hyperparameters
- Model artifacts

### Model Registry
- Tự động log models đã train
- Version management
- Metadata và metrics
- Easy deployment

### Truy cập experiments
```python
import comet_ml

# Get experiment
api = comet_ml.API()
experiment = api.get_experiment(
    workspace="your-workspace",
    project_name="mathpal-gemma3n", 
    experiment_id="experiment-id"
)

# Download model
experiment.download_model("model-name", "./downloaded_model")
```

## 💾 Model Formats

Script hỗ trợ nhiều định dạng save:

### LoRA Adapters (mặc định)
```bash
python train_gemma3n.py --save-formats lora
```
- Chỉ save LoRA weights (~100MB)
- Nhanh và tiết kiệm storage
- Requires base model để inference

### Merged 16-bit
```bash
python train_gemma3n.py --save-formats merged_16bit
```
- Full model precision cao (~14GB)
- Ready cho production
- Tương thích với VLLM

### Merged 4-bit  
```bash
python train_gemma3n.py --save-formats merged_4bit
```
- Quantized model (~4GB)
- Cân bằng size/quality
- Phù hợp cho deployment

### GGUF Format
```bash
python train_gemma3n.py --save-formats gguf_q8_0 gguf_q4_k_m
```
- Cho llama.cpp inference
- Rất nhỏ gọn
- CPU inference friendly

## 🧪 Testing và Inference

### Chạy inference test
```bash
python train_gemma3n.py --inference-only
```

### Test với model đã train
```python
from model_manager import create_model_manager
from config import ModelConfig, InferenceConfig

# Load model
config = ModelConfig()
manager = create_model_manager(config)
model, tokenizer = manager.load_base_model()
model = manager.apply_lora()

# Load checkpoint nếu có
# manager.load_adapter("path/to/checkpoint")

# Generate response
response = manager.generate_response(
    question="Tính 5 × 7 = ?",
    inference_config=InferenceConfig()
)
print(response)
```

## 🔧 Advanced Usage

### Custom LoRA Configuration
```python
from config import ModelConfig

config = ModelConfig()
config.lora_r = 32              # Higher rank for better quality
config.lora_alpha = 32          # Match với lora_r
config.target_modules = [       # More modules for comprehensive training
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "lm_head", "embed_tokens"   # Include output layers
]
```

### Memory Optimization
```python
from config import TrainingConfig

config = TrainingConfig()
config.gradient_accumulation_steps = 16    # Larger effective batch
config.per_device_train_batch_size = 1     # Smaller per-device batch
config.fp16 = True                          # Half precision
config.dataloader_num_workers = 0          # Reduce memory usage
```

### Early Stopping
```python
config.early_stopping_patience = 3
config.load_best_model_at_end = True
config.metric_for_best_model = "eval_loss"
config.eval_strategy = "steps"
config.eval_steps = 50
```

## 📁 Project Structure

```
mathpal/
├── train_gemma3n.py          # Main training script
├── config.py                 # Configuration classes
├── model_manager.py          # Model operations
├── data_processor.py         # Dataset processing  
├── trainer_wrapper.py        # Training with Comet ML
├── requirements.txt          # Dependencies
├── README.md                 # This file
└── outputs/                  # Training outputs
    ├── model_lora/           # LoRA adapters
    ├── model_merged_16bit/   # Merged models
    ├── checkpoints/          # Training checkpoints
    └── logs/                 # Training logs
```

## 🚨 Troubleshooting

### GPU Memory Issues
```bash
# Reduce batch size
python train_gemma3n.py --batch-size 1 --max-seq-length 1024

# Use 4-bit quantization
# (Đã enable mặc định trong T4 config)
```

### Comet ML Connection Issues  
```bash
# Check API key
echo $COMET_API_KEY

# Test connection
python -c "import comet_ml; print('Comet ML OK')"
```

### Unsloth Installation Issues
```bash
# Auto-detect và install phiên bản phù hợp
wget -qO- https://raw.githubusercontent.com/unslothai/unsloth/main/unsloth/_auto_install.py | python -

# Hoặc manual cho CUDA 12.1 + PyTorch 2.4.0
pip install "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git"
```

### Dataset Loading Issues
```bash
# Test dataset loading
python -c "
from data_processor import test_data_processor
test_data_processor()
"
```

## 📚 Tài liệu tham khảo

- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [Comet ML Documentation](https://www.comet.com/docs/)
- [Gemma Model Documentation](https://ai.google.dev/gemma)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)

## 🤝 Contributing

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Tạo Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Unsloth Team** - Cho framework tối ưu hóa training
- **Comet ML** - Cho platform experiment tracking
- **Google** - Cho Gemma models
- **HuggingFace** - Cho transformers library và model hub
- **Dataset Contributors** - Cho Vietnamese math dataset

---

**Happy Fine-tuning! 🚀**
