# Fine-tuning Gemma3N với Unsloth

Dự án này cung cấp code hoàn chỉnh để fine-tuning model Gemma3N (một vision-language model) sử dụng thư viện Unsloth.

## 📁 Cấu trúc dự án

```
gemma3n/
├── requirements.md          # Yêu cầu chi tiết từ user
├── gemma3n_finetuning.ipynb # Jupyter notebook chính
├── demo_script.py           # Python script demo nhanh
└── README.md               # Hướng dẫn này
```

## 🚀 Bắt đầu nhanh

### 1. Cài đặt

```bash
# Cài đặt Unsloth
pip install unsloth

# Cài đặt các dependencies cần thiết
pip install torch torchvision torchaudio
pip install transformers datasets pillow
```

### 2. Chạy demo nhanh

```bash
python demo_script.py
```

### 3. Sử dụng Jupyter notebook

```bash
jupyter notebook gemma3n_finetuning.ipynb
```

## 📋 Các tính năng chính

### ✅ Model Loading
- Sử dụng `FastVisionModel` từ Unsloth
- Load model "unsloth/gemma-3n-E4B"  
- 4-bit quantization để tiết kiệm memory
- Gradient checkpointing optimization

### ✅ PEFT Configuration
- Fine-tuning cả vision và language layers
- LoRA với r=32, alpha=32
- Target modules = "all-linear"
- Modules to save: ["lm_head", "embed_tokens"]

### ✅ Dataset Support
- Support instruction với text + nhiều images
- Support answer với text + nhiều images  
- Chat format với system/user/assistant roles
- Sample dataset được tạo sẵn cho demo

### ✅ Training Configuration
- SFTTrainer từ TRL library
- Memory-optimized settings
- AdamW 8-bit optimizer
- Mixed precision training (fp16/bf16)

### ✅ Inference & Deployment  
- Fast inference mode
- Model saving & loading
- Merged model export cho deployment

## 📊 Yêu cầu hệ thống

- **GPU**: Ít nhất 16GB VRAM (khuyến nghị RTX 4090 hoặc A100)
- **RAM**: Ít nhất 32GB
- **Storage**: 50GB+ cho model và checkpoints
- **CUDA**: Version 11.8+ hoặc 12.x

## 🔧 Cấu hình dataset thực tế

Để sử dụng với dataset thực tế, hãy chuẩn bị data theo format:

```python
{
    "messages": [
        {
            "role": "system",
            "content": [{"type": "text", "text": "System prompt"}]
        },
        {
            "role": "user", 
            "content": [
                {"type": "text", "text": "Your instruction"},
                {"type": "image", "image": PIL_Image_object},
                {"type": "image", "image": PIL_Image_object},  # Nhiều images
                {"type": "text", "text": "Additional text"}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Response text"},
                {"type": "image", "image": PIL_Image_object}  # Optional response image
            ]
        }
    ]
}
```

## ⚙️ Tùy chỉnh hyperparameters

### Training Parameters
```python
training_args = TrainingArguments(
    per_device_train_batch_size=1,     # Điều chỉnh theo GPU memory
    gradient_accumulation_steps=4,      # Effective batch size = 4
    max_steps=1000,                    # Tăng cho training thực tế
    learning_rate=2e-5,                # Điều chỉnh theo dataset
    warmup_steps=100,                  # 10% của max_steps
    save_steps=250,                    # Save checkpoint mỗi 250 steps
)
```

### PEFT Parameters
```python
model = FastVisionModel.get_peft_model(
    model,
    r=32,                              # Rank: cao hơn = accuracy tốt hơn, risk overfit
    lora_alpha=32,                     # Khuyến nghị = r
    finetune_vision_layers=True,       # Fine-tune vision encoder
    finetune_language_layers=True,     # Fine-tune language model
)
```

## 🚨 Troubleshooting

### Out of Memory (OOM)
- Giảm `per_device_train_batch_size` xuống 1
- Giảm `max_seq_length` xuống 1024  
- Tăng `gradient_accumulation_steps`
- Sử dụng `load_in_4bit=True`

### Slow Training
- Bật `use_gradient_checkpointing="unsloth"`
- Sử dụng `optim="adamw_8bit"`
- Tắt `packing=False` cho vision model
- Set `dataloader_pin_memory=False`

### Model Loading Issues
- Kiểm tra internet connection
- Sử dụng HuggingFace token nếu cần
- Thử download model trước: `huggingface-cli download unsloth/gemma-3n-E4B`

## 📈 Monitoring Training

### Using TensorBoard
```bash
pip install tensorboard
tensorboard --logdir ./gemma3n_finetuned/logs
```

### Using Weights & Biases
```python
import wandb
wandb.init(project="gemma3n-finetuning")

training_args = TrainingArguments(
    ...
    report_to="wandb",
    run_name="gemma3n-experiment-1"
)
```

## 🎯 Use Cases

Gemma3N fine-tuned model có thể được sử dụng cho:

- **Multimodal Question Answering**: Trả lời câu hỏi về nhiều images
- **Image Captioning**: Tạo mô tả cho images  
- **Visual Instruction Following**: Thực hiện instructions dựa trên images
- **Document Understanding**: Phân tích documents với text + images
- **Educational Content**: Tạo content giáo dục với visual aids

## 📚 Tài liệu tham khảo

- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [Gemma Model Card](https://huggingface.co/google/gemma-3n-e4b)
- [TRL Library](https://github.com/huggingface/trl)
- [PEFT Documentation](https://github.com/huggingface/peft)

## 🤝 Đóng góp

Mọi contributions, issues và feature requests đều được welcome!

## 📄 License

Project này sử dụng MIT License.

---

**Chúc bạn fine-tuning thành công! 🎉**