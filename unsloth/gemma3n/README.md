# Gemma3N Math Tutor Fine-tuning

Tự động fine-tune model Gemma3N để giải các bài toán Toán lớp 6 với hỗ trợ đa phương thức (text + image).

## 🎯 Mục tiêu

Fine-tune model [unsloth/gemma-3n-E4B](https://huggingface.co/unsloth/gemma-3n-E4B) để:
- Giải các bài toán Toán lớp 6
- Xử lý cả text và hình ảnh minh họa
- Tạo ra các lời giải chi tiết và chính xác

## 📊 Dataset

Sử dụng dataset: [ngohongthai/exam-sixth_grade-instruct-dataset](https://huggingface.co/datasets/ngohongthai/exam-sixth_grade-instruct-dataset)

**Đặc điểm dataset:**
- 1,123 mẫu dữ liệu (1,010 train + 113 test)
- 2 cột: `question` và `solution`
- Chứa hình ảnh minh họa trong format markdown
- Bao gồm các dạng toán: hình học, số học, ứng dụng

## 🛠️ Cài đặt

1. **Clone repository:**
```bash
git clone <repository-url>
cd gemma3n-math-tutor
```

2. **Tạo virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate  # Windows
```

3. **Cài đặt dependencies:**
```bash
pip install -r requirements.txt
```

## 🚀 Cách sử dụng

### Training cơ bản
```bash
python gemma3n_math_finetuning.py
```

### Tùy chỉnh cấu hình
Chỉnh sửa `CONFIG` trong file `gemma3n_math_finetuning.py`:

```python
CONFIG = {
    # Model settings
    "model_name": "unsloth/gemma-3n-E4B",
    "max_seq_length": 2048,
    
    # Training settings
    "max_steps": 200,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    
    # Output
    "output_dir": "outputs/gemma3n-math-tutor",
    "report_to": "tensorboard",  # hoặc "wandb", "comet_ml"
}
```

## 🏗️ Kiến trúc giải pháp

### 1. Xử lý dữ liệu đa phương thức
- **Image URL extraction**: Tự động tìm và tải hình ảnh từ markdown
- **Conversation format**: Chuyển đổi sang format chat chuẩn
- **Mixed data handling**: Xử lý cả samples có và không có hình ảnh

### 2. Tối ưu hóa training
- **Custom Data Collator**: Xử lý batch mixed text/image
- **Memory optimization**: Tắt gradient checkpointing để tránh lỗi
- **LoRA fine-tuning**: Hiệu quả và tiết kiệm memory

### 3. Robustness
- **Error handling**: Graceful fallback cho images bị lỗi
- **Placeholder images**: Đảm bảo consistency cho text-only samples
- **Progress tracking**: Detailed logging và monitoring

## 📁 Cấu trúc file

```
.
├── gemma3n_math_finetuning.py  # Script chính
├── comet_config.py             # Cấu hình Comet ML
├── requirements.txt            # Dependencies
├── README.md                  # Hướng dẫn
└── outputs/                   # Thư mục output models
    └── gemma3n-math-tutor/    # Model đã train
```

## ⚡ Optimizations

### Giải quyết các vấn đề thường gặp:

1. **CheckpointError**: Tắt gradient checkpointing
2. **Image token mismatch**: Custom collator với auto image token insertion
3. **Mixed data types**: Unified processing pipeline
4. **Memory efficiency**: 4-bit quantization + LoRA

### Performance metrics:
- **Memory usage**: ~14GB VRAM (Tesla T4)
- **Training speed**: 2x faster với Unsloth
- **Model size**: ~97% parameters frozen (LoRA)

## 🧪 Testing và Inference

```python
# Test inference sau khi training
from gemma3n_math_finetuning import test_inference

test_inference("outputs/gemma3n-math-tutor")
```

## 📈 Monitoring

### TensorBoard
```bash
tensorboard --logdir outputs/gemma3n-math-tutor/logs
```

### Comet ML (recommended)
1. **Tạo tài khoản**: Đăng ký tại [comet.com](https://www.comet.com/)
2. **Tạo workspace và project** trên Comet ML dashboard
3. **Lấy API key**: Từ [Settings](https://www.comet.com/api/my/settings/)
4. **Cấu hình**:
```bash
# Cách 1: Environment variable (khuyến nghị)
export COMET_API_KEY="your-api-key-here"

# Cách 2: Chỉnh sửa comet_config.py
```
5. **Cập nhật workspace và project** trong `comet_config.py`:
```python
COMET_CONFIG = {
    "workspace": "your-workspace-name",  # Thay bằng workspace của bạn
    "project": "gemma3n-math-tutor",     # Tên project
}
```

### Weights & Biases (alternative)
1. Uncomment `wandb` trong requirements.txt
2. Set `"report_to": "wandb"` trong CONFIG
3. Run: `wandb login`

## 🔧 Troubleshooting

### Lỗi thường gặp:

1. **CUDA out of memory**:
   - Giảm `per_device_train_batch_size`
   - Tăng `gradient_accumulation_steps`

2. **Image loading failed**:
   - Check internet connection
   - Một số URLs có thể bị expired

3. **Model convergence issues**:
   - Adjust learning rate
   - Increase max_steps

### Debug mode:
```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📊 Expected Results

Sau khi training, model sẽ có khả năng:
- ✅ Hiểu và phân tích đề bài toán có hình ảnh
- ✅ Tạo lời giải step-by-step chi tiết
- ✅ Xử lý các dạng toán: hình học, đại số, ứng dụng
- ✅ Format output mathematical theo chuẩn LaTeX

## 🤝 Contributing

1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Tạo Pull Request

## 📜 License

MIT License - xem file LICENSE để biết thêm chi tiết.

## 🙏 Acknowledgments

- [Unsloth AI](https://github.com/unslothai/unsloth) - Fast LLM training
- [Hugging Face](https://huggingface.co/) - Transformers và Datasets
- Dataset creator: [ngohongthai](https://huggingface.co/ngohongthai)