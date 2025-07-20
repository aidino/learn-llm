#!/bin/bash

# Script chạy tự động toàn bộ quá trình fine-tuning Gemma 3n
# Tối ưu cho RTX 3070 8GB VRAM

set -e  # Exit on any error

echo "🚀 Bắt đầu quá trình Fine-tuning Gemma 3n với QLora"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if we're in the right directory
if [ ! -f "gemma3n_qlora_finetune.py" ]; then
    print_error "Không tìm thấy gemma3n_qlora_finetune.py"
    print_error "Vui lòng chạy script từ thư mục chứa các file cần thiết"
    exit 1
fi

# Step 1: Activate virtual environment
print_status "Kích hoạt virtual environment..."
if [ -f "/home/dino/workdpace/learn-llm/.venv/bin/activate" ]; then
    source /home/dino/workdpace/learn-llm/.venv/bin/activate
    print_success "Virtual environment đã được kích hoạt"
else
    print_warning "Không tìm thấy virtual environment tại /home/dino/workdpace/learn-llm/.venv/"
    print_warning "Tiếp tục với Python global environment"
fi

# Step 2: Check Python and CUDA
print_status "Kiểm tra Python và CUDA..."
python --version
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
    print_success "CUDA và GPU available"
else
    print_error "CUDA hoặc nvidia-smi không available"
    exit 1
fi

# Step 3: Install dependencies if needed
print_status "Kiểm tra và cài đặt dependencies..."
if [ ! -f "requirements.txt" ]; then
    print_error "Không tìm thấy requirements.txt"
    exit 1
fi

# Check if unsloth is installed
if ! python -c "import unsloth" 2>/dev/null; then
    print_status "Cài đặt Unsloth và dependencies..."
    python setup_environment.py
else
    print_success "Dependencies đã được cài đặt"
fi

# Step 4: Check .env file
print_status "Kiểm tra cấu hình Opik..."
if [ ! -f ".env" ]; then
    print_warning "Không tìm thấy file .env"
    print_status "Tạo file .env mẫu..."
    cat > .env << EOF
# Opik Configuration
OPIK_API_KEY=your_opik_api_key_here
OPIK_WORKSPACE=your_workspace_name

# HuggingFace (if needed)
HF_TOKEN=your_huggingface_token

# Other settings
CUDA_VISIBLE_DEVICES=0
EOF
    print_warning "Vui lòng chỉnh sửa file .env với credentials thực tế"
    print_warning "Press Enter để tiếp tục với dummy credentials (chỉ cho testing)..."
    read -r
fi

# Step 5: Clear GPU memory
print_status "Dọn dẹp GPU memory..."
python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('GPU memory cleared')
"

# Step 6: Run the training
print_status "Bắt đầu fine-tuning..."
print_status "Quá trình này có thể mất vài phút đến vài giờ tùy thuộc vào cấu hình"

# Run with error handling
# Check which file to run
if [ -f "gemma_multimodal_qlora_finetuning.py" ]; then
    print_status "Chạy phiên bản multimodal đã sửa lỗi..."
    TRAINING_SCRIPT="gemma_multimodal_qlora_finetuning.py"
else
    print_status "Chạy phiên bản cơ bản..."
    TRAINING_SCRIPT="gemma3n_qlora_finetune.py"
fi

if python "$TRAINING_SCRIPT"; then
    print_success "Fine-tuning hoàn tất thành công!"
    
    # Show results
    echo ""
    echo "📊 Kết quả:"
    if [ -d "gemma3n_qlora_finetuned" ]; then
        print_success "Model đã được save tại: ./gemma3n_qlora_finetuned/"
        ls -la gemma3n_qlora_finetuned/
    fi
    
    if [ -d "gemma3n_qlora_finetuned_merged" ]; then
        print_success "Merged model: ./gemma3n_qlora_finetuned_merged/"
    fi
    
    if [ -d "gemma3n_qlora_finetuned_gguf" ]; then
        print_success "GGUF model: ./gemma3n_qlora_finetuned_gguf/"
    fi
    
    echo ""
    print_success "🎉 HOÀN TẤT!"
    echo "Kiểm tra Opik dashboard để xem experiment tracking"
    echo ""
    echo "Để test model:"
    echo "python -c \"from gemma3n_qlora_finetune import *; trainer = GemmaTrainer(Config()); trainer.load_model_and_tokenizer(); trainer.test_inference()\""
    
else
    print_error "Fine-tuning thất bại!"
    echo ""
    echo "🔍 Debugging tips:"
    echo "1. Kiểm tra VRAM: nvidia-smi"
    echo "2. Giảm batch size trong Config class"
    echo "3. Kiểm tra log errors ở trên"
    echo "4. Chạy: python setup_environment.py để kiểm tra setup"
    exit 1
fi

# Step 7: Cleanup (optional)
read -p "Có muốn dọn dẹp temporary files? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Dọn dẹp temporary files..."
    if [ -d "outputs" ]; then
        rm -rf outputs
        print_success "Đã xóa thư mục outputs"
    fi
    
    python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('✅ GPU memory cleared')
"
fi

echo ""
print_success "Script hoàn tất!"
echo "Happy fine-tuning! 🚀" 