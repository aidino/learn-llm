#!/bin/bash

# Script chạy evaluation model với deepeval
# Sử dụng: ./run_evaluation.sh [simple|full|visualize|all]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if .env file exists
if [ ! -f ".env" ]; then
    print_error "File .env không tồn tại. Vui lòng tạo file .env với OpenAI API key."
    exit 1
fi

# Check if models exist
if [ ! -d "./gemma_multimodal_finetuned" ]; then
    print_warning "Thư mục model finetuned không tồn tại: ./gemma_multimodal_finetuned"
    print_warning "Vui lòng chạy finetuning trước khi evaluate"
    exit 1
fi

# Function to install dependencies
# install_dependencies() {
#     print_status "Installing dependencies..."
#     pip install -r requirements.txt
#     print_success "Dependencies installed successfully"
# }

# Function to run simple evaluation
run_simple_evaluation() {
    print_status "Running simple evaluation..."
    python simple_evaluation.py
    print_success "Simple evaluation completed"
}

# Function to run simple evaluation without API
run_simple_evaluation_no_api() {
    print_status "Running simple evaluation (no API required)..."
    python simple_evaluation_no_api.py
    print_success "Simple evaluation (no API) completed"
}

# Function to run full evaluation
run_full_evaluation() {
    print_status "Running full evaluation..."
    python model_evaluation.py
    print_success "Full evaluation completed"
}

# Function to run visualization
run_visualization() {
    print_status "Running visualization..."
    python visualize_evaluation.py
    print_success "Visualization completed"
}

# Function to show results
show_results() {
    print_status "Evaluation results:"
    echo ""
    
    if [ -f "./evaluation_results/simple_evaluation.json" ]; then
        echo "Simple evaluation results:"
        echo "=========================="
        python -c "
import json
with open('./evaluation_results/simple_evaluation.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    print(f'Base Model Score: {data[\"base_model\"][\"weighted_score\"]:.3f}')
    print(f'Finetuned Model Score: {data[\"finetuned_model\"][\"weighted_score\"]:.3f}')
    print(f'Improvement: {data[\"comparison\"][\"improvement\"]:.3f} ({data[\"comparison\"][\"improvement_percentage\"]:.1f}%)')
"
        echo ""
    fi
    
    if [ -d "./evaluation_visualizations" ]; then
        echo "Visualization files:"
        echo "==================="
        ls -la ./evaluation_visualizations/
        echo ""
    fi
}

# Function to clean up
cleanup() {
    print_status "Cleaning up..."
    # Remove cache files if needed
    find . -name "*.pyc" -delete
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    print_success "Cleanup completed"
}

# Main script logic
case "${1:-all}" in
    "simple")
        print_status "Running simple evaluation only..."
        # install_dependencies
        run_simple_evaluation
        show_results
        ;;
    "simple-no-api")
        print_status "Running simple evaluation (no API) only..."
        # install_dependencies
        run_simple_evaluation_no_api
        show_results
        ;;
    "full")
        print_status "Running full evaluation only..."
        # install_dependencies
        run_full_evaluation
        show_results
        ;;
    "visualize")
        print_status "Running visualization only..."
        # install_dependencies
        run_visualization
        show_results
        ;;
    "all"|"")
        print_status "Running complete evaluation pipeline..."
        # install_dependencies
        run_simple_evaluation
        run_full_evaluation
        run_visualization
        show_results
        ;;
    "clean")
        cleanup
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [simple|simple-no-api|full|visualize|all|clean|help]"
        echo ""
        echo "Options:"
        echo "  simple        - Run simple evaluation only (requires OpenAI API)"
        echo "  simple-no-api - Run simple evaluation without API"
        echo "  full          - Run full evaluation only"
        echo "  visualize     - Run visualization only"
        echo "  all           - Run complete pipeline (default)"
        echo "  clean         - Clean up cache files"
        echo "  help          - Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0             # Run all evaluations"
        echo "  $0 simple      # Run simple evaluation only"
        echo "  $0 simple-no-api # Run simple evaluation without API"
        echo "  $0 visualize   # Run visualization only"
        ;;
    *)
        print_error "Unknown option: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac

print_success "Evaluation script completed successfully!" 