# Hướng dẫn sử dụng Deepeval Framework để Evaluate Model

## Tổng quan

Framework này sử dụng **deepeval** để đánh giá model trước và sau finetuning với các metrics khác nhau như Answer Relevancy, Faithfulness, Answer Correctness, và custom metrics cho toán học tiếng Việt.

## Cài đặt

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Cấu hình OpenAI API

Tạo file `.env` với OpenAI API key:

```bash
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```

## Cấu trúc Files

```
├── model_evaluation.py          # Script evaluation đầy đủ
├── simple_evaluation.py         # Script evaluation đơn giản
├── visualize_evaluation.py      # Script tạo biểu đồ
├── run_evaluation.sh           # Script bash để chạy evaluation
├── evaluation_results/         # Thư mục chứa kết quả
├── evaluation_visualizations/  # Thư mục chứa biểu đồ
└── EVALUATION_README.md        # File hướng dẫn này
```

## Sử dụng

### 1. Chạy evaluation đơn giản (Khuyến nghị)

```bash
./run_evaluation.sh simple
```

Hoặc:

```bash
python simple_evaluation.py
```

### 2. Chạy evaluation đầy đủ

```bash
./run_evaluation.sh full
```

Hoặc:

```bash
python model_evaluation.py
```

### 3. Tạo biểu đồ visualization

```bash
./run_evaluation.sh visualize
```

Hoặc:

```bash
python visualize_evaluation.py
```

### 4. Chạy toàn bộ pipeline

```bash
./run_evaluation.sh all
```

### 5. Xem help

```bash
./run_evaluation.sh help
```

## Metrics được sử dụng

### 1. Answer Relevancy
- **Mô tả**: Đánh giá mức độ liên quan của câu trả lời với câu hỏi
- **Threshold**: 0.7
- **Weight**: 0.4

### 2. Faithfulness
- **Mô tả**: Đánh giá tính trung thực của câu trả lời
- **Threshold**: 0.7
- **Weight**: 0.3

### 3. Answer Correctness
- **Mô tả**: Đánh giá độ chính xác của câu trả lời
- **Threshold**: 0.7
- **Weight**: 0.3

### 4. Custom Vietnamese Math Metric
- **Mô tả**: Metric tùy chỉnh cho đánh giá toán học tiếng Việt
- **Cách tính**: Kiểm tra từ khóa toán học trong câu trả lời

## Test Questions

Script sử dụng các câu hỏi test sau:

1. **Tính 15 + 27 = ?**
   - Expected: "15 + 27 = 42"

2. **Một hình chữ nhật có chiều dài 8cm, chiều rộng 5cm. Tính diện tích.**
   - Expected: "Diện tích = 8 × 5 = 40 cm²"

3. **Tìm x biết: 3x + 7 = 22**
   - Expected: "3x = 22 - 7 = 15, x = 15 ÷ 3 = 5"

4. **Tính chu vi hình tròn có bán kính 5cm**
   - Expected: "Chu vi = 2 × π × 5 = 10π ≈ 31.4 cm"

## Kết quả

### 1. JSON Results

Kết quả được lưu trong thư mục `evaluation_results/`:

- `simple_evaluation.json`: Kết quả evaluation đơn giản
- `base_model_evaluation.json`: Kết quả model gốc (evaluation đầy đủ)
- `finetuned_model_evaluation.json`: Kết quả model finetuned (evaluation đầy đủ)
- `model_comparison.json`: So sánh hai model

### 2. Visualizations

Biểu đồ được tạo trong thư mục `evaluation_visualizations/`:

- `metrics_comparison.png`: So sánh metrics giữa hai model
- `weighted_score_comparison.png`: So sánh weighted score
- `relevancy_by_question.png`: Relevancy score theo từng câu hỏi
- `faithfulness_by_question.png`: Faithfulness score theo từng câu hỏi
- `correctness_by_question.png`: Correctness score theo từng câu hỏi
- `radar_chart.png`: Biểu đồ radar so sánh metrics
- `evaluation_report.md`: Báo cáo tóm tắt

## Cấu hình

### Thay đổi test questions

Chỉnh sửa trong `simple_evaluation.py`:

```python
self.test_questions = [
    "Câu hỏi mới của bạn",
    # Thêm câu hỏi khác...
]

self.expected_answers = [
    "Câu trả lời mong đợi",
    # Thêm câu trả lời tương ứng...
]
```

### Thay đổi metrics weights

Chỉnh sửa trong `simple_evaluation.py`:

```python
weighted_score = (
    results["metrics"]["answer_relevancy"]["mean"] * 0.4 +  # Thay đổi weight
    results["metrics"]["faithfulness"]["mean"] * 0.3 +      # Thay đổi weight
    results["metrics"]["answer_correctness"]["mean"] * 0.3  # Thay đổi weight
)
```

### Thay đổi model paths

Chỉnh sửa trong `Config` class:

```python
class Config:
    BASE_MODEL_NAME = "your_base_model_path"
    FINETUNED_MODEL_PATH = "./your_finetuned_model_path"
```

## Troubleshooting

### 1. Lỗi "Model not found"

```bash
# Kiểm tra xem model đã được finetune chưa
ls -la ./gemma_multimodal_finetuned/
```

### 2. Lỗi "OpenAI API key not found"

```bash
# Kiểm tra file .env
cat .env
```

### 3. Lỗi memory khi load model

Giảm `MAX_SAMPLES` trong config hoặc sử dụng model nhỏ hơn.

### 4. Lỗi deepeval metrics

```bash
# Reinstall deepeval
pip uninstall deepeval
pip install deepeval>=0.20.0
```

## Ví dụ Output

### Console Output

```
[INFO] Loading models...
[INFO] Base model loaded
[INFO] Finetuned model loaded
[INFO] Starting simple evaluation...
[INFO] Evaluating Base Model...
[INFO] Question 1: Tính 15 + 27 = ?
[INFO] Response: 15 + 27 = 42...
[INFO] Base Model evaluation completed. Weighted score: 0.750
[INFO] Evaluating Finetuned Model...
[INFO] Question 1: Tính 15 + 27 = ?
[INFO] Response: 15 + 27 = 42...
[INFO] Finetuned Model evaluation completed. Weighted score: 0.850

================================================================================
MODEL COMPARISON RESULTS
================================================================================

Base Model: Base Model
Weighted Score: 0.750

Finetuned Model: Finetuned Model
Weighted Score: 0.850

Overall Improvement: 0.100

Detailed Metrics Comparison:
  answer_relevancy: 0.800 → 0.900 (+0.100)
  faithfulness: 0.700 → 0.800 (+0.100)
  answer_correctness: 0.750 → 0.850 (+0.100)
```

### JSON Output Example

```json
{
  "evaluation_time": "2024-01-15T10:30:00",
  "base_model": {
    "model_name": "Base Model",
    "weighted_score": 0.750,
    "metrics": {
      "answer_relevancy": {"mean": 0.800, "scores": [0.8, 0.8, 0.8, 0.8]},
      "faithfulness": {"mean": 0.700, "scores": [0.7, 0.7, 0.7, 0.7]},
      "answer_correctness": {"mean": 0.750, "scores": [0.75, 0.75, 0.75, 0.75]}
    }
  },
  "finetuned_model": {
    "model_name": "Finetuned Model",
    "weighted_score": 0.850,
    "metrics": {
      "answer_relevancy": {"mean": 0.900, "scores": [0.9, 0.9, 0.9, 0.9]},
      "faithfulness": {"mean": 0.800, "scores": [0.8, 0.8, 0.8, 0.8]},
      "answer_correctness": {"mean": 0.850, "scores": [0.85, 0.85, 0.85, 0.85]}
    }
  },
  "comparison": {
    "improvement": 0.100,
    "improvement_percentage": 13.3
  }
}
```

## Lưu ý

1. **Memory Usage**: Evaluation có thể tốn nhiều RAM khi load cả hai model. Đảm bảo có đủ memory.

2. **API Costs**: Deepeval sử dụng OpenAI API cho một số metrics, có thể phát sinh chi phí.

3. **Model Compatibility**: Đảm bảo model được finetune với cùng format prompt như trong evaluation.

4. **Customization**: Có thể dễ dàng tùy chỉnh metrics, test questions, và weights theo nhu cầu.

## Hỗ trợ

Nếu gặp vấn đề, hãy kiểm tra:

1. Logs trong console
2. File kết quả JSON
3. Cấu hình model paths
4. OpenAI API key
5. Memory usage

Để debug chi tiết hơn, có thể thêm logging level DEBUG:

```python
logging.basicConfig(level=logging.DEBUG)
``` 