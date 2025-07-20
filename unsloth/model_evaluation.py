#!/usr/bin/env python3
"""
Script đánh giá model sử dụng deepeval framework
Đánh giá model trước và sau finetuning với các metrics khác nhau
"""

import os
import json
import torch
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Deepeval imports
from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    ContextualRecallMetric,
    TaskCompletionMetric,
    HallucinationMetric,
    BiasMetric,
    ToxicityMetric,
    BaseMetric
)
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset
from deepeval.models import OpenAI, DeepEvalBaseLLM

# Unsloth imports
from unsloth import FastLanguageModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

class Config:
    """Configuration cho evaluation"""
    
    # Model paths
    BASE_MODEL_NAME = "unsloth/gemma-3-4b-pt-unsloth-bnb-4bit"
    FINETUNED_MODEL_PATH = "./gemma_multimodal_finetuned"
    
    # Evaluation dataset
    EVAL_DATASET_PATH = "./evaluation_dataset.json"
    
    # Output paths
    EVAL_RESULTS_DIR = "./evaluation_results"
    BASE_MODEL_RESULTS = "./evaluation_results/base_model_evaluation.json"
    FINETUNED_MODEL_RESULTS = "./evaluation_results/finetuned_model_evaluation.json"
    COMPARISON_RESULTS = "./evaluation_results/model_comparison.json"
    
    # Evaluation settings
    MAX_SAMPLES = 50  # Số lượng samples để evaluate
    TEMPERATURE = 0.1
    MAX_TOKENS = 512
    
    # Metrics weights
    METRICS_WEIGHTS = {
        "answer_relevancy": 0.25,
        "faithfulness": 0.20,
        "answer_correctness": 0.30,
        "hallucination": 0.15,
        "bias": 0.10
    }

class CustomVietnameseMathMetric(BaseMetric):
    """Custom metric cho đánh giá toán học tiếng Việt"""
    
    def __init__(self):
        super().__init__()
        self.name = "Vietnamese Math Accuracy"
        self.description = "Đánh giá độ chính xác của model với bài toán tiếng Việt"
    
    def measure(self, test_case: LLMTestCase) -> float:
        """Đo lường độ chính xác của câu trả lời"""
        if not test_case.actual_output or not test_case.expected_output:
            return 0.0
        
        # Simple keyword matching cho toán học
        math_keywords = ["đúng", "sai", "kết quả", "bằng", "=", "+", "-", "*", "/"]
        actual_lower = test_case.actual_output.lower()
        expected_lower = test_case.expected_output.lower()
        
        # Check if actual output contains expected keywords
        score = 0.0
        for keyword in math_keywords:
            if keyword in expected_lower and keyword in actual_lower:
                score += 1.0
        
        # Normalize score
        return min(score / len(math_keywords), 1.0)

class ModelEvaluator:
    """Class để evaluate model sử dụng deepeval"""
    
    def __init__(self, config: Config):
        self.config = config
        self.base_model = None
        self.finetuned_model = None
        self.eval_dataset = None
        
        # Create output directory
        Path(self.config.EVAL_RESULTS_DIR).mkdir(exist_ok=True)
        
        # Setup metrics
        self.metrics = [
            AnswerRelevancyMetric(threshold=0.7),
            FaithfulnessMetric(threshold=0.7),
            TaskCompletionMetric(threshold=0.7),
            HallucinationMetric(threshold=0.7),
            BiasMetric(threshold=0.7),
            CustomVietnameseMathMetric()
        ]
    
    def load_base_model(self):
        """Load base model"""
        logger.info("Loading base model...")
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.config.BASE_MODEL_NAME,
                max_seq_length=512,
                dtype=None,
                load_in_4bit=True,
            )
            self.base_model = model
            self.base_tokenizer = tokenizer
            logger.info("Base model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading base model: {e}")
            raise
    
    def load_finetuned_model(self):
        """Load finetuned model"""
        logger.info("Loading finetuned model...")
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.config.FINETUNED_MODEL_PATH,
                max_seq_length=512,
                dtype=None,
                load_in_4bit=True,
            )
            self.finetuned_model = model
            self.finetuned_tokenizer = tokenizer
            logger.info("Finetuned model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading finetuned model: {e}")
            raise
    
    def create_evaluation_dataset(self):
        """Tạo dataset evaluation từ dataset gốc"""
        logger.info("Creating evaluation dataset...")
        
        # Sample questions cho evaluation
        eval_questions = [
            {
                "question": "Tính tổng của 15 và 27",
                "expected_answer": "15 + 27 = 42",
                "context": "Bài toán cộng hai số tự nhiên"
            },
            {
                "question": "Một hình chữ nhật có chiều dài 8cm và chiều rộng 5cm. Tính diện tích.",
                "expected_answer": "Diện tích = 8 × 5 = 40 cm²",
                "context": "Tính diện tích hình chữ nhật"
            },
            {
                "question": "Tìm x biết: 3x + 7 = 22",
                "expected_answer": "3x = 22 - 7 = 15, x = 15 ÷ 3 = 5",
                "context": "Giải phương trình bậc nhất"
            },
            {
                "question": "Một lớp có 30 học sinh, trong đó có 18 học sinh nam. Tính tỷ lệ học sinh nữ.",
                "expected_answer": "Học sinh nữ = 30 - 18 = 12, Tỷ lệ = 12/30 = 40%",
                "context": "Tính tỷ lệ phần trăm"
            },
            {
                "question": "Tính chu vi hình tròn có bán kính 5cm",
                "expected_answer": "Chu vi = 2 × π × 5 = 10π ≈ 31.4 cm",
                "context": "Tính chu vi hình tròn"
            }
        ]
        
        # Tạo test cases
        test_cases = []
        for i, q in enumerate(eval_questions[:self.config.MAX_SAMPLES]):
            test_case = LLMTestCase(
                input=q["question"],
                actual_output="",  # Sẽ được fill sau
                expected_output=q["expected_answer"],
                context=q["context"]
            )
            test_cases.append(test_case)
        
        self.eval_dataset = EvaluationDataset(test_cases=test_cases)
        logger.info(f"Created evaluation dataset with {len(test_cases)} test cases")
    
    def generate_response(self, model, tokenizer, prompt: str) -> str:
        """Generate response từ model"""
        try:
            # Format prompt
            formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            # Tokenize
            inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=512)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.config.MAX_TOKENS,
                    temperature=self.config.TEMPERATURE,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract assistant response
            if "<|im_start|>assistant" in response:
                response = response.split("<|im_start|>assistant")[-1].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Lỗi khi tạo câu trả lời"
    
    def evaluate_model(self, model, tokenizer, model_name: str) -> Dict[str, Any]:
        """Evaluate một model cụ thể"""
        logger.info(f"Evaluating {model_name}...")
        
        results = {
            "model_name": model_name,
            "evaluation_time": datetime.now().isoformat(),
            "metrics": {},
            "test_cases": []
        }
        
        # Evaluate từng test case
        for i, test_case in enumerate(self.eval_dataset.test_cases):
            logger.info(f"Evaluating test case {i+1}/{len(self.eval_dataset.test_cases)}")
            
            # Generate response
            actual_output = self.generate_response(model, tokenizer, test_case.input)
            test_case.actual_output = actual_output
            
            # Calculate metrics
            case_metrics = {}
            for metric in self.metrics:
                try:
                    score = metric.measure(test_case)
                    case_metrics[metric.name] = score
                except Exception as e:
                    logger.warning(f"Error calculating {metric.name}: {e}")
                    case_metrics[metric.name] = 0.0
            
            # Store test case results
            case_result = {
                "input": test_case.input,
                "expected_output": test_case.expected_output,
                "actual_output": actual_output,
                "metrics": case_metrics
            }
            results["test_cases"].append(case_result)
        
        # Calculate overall metrics
        overall_metrics = {}
        for metric in self.metrics:
            scores = [case["metrics"].get(metric.name, 0.0) for case in results["test_cases"]]
            overall_metrics[metric.name] = {
                "mean": sum(scores) / len(scores) if scores else 0.0,
                "min": min(scores) if scores else 0.0,
                "max": max(scores) if scores else 0.0
            }
        
        results["metrics"] = overall_metrics
        
        # Calculate weighted score
        weighted_score = 0.0
        for metric_name, weight in self.config.METRICS_WEIGHTS.items():
            if metric_name in overall_metrics:
                weighted_score += overall_metrics[metric_name]["mean"] * weight
        
        results["weighted_score"] = weighted_score
        
        logger.info(f"Evaluation completed for {model_name}. Weighted score: {weighted_score:.3f}")
        return results
    
    def save_results(self, results: Dict[str, Any], filepath: str):
        """Lưu kết quả evaluation"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Results saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def compare_models(self, base_results: Dict[str, Any], finetuned_results: Dict[str, Any]) -> Dict[str, Any]:
        """So sánh kết quả của hai model"""
        logger.info("Comparing model results...")
        
        comparison = {
            "comparison_time": datetime.now().isoformat(),
            "base_model": base_results["model_name"],
            "finetuned_model": finetuned_results["model_name"],
            "improvements": {},
            "overall_comparison": {}
        }
        
        # Compare metrics
        for metric_name in base_results["metrics"]:
            if metric_name in finetuned_results["metrics"]:
                base_score = base_results["metrics"][metric_name]["mean"]
                finetuned_score = finetuned_results["metrics"][metric_name]["mean"]
                improvement = finetuned_score - base_score
                
                comparison["improvements"][metric_name] = {
                    "base_score": base_score,
                    "finetuned_score": finetuned_score,
                    "improvement": improvement,
                    "improvement_percentage": (improvement / base_score * 100) if base_score > 0 else 0
                }
        
        # Overall comparison
        base_weighted = base_results.get("weighted_score", 0.0)
        finetuned_weighted = finetuned_results.get("weighted_score", 0.0)
        overall_improvement = finetuned_weighted - base_weighted
        
        comparison["overall_comparison"] = {
            "base_weighted_score": base_weighted,
            "finetuned_weighted_score": finetuned_weighted,
            "overall_improvement": overall_improvement,
            "improvement_percentage": (overall_improvement / base_weighted * 100) if base_weighted > 0 else 0
        }
        
        logger.info(f"Model comparison completed. Overall improvement: {overall_improvement:.3f}")
        return comparison
    
    def run_full_evaluation(self):
        """Chạy toàn bộ quá trình evaluation"""
        logger.info("Starting full model evaluation...")
        
        try:
            # Load models
            self.load_base_model()
            self.load_finetuned_model()
            
            # Create evaluation dataset
            self.create_evaluation_dataset()
            
            # Evaluate base model
            logger.info("Evaluating base model...")
            base_results = self.evaluate_model(self.base_model, self.base_tokenizer, "Base Model")
            self.save_results(base_results, self.config.BASE_MODEL_RESULTS)
            
            # Evaluate finetuned model
            logger.info("Evaluating finetuned model...")
            finetuned_results = self.evaluate_model(self.finetuned_model, self.finetuned_tokenizer, "Finetuned Model")
            self.save_results(finetuned_results, self.config.FINETUNED_MODEL_RESULTS)
            
            # Compare models
            comparison_results = self.compare_models(base_results, finetuned_results)
            self.save_results(comparison_results, self.config.COMPARISON_RESULTS)
            
            # Print summary
            self.print_evaluation_summary(base_results, finetuned_results, comparison_results)
            
            logger.info("Full evaluation completed successfully!")
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            raise

    def print_evaluation_summary(self, base_results: Dict[str, Any], finetuned_results: Dict[str, Any], comparison_results: Dict[str, Any]):
        """In kết quả tóm tắt evaluation"""
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        print(f"\nBase Model: {base_results['model_name']}")
        print(f"Weighted Score: {base_results.get('weighted_score', 0.0):.3f}")
        
        print(f"\nFinetuned Model: {finetuned_results['model_name']}")
        print(f"Weighted Score: {finetuned_results.get('weighted_score', 0.0):.3f}")
        
        print(f"\nOverall Improvement:")
        overall = comparison_results["overall_comparison"]
        print(f"  Score Improvement: {overall['overall_improvement']:.3f}")
        print(f"  Percentage Improvement: {overall['improvement_percentage']:.1f}%")
        
        print(f"\nMetric Improvements:")
        for metric, data in comparison_results["improvements"].items():
            print(f"  {metric}: {data['improvement']:.3f} ({data['improvement_percentage']:.1f}%)")
        
        print("="*60)

def main():
    """Main function"""
    config = Config()
    evaluator = ModelEvaluator(config)
    evaluator.run_full_evaluation()

if __name__ == "__main__":
    main() 