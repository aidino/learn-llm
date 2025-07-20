#!/usr/bin/env python3
"""
Script evaluation đơn giản không cần OpenAI API
Sử dụng custom metrics và simple scoring
"""

import os
import json
import torch
import warnings
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import logging

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import Unsloth FIRST để tránh warning
from unsloth import FastLanguageModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

class SimpleMathEvaluator:
    """Simple evaluator không cần OpenAI API"""
    
    def __init__(self):
        self.base_model = None
        self.base_tokenizer = None
        self.finetuned_model = None
        self.finetuned_tokenizer = None
        
        # Test questions
        self.test_questions = [
            "Tính 15 + 27 = ?",
            "Một hình chữ nhật có chiều dài 8cm, chiều rộng 5cm. Tính diện tích.",
            "Tìm x biết: 3x + 7 = 22",
            "Tính chu vi hình tròn có bán kính 5cm"
        ]
        
        self.expected_answers = [
            "15 + 27 = 42",
            "Diện tích = 8 × 5 = 40 cm²",
            "3x = 22 - 7 = 15, x = 15 ÷ 3 = 5",
            "Chu vi = 2 × π × 5 = 10π ≈ 31.4 cm"
        ]
    
    def load_models(self):
        """Load cả base và finetuned model"""
        logger.info("Loading models...")
        
        try:
            # Load base model
            self.base_model, self.base_tokenizer = FastLanguageModel.from_pretrained(
                model_name="unsloth/gemma-3-4b-pt-unsloth-bnb-4bit",
                max_seq_length=512,
                dtype=None,
                load_in_4bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )
            logger.info("Base model loaded")
            
            # Load finetuned model
            self.finetuned_model, self.finetuned_tokenizer = FastLanguageModel.from_pretrained(
                model_name="./gemma_multimodal_finetuned",
                max_seq_length=512,
                dtype=None,
                load_in_4bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )
            logger.info("Finetuned model loaded")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def generate_response(self, model, tokenizer, question: str) -> str:
        """Generate response từ model"""
        try:
            prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if "<|im_start|>assistant" in response:
                response = response.split("<|im_start|>assistant")[-1].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Lỗi khi tạo câu trả lời"
    
    def calculate_simple_metrics(self, actual: str, expected: str) -> Dict[str, float]:
        """Tính toán metrics đơn giản không cần API"""
        
        # Convert to lowercase for comparison
        actual_lower = actual.lower()
        expected_lower = expected.lower()
        
        # 1. Keyword matching score
        math_keywords = ["đúng", "sai", "kết quả", "bằng", "=", "+", "-", "*", "/", "×", "÷", "π"]
        keyword_matches = 0
        for keyword in math_keywords:
            if keyword in expected_lower and keyword in actual_lower:
                keyword_matches += 1
        
        keyword_score = min(keyword_matches / len(math_keywords), 1.0)
        
        # 2. Number matching score
        import re
        expected_numbers = set(re.findall(r'\d+', expected_lower))
        actual_numbers = set(re.findall(r'\d+', actual_lower))
        
        if expected_numbers:
            number_matches = len(expected_numbers.intersection(actual_numbers))
            number_score = number_matches / len(expected_numbers)
        else:
            number_score = 0.0
        
        # 3. Length similarity score
        expected_len = len(expected)
        actual_len = len(actual)
        if expected_len > 0:
            length_score = 1.0 - abs(actual_len - expected_len) / max(expected_len, actual_len)
        else:
            length_score = 0.0
        
        # 4. Exact match score
        exact_match = 1.0 if actual_lower.strip() == expected_lower.strip() else 0.0
        
        return {
            "keyword_matching": keyword_score,
            "number_matching": number_score,
            "length_similarity": length_score,
            "exact_match": exact_match
        }
    
    def evaluate_single_model(self, model, tokenizer, model_name: str) -> Dict[str, Any]:
        """Evaluate một model"""
        logger.info(f"Evaluating {model_name}...")
        
        results = {
            "model_name": model_name,
            "responses": [],
            "metrics": {}
        }
        
        # Generate responses
        for i, (question, expected) in enumerate(zip(self.test_questions, self.expected_answers)):
            logger.info(f"Question {i+1}: {question}")
            
            response = self.generate_response(model, tokenizer, question)
            
            # Calculate simple metrics
            metrics = self.calculate_simple_metrics(response, expected)
            
            case_result = {
                "question": question,
                "expected": expected,
                "response": response,
                "metrics": metrics
            }
            
            results["responses"].append(case_result)
            logger.info(f"Response: {response[:100]}...")
            logger.info(f"Metrics: {metrics}")
        
        # Calculate overall metrics
        overall_metrics = {}
        for metric_name in ["keyword_matching", "number_matching", "length_similarity", "exact_match"]:
            scores = [r["metrics"][metric_name] for r in results["responses"]]
            overall_metrics[metric_name] = {
                "mean": sum(scores) / len(scores),
                "scores": scores
            }
        
        results["metrics"] = overall_metrics
        
        # Calculate weighted score
        weighted_score = (
            overall_metrics["keyword_matching"]["mean"] * 0.3 +
            overall_metrics["number_matching"]["mean"] * 0.4 +
            overall_metrics["length_similarity"]["mean"] * 0.2 +
            overall_metrics["exact_match"]["mean"] * 0.1
        )
        results["weighted_score"] = weighted_score
        
        logger.info(f"{model_name} evaluation completed. Weighted score: {weighted_score:.3f}")
        return results
    
    def run_evaluation(self):
        """Chạy evaluation"""
        logger.info("Starting simple evaluation (no API required)...")
        
        try:
            # Load models
            self.load_models()
            
            # Evaluate base model
            base_results = self.evaluate_single_model(
                self.base_model, 
                self.base_tokenizer, 
                "Base Model"
            )
            
            # Evaluate finetuned model
            finetuned_results = self.evaluate_single_model(
                self.finetuned_model, 
                self.finetuned_tokenizer, 
                "Finetuned Model"
            )
            
            # Compare results
            self.print_comparison(base_results, finetuned_results)
            
            # Save results
            self.save_results(base_results, finetuned_results)
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            raise
    
    def print_comparison(self, base_results: Dict[str, Any], finetuned_results: Dict[str, Any]):
        """In kết quả so sánh"""
        print("\n" + "="*80)
        print("MODEL COMPARISON RESULTS (No API Required)")
        print("="*80)
        
        print(f"\nBase Model: {base_results['model_name']}")
        print(f"Weighted Score: {base_results['weighted_score']:.3f}")
        
        print(f"\nFinetuned Model: {finetuned_results['model_name']}")
        print(f"Weighted Score: {finetuned_results['weighted_score']:.3f}")
        
        improvement = finetuned_results['weighted_score'] - base_results['weighted_score']
        print(f"\nOverall Improvement: {improvement:.3f}")
        
        print(f"\nDetailed Metrics Comparison:")
        for metric in ['keyword_matching', 'number_matching', 'length_similarity', 'exact_match']:
            base_score = base_results['metrics'][metric]['mean']
            finetuned_score = finetuned_results['metrics'][metric]['mean']
            metric_improvement = finetuned_score - base_score
            
            print(f"  {metric}: {base_score:.3f} → {finetuned_score:.3f} ({metric_improvement:+.3f})")
        
        print("\nDetailed Responses:")
        for i, (base_resp, finetuned_resp) in enumerate(zip(base_results['responses'], finetuned_results['responses'])):
            print(f"\nQuestion {i+1}: {base_resp['question']}")
            print(f"Expected: {base_resp['expected']}")
            print(f"Base: {base_resp['response'][:100]}...")
            print(f"Finetuned: {finetuned_resp['response'][:100]}...")
        
        print("="*80)
    
    def save_results(self, base_results: Dict[str, Any], finetuned_results: Dict[str, Any]):
        """Lưu kết quả"""
        results = {
            "evaluation_time": datetime.now().isoformat(),
            "evaluation_type": "simple_no_api",
            "base_model": base_results,
            "finetuned_model": finetuned_results,
            "comparison": {
                "improvement": finetuned_results['weighted_score'] - base_results['weighted_score'],
                "improvement_percentage": (
                    (finetuned_results['weighted_score'] - base_results['weighted_score']) / 
                    base_results['weighted_score'] * 100
                ) if base_results['weighted_score'] > 0 else 0
            }
        }
        
        output_path = "./evaluation_results/simple_evaluation_no_api.json"
        Path("./evaluation_results").mkdir(exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Results saved to {output_path}")

def main():
    """Main function"""
    evaluator = SimpleMathEvaluator()
    evaluator.run_evaluation()

if __name__ == "__main__":
    main() 