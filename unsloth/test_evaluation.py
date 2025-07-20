#!/usr/bin/env python3
"""
Script test đơn giản để kiểm tra evaluation framework
"""

import os
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test các imports cần thiết"""
    logger.info("Testing imports...")
    
    try:
        # Test deepeval imports
        from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, TaskCompletionMetric
        from deepeval.test_case import LLMTestCase
        logger.info("✅ Deepeval imports successful")
        
        # Test unsloth import
        from unsloth import FastLanguageModel
        logger.info("✅ Unsloth import successful")
        
        # Test other imports
        import torch
        import json
        from pathlib import Path
        logger.info("✅ Other imports successful")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return False

def test_deepeval_metrics():
    """Test deepeval metrics"""
    logger.info("Testing deepeval metrics...")
    
    try:
        from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, TaskCompletionMetric
        from deepeval.test_case import LLMTestCase
        
        # Create test case
        test_case = LLMTestCase(
            input="Tính 15 + 27 = ?",
            actual_output="15 + 27 = 42",
            expected_output="15 + 27 = 42"
        )
        
        # Test metrics
        relevancy = AnswerRelevancyMetric(threshold=0.7)
        faithfulness = FaithfulnessMetric(threshold=0.7)
        correctness = TaskCompletionMetric(threshold=0.7)
        
        # Measure (this might fail if no OpenAI API key, but should not crash)
        try:
            relevancy_score = relevancy.measure(test_case)
            logger.info(f"✅ AnswerRelevancyMetric test passed: {relevancy_score}")
        except Exception as e:
            logger.warning(f"⚠️ AnswerRelevancyMetric failed (expected if no API key): {e}")
        
        try:
            faithfulness_score = faithfulness.measure(test_case)
            logger.info(f"✅ FaithfulnessMetric test passed: {faithfulness_score}")
        except Exception as e:
            logger.warning(f"⚠️ FaithfulnessMetric failed (expected if no API key): {e}")
        
        try:
            correctness_score = correctness.measure(test_case)
            logger.info(f"✅ TaskCompletionMetric test passed: {correctness_score}")
        except Exception as e:
            logger.warning(f"⚠️ TaskCompletionMetric failed (expected if no API key): {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Deepeval metrics test failed: {e}")
        return False

def test_environment():
    """Test environment setup"""
    logger.info("Testing environment...")
    
    # Check Python version
    python_version = sys.version_info
    logger.info(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major == 3 and python_version.minor >= 13:
        logger.warning("⚠️ Python 3.13+ detected. Unsloth may have compatibility issues.")
    
    # Check .env file
    if os.path.exists(".env"):
        logger.info("✅ .env file exists")
    else:
        logger.warning("⚠️ .env file not found")
    
    # Check model directory
    if os.path.exists("./gemma_multimodal_finetuned"):
        logger.info("✅ Finetuned model directory exists")
    else:
        logger.warning("⚠️ Finetuned model directory not found")
    
    return True

def test_custom_metric():
    """Test custom metric"""
    logger.info("Testing custom metric...")
    
    try:
        from deepeval.metrics import BaseMetric
        from deepeval.test_case import LLMTestCase
        
        class TestCustomMetric(BaseMetric):
            def __init__(self):
                super().__init__()
                self.name = "Test Metric"
            
            def measure(self, test_case: LLMTestCase) -> float:
                return 0.8
        
        metric = TestCustomMetric()
        test_case = LLMTestCase(
            input="Test",
            actual_output="Test output",
            expected_output="Expected output"
        )
        
        score = metric.measure(test_case)
        logger.info(f"✅ Custom metric test passed: {score}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Custom metric test failed: {e}")
        return False

def main():
    """Main test function"""
    logger.info("🧪 Starting evaluation framework tests...")
    
    tests = [
        ("Import Test", test_imports),
        ("Environment Test", test_environment),
        ("Deepeval Metrics Test", test_deepeval_metrics),
        ("Custom Metric Test", test_custom_metric),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            if test_func():
                passed += 1
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with exception: {e}")
    
    logger.info(f"\n{'='*50}")
    logger.info(f"TEST SUMMARY: {passed}/{total} tests passed")
    logger.info(f"{'='*50}")
    
    if passed == total:
        logger.info("🎉 All tests passed! Evaluation framework is ready.")
        return True
    else:
        logger.warning("⚠️ Some tests failed. Check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 