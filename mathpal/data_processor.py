"""
Data processing module for Gemma3N fine-tuning
Handles dataset loading, preprocessing, and formatting
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datasets import Dataset, load_dataset, DatasetDict
from transformers import PreTrainedTokenizer
import torch
from pathlib import Path
import json
import pandas as pd

from config import DatasetConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MathDatasetProcessor:
    """
    Processor cho math dataset với Gemma3N format
    """
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.tokenizer = None
        self.raw_datasets = None
        self.processed_datasets = None
        
    def set_tokenizer(self, tokenizer: PreTrainedTokenizer):
        """Set tokenizer for dataset processing"""
        self.tokenizer = tokenizer
        logger.info(f"Tokenizer set: {type(tokenizer).__name__}")
    
    def load_dataset(self) -> DatasetDict:
        """
        Load dataset từ HuggingFace hoặc local path
        """
        try:
            logger.info(f"Loading dataset: {self.config.dataset_name}")
            
            # Load dataset
            if Path(self.config.dataset_name).exists():
                # Local dataset
                self.raw_datasets = load_dataset("json", data_files=self.config.dataset_name)
            else:
                # HuggingFace dataset
                self.raw_datasets = load_dataset(self.config.dataset_name)
            
            logger.info(f"Dataset loaded successfully:")
            for split, dataset in self.raw_datasets.items():
                logger.info(f"  {split}: {len(dataset)} samples")
                
            return self.raw_datasets
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def process_sample(self, sample: Dict[str, str]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Convert single sample to Gemma3N conversation format
        
        Expected input format:
        {
            "question": "Câu hỏi toán học",
            "solution": "Lời giải chi tiết"
        }
        """
        try:
            # Create conversation format
            conversations = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": sample["question"]}]
                },
                {
                    "role": "assistant", 
                    "content": [{"type": "text", "text": sample["solution"]}]
                }
            ]
            
            return {"conversations": conversations}
            
        except KeyError as e:
            logger.error(f"Missing required field in sample: {e}")
            logger.error(f"Sample keys: {list(sample.keys())}")
            raise
        except Exception as e:
            logger.error(f"Error processing sample: {e}")
            raise
    
    def format_conversations(self, examples: Dict[str, List]) -> Dict[str, List[str]]:
        """
        Apply chat template to conversations using tokenizer
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not set. Call set_tokenizer() first.")
        
        convos = examples["conversations"]
        texts = []
        
        for convo in convos:
            try:
                # Apply chat template
                formatted_text = self.tokenizer.apply_chat_template(
                    convo, 
                    tokenize=False, 
                    add_generation_prompt=False
                ).removeprefix('<bos>')  # Remove BOS token as per notebook
                
                texts.append(formatted_text)
                
            except Exception as e:
                logger.error(f"Error formatting conversation: {e}")
                logger.error(f"Conversation: {convo}")
                # Add fallback formatting
                texts.append(self._fallback_format(convo))
        
        return {"text": texts}
    
    def _fallback_format(self, conversation: List[Dict]) -> str:
        """
        Fallback formatting nếu chat template fails
        """
        formatted_parts = []
        
        for message in conversation:
            role = message["role"]
            content = message["content"][0]["text"] if isinstance(message["content"], list) else message["content"]
            
            if role == "user":
                formatted_parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
            elif role == "assistant":
                formatted_parts.append(f"<start_of_turn>model\n{content}<end_of_turn>")
        
        return "\n".join(formatted_parts)
    
    def process_dataset(self, dataset: Dataset) -> Dataset:
        """
        Process entire dataset: sample processing + formatting
        """
        logger.info(f"Processing dataset with {len(dataset)} samples...")
        
        # Step 1: Convert to conversation format
        logger.info("Step 1: Converting to conversation format...")
        processed_dataset = dataset.map(
            self.process_sample,
            num_proc=self.config.dataset_num_proc,
            desc="Converting to conversations"
        )
        
        # Step 2: Apply chat template formatting
        if self.tokenizer is not None:
            logger.info("Step 2: Applying chat template...")
            processed_dataset = processed_dataset.map(
                self.format_conversations,
                batched=True,
                num_proc=self.config.dataset_num_proc,
                desc="Formatting conversations"
            )
        else:
            logger.warning("Tokenizer not set, skipping chat template formatting")
        
        # Step 3: Limit samples if configured
        if self.config.max_samples is not None:
            logger.info(f"Step 3: Limiting to {self.config.max_samples} samples...")
            processed_dataset = processed_dataset.select(range(min(self.config.max_samples, len(processed_dataset))))
        
        logger.info(f"Dataset processing complete: {len(processed_dataset)} samples")
        return processed_dataset
    
    def prepare_datasets(self) -> Dict[str, Dataset]:
        """
        Chuẩn bị train/eval datasets
        """
        if self.raw_datasets is None:
            self.load_dataset()
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer not set. Call set_tokenizer() first.")
        
        prepared_datasets = {}
        
        # Process train dataset
        if self.config.train_split in self.raw_datasets:
            logger.info(f"Preparing train dataset...")
            train_dataset = self.process_dataset(self.raw_datasets[self.config.train_split])
            prepared_datasets["train"] = train_dataset
            
        # Process test/eval dataset
        if self.config.test_split in self.raw_datasets:
            logger.info(f"Preparing eval dataset...")
            eval_dataset = self.process_dataset(self.raw_datasets[self.config.test_split])
            prepared_datasets["eval"] = eval_dataset
        elif "train" in prepared_datasets:
            # Split train dataset if no eval split exists
            logger.info("No eval split found, splitting train dataset...")
            train_test = prepared_datasets["train"].train_test_split(test_size=0.1, seed=42)
            prepared_datasets["train"] = train_test["train"]
            prepared_datasets["eval"] = train_test["test"]
        
        self.processed_datasets = prepared_datasets
        
        # Log dataset statistics
        self.log_dataset_stats()
        
        return prepared_datasets
    
    def log_dataset_stats(self):
        """Log dataset statistics"""
        if self.processed_datasets is None:
            return
            
        logger.info("📊 Dataset Statistics:")
        logger.info("=" * 50)
        
        for split_name, dataset in self.processed_datasets.items():
            logger.info(f"{split_name.upper()} SET:")
            logger.info(f"  Samples: {len(dataset)}")
            
            if "text" in dataset.column_names:
                # Analyze text lengths
                lengths = [len(text.split()) for text in dataset["text"]]
                avg_length = sum(lengths) / len(lengths)
                max_length = max(lengths)
                min_length = min(lengths)
                
                logger.info(f"  Avg text length: {avg_length:.1f} words")
                logger.info(f"  Max text length: {max_length} words")
                logger.info(f"  Min text length: {min_length} words")
            
            # Sample a few examples
            if len(dataset) > 0:
                logger.info(f"  Sample questions:")
                for i in range(min(2, len(dataset))):
                    if "conversations" in dataset[i]:
                        question = dataset[i]["conversations"][0]["content"][0]["text"]
                        logger.info(f"    {i+1}. {question[:100]}...")
            
            logger.info("")
    
    def save_processed_datasets(self, output_dir: str):
        """Save processed datasets to disk"""
        if self.processed_datasets is None:
            logger.warning("No processed datasets to save")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for split_name, dataset in self.processed_datasets.items():
            save_path = output_path / f"processed_{split_name}.json"
            dataset.to_json(save_path)
            logger.info(f"Saved {split_name} dataset to {save_path}")
    
    def load_processed_datasets(self, input_dir: str) -> Dict[str, Dataset]:
        """Load processed datasets from disk"""
        input_path = Path(input_dir)
        
        datasets = {}
        for split_name in ["train", "eval"]:
            file_path = input_path / f"processed_{split_name}.json"
            if file_path.exists():
                dataset = Dataset.from_json(str(file_path))
                datasets[split_name] = dataset
                logger.info(f"Loaded {split_name} dataset from {file_path}")
        
        self.processed_datasets = datasets
        return datasets
    
    def validate_dataset_format(self, dataset: Dataset) -> bool:
        """
        Validate dataset format before training
        """
        required_columns = [self.config.dataset_text_field]
        
        # Check required columns
        missing_columns = set(required_columns) - set(dataset.column_names)
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        # Check for empty samples
        if len(dataset) == 0:
            logger.error("Dataset is empty")
            return False
        
        # Check sample format
        try:
            sample = dataset[0]
            if self.config.dataset_text_field not in sample:
                logger.error(f"Missing {self.config.dataset_text_field} field in sample")
                return False
            
            text = sample[self.config.dataset_text_field]
            if not isinstance(text, str) or len(text.strip()) == 0:
                logger.error("Empty or invalid text in sample")
                return False
                
        except Exception as e:
            logger.error(f"Error validating sample format: {e}")
            return False
        
        logger.info("✅ Dataset format validation passed")
        return True
    
    def get_sample_for_inference(self, index: int = 0) -> Dict[str, Any]:
        """
        Get formatted sample for inference testing
        """
        if self.processed_datasets is None or "eval" not in self.processed_datasets:
            raise ValueError("No processed eval dataset available")
        
        dataset = self.processed_datasets["eval"]
        if index >= len(dataset):
            raise IndexError(f"Index {index} out of range for dataset size {len(dataset)}")
        
        sample = dataset[index]
        
        # Extract original question and solution
        if "conversations" in sample:
            conversations = sample["conversations"]
            question = conversations[0]["content"][0]["text"]
            solution = conversations[1]["content"][0]["text"]
            
            return {
                "question": question,
                "expected_solution": solution,
                "formatted_text": sample.get("text", ""),
                "index": index
            }
        else:
            return sample


def create_data_processor(config: DatasetConfig) -> MathDatasetProcessor:
    """
    Factory function to create data processor
    """
    return MathDatasetProcessor(config)


def test_data_processor():
    """
    Test function for data processor
    """
    from config import DatasetConfig
    
    # Create test config
    config = DatasetConfig()
    config.max_samples = 10  # Limit for testing
    
    # Create processor
    processor = create_data_processor(config)
    
    try:
        # Load dataset
        raw_datasets = processor.load_dataset()
        logger.info("✅ Dataset loading test passed")
        
        # Test sample processing
        if "train" in raw_datasets:
            sample = raw_datasets["train"][0]
            processed_sample = processor.process_sample(sample)
            logger.info("✅ Sample processing test passed")
            logger.info(f"Processed sample keys: {list(processed_sample.keys())}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Data processor test failed: {e}")
        return False


if __name__ == "__main__":
    # Run test
    test_data_processor()
