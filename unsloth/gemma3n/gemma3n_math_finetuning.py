#!/usr/bin/env python3
"""
Optimized Gemma3N Fine-tuning Script for 6th Grade Math Problems
Dataset: https://huggingface.co/datasets/ngohongthai/exam-sixth_grade-instruct-dataset

Features:
- Handles text + image multimodal data
- Processes markdown image URLs
- Optimized for Gemma3N vision model
- Custom data collator for stable training
"""

import os
import re
import io
import zipfile
from typing import Tuple, List, Dict, Any, Optional
from PIL import Image
import requests
import torch
from datasets import load_dataset, Dataset
from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig
from unsloth import FastVisionModel, get_chat_template

# Import Comet ML configuration
try:
    from comet_config import COMET_CONFIG, validate_comet_config, setup_comet_environment
    COMET_AVAILABLE = True
except ImportError:
    print("Warning: comet_config.py not found. Using default Comet ML settings.")
    COMET_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Model settings
    "model_name": "unsloth/gemma-3n-E4B",
    "max_seq_length": 2048,
    "load_in_4bit": True,
    
    # Dataset settings
    "dataset_name": "ngohongthai/exam-sixth_grade-instruct-dataset",
    "train_split": "train",
    "test_split": "test",
    
    # Training settings
    "output_dir": "outputs/gemma3n-math-tutor",
    "max_steps": 200,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "warmup_ratio": 0.03,
    "weight_decay": 0.01,
    "logging_steps": 5,
    "save_steps": 50,
    
    # LoRA settings
    "lora_r": 32,
    "lora_alpha": 32,
    "lora_dropout": 0.0,
    
    # System settings  
    "use_gradient_checkpointing": False,  # MUST be False for vision models to avoid CheckpointError
    "report_to": "comet_ml",  # Change to "tensorboard", "wandb" if needed
    "seed": 42,
    
    # Comet ML settings
    "comet_workspace": None,  # Set your Comet workspace name
    "comet_project": "gemma3n-math-tutor",  # Set your Comet project name
    
    # Data collator settings
    "handle_text_only_samples": True,  # Whether to include text-only samples with placeholder images
    "debug_data_collator": True,  # Enable detailed logging for debugging
    "force_image_tokens": True,  # Force add image tokens even if insertion fails
    "use_fixed_collator": True,  # Use fixed collator that modifies conversation structure
}

# =============================================================================
# IMAGE PROCESSING UTILITIES
# =============================================================================

def url_to_image(url: str, timeout: int = 10) -> Optional[Image.Image]:
    """
    Download and convert URL to PIL Image.
    
    Args:
        url: Image URL
        timeout: Request timeout in seconds
        
    Returns:
        PIL Image object or None if failed
    """
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        return image
    except (requests.exceptions.RequestException, IOError) as e:
        print(f"Failed to load image from {url}: {e}")
        return None

def extract_image_urls_from_markdown(text: str) -> Tuple[str, List[str]]:
    """
    Extract image URLs from markdown text and replace with placeholders.
    
    Args:
        text: Markdown text containing image links
        
    Returns:
        Tuple of (cleaned_text, list_of_image_urls)
    """
    # Pattern for markdown images: ![alt](url)
    image_pattern = r"!\[.*?\]\((.*?)\)"
    image_urls = re.findall(image_pattern, text)
    
    # Remove image markdown syntax
    cleaned_text = re.sub(image_pattern, " ", text).strip()
    
    return cleaned_text, image_urls

def process_markdown_for_model(text: str) -> Tuple[str, List[Image.Image]]:
    """
    Process markdown text to extract text and images for multimodal model.
    
    Args:
        text: Input markdown text
        
    Returns:
        Tuple of (processed_text, list_of_pil_images)
    """
    cleaned_text, image_urls = extract_image_urls_from_markdown(text)
    
    # Download images
    images = []
    for url in image_urls:
        image = url_to_image(url)
        if image:
            images.append(image)
        else:
            print(f"Warning: Failed to load image from {url}")
    
    return cleaned_text, images

# =============================================================================
# DATASET PROCESSING
# =============================================================================

def create_conversation_content(text: str, images: List[Image.Image]) -> List[Dict[str, Any]]:
    """
    Create conversation content list with text and images.
    
    Args:
        text: Text content
        images: List of PIL images
        
    Returns:
        List of content dictionaries
    """
    content = [{"type": "text", "text": text}]
    
    # Add images
    for image in images:
        content.append({"type": "image", "image": image})
    
    return content

def process_math_sample(sample: Dict[str, str]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Process a single math problem sample into conversation format.
    
    Args:
        sample: Dataset sample with 'question' and 'solution' keys
        
    Returns:
        Dictionary with 'conversations' key containing the formatted conversation
    """
    # Process question
    question_text, question_images = process_markdown_for_model(sample["question"])
    user_content = create_conversation_content(question_text, question_images)
    
    # Process solution (usually text-only, but check for images)
    solution_text, solution_images = process_markdown_for_model(sample["solution"])
    assistant_content = create_conversation_content(solution_text, solution_images)
    
    # Create conversation
    conversations = [
        {
            "role": "user", 
            "content": user_content
        },
        {
            "role": "assistant",
            "content": assistant_content
        }
    ]
    
    return {"conversations": conversations}

def debug_dataset_sample(sample, index=0):
    """Debug a single dataset sample to understand its structure."""
    print(f"\n🔍 Debugging sample {index}:")
    print(f"Sample keys: {sample.keys()}")
    
    if "conversations" in sample:
        conversations = sample["conversations"]
        print(f"Number of conversations: {len(conversations)}")
        
        for i, conv in enumerate(conversations):
            print(f"\nConversation {i}:")
            print(f"  Role: {conv.get('role', 'unknown')}")
            content = conv.get('content', [])
            print(f"  Content items: {len(content)}")
            
            for j, item in enumerate(content):
                print(f"    Item {j}: type={item.get('type', 'unknown')}")
                if item.get('type') == 'image':
                    img = item.get('image')
                    if img is not None:
                        print(f"      Image: {type(img)}, size={getattr(img, 'size', 'unknown')}")
                    else:
                        print(f"      Image: None")
                elif item.get('type') == 'text':
                    text = item.get('text', '')
                    print(f"      Text: {text[:50]}...")

def prepare_dataset(dataset_name: str, split: str) -> Dataset:
    """
    Load and prepare the math dataset.
    
    Args:
        dataset_name: HuggingFace dataset name
        split: Dataset split to load
        
    Returns:
        Processed Dataset object
    """
    print(f"Loading dataset: {dataset_name}, split: {split}")
    raw_dataset = load_dataset(dataset_name, split=split)
    
    print(f"Processing {len(raw_dataset)} samples...")
    processed_data = []
    
    for i, sample in enumerate(raw_dataset):
        try:
            processed_sample = process_math_sample(sample)
            processed_data.append(processed_sample)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(raw_dataset)} samples")
                
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    print(f"Successfully processed {len(processed_data)} samples")
    
    # Debug first few samples
    if len(processed_data) > 0:
        print("\n📊 Dataset debugging:")
        debug_dataset_sample(processed_data[0], 0)
        if len(processed_data) > 1:
            debug_dataset_sample(processed_data[1], 1)
    
    return Dataset.from_list(processed_data)

# =============================================================================
# CUSTOM DATA COLLATOR
# =============================================================================

class HybridVisionDataCollator:
    """
    Advanced data collator xử lý cả text-only và text+image samples.
    Tự động detect và xử lý mixed batches một cách thông minh.
    """
    
    def __init__(self, processor, handle_text_only=True, debug_mode=True, force_image_tokens=True):
        self.processor = processor
        self.handle_text_only = handle_text_only
        self.debug_mode = debug_mode
        self.force_image_tokens = force_image_tokens
        self.placeholder_image = None
        
    def _create_placeholder_image(self):
        """Create a minimal placeholder image cho text-only samples."""
        if self.placeholder_image is None:
            # Tạo image nhỏ để minimize memory impact
            self.placeholder_image = Image.new('RGB', (32, 32), color=(245, 245, 245))
        return self.placeholder_image
        
    def _validate_and_process_image(self, img):
        """Validate and process a single image."""
        if img is None:
            return None
            
        try:
            if not hasattr(img, 'convert'):
                return None
                
            img = img.convert('RGB')
            
            if img.size[0] < 1 or img.size[1] < 1:
                return None
                
            return img
            
        except Exception as e:
            return None
    
    def _extract_images_from_conversation(self, conv):
        """Extract and validate all images from a conversation."""
        images = []
        
        for message in conv:
            for content in message.get("content", []):
                if content.get("type") == "image" and "image" in content:
                    img = content["image"]
                    processed_img = self._validate_and_process_image(img)
                    if processed_img is not None:
                        images.append(processed_img)
        
        return images
    
    def _has_real_images(self, conv):
        """Check if conversation has any real images."""
        images = self._extract_images_from_conversation(conv)
        return len(images) > 0
    
    def _create_text_only_conversation(self, conv):
        """Convert conversation to text-only format cho processor."""
        text_only_conv = []
        
        for message in conv:
            text_only_message = {
                "role": message["role"],
                "content": []
            }
            
            # Extract chỉ text content
            for content in message.get("content", []):
                if content.get("type") == "text":
                    text_only_message["content"].append(content)
            
            # Ensure có ít nhất empty text content
            if not text_only_message["content"]:
                text_only_message["content"] = [{"type": "text", "text": ""}]
                
            text_only_conv.append(text_only_message)
        
        return text_only_conv
    
    def _insert_image_token_strategically(self, text, num_images=1):
        """Insert image token vào position thông minh trong text."""
        if '<image>' in text:
            return text
            
        # Simple và reliable strategy: Insert ở đầu text
        # Điều này đảm bảo image token luôn có mặt và ở vị trí model có thể nhận diện
        image_tokens = ['<image>'] * num_images
        
        # Nếu text bắt đầu với BOS token hoặc special token, insert sau đó
        if text.startswith(('<|', '<bos>', '<s>')):
            lines = text.split('\n', 1)
            if len(lines) == 2:
                result = lines[0] + '\n' + '\n'.join(image_tokens) + '\n' + lines[1]
            else:
                result = text + '\n' + '\n'.join(image_tokens)
        else:
            # Insert ở đầu hoàn toàn
            result = '\n'.join(image_tokens) + '\n' + text
        
        # Debug log để track
        if self.debug_mode:
            print(f"🔧 Inserted {num_images} image tokens at beginning")
        
        return result
    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate mixed batch của text-only và text+image samples.
        
        Args:
            examples: List of processed conversation examples
            
        Returns:
            Batch dictionary with tensors
        """
        try:
            print(f"Processing hybrid batch of {len(examples)} examples...")
            
            # Classify samples
            image_samples = []
            text_only_samples = []
            
            for idx, example in enumerate(examples):
                conv = example["conversations"]
                if self._has_real_images(conv):
                    image_samples.append((idx, example))
                    print(f"Sample {idx}: IMAGE SAMPLE ✅")
                else:
                    text_only_samples.append((idx, example))
                    print(f"Sample {idx}: TEXT ONLY 📝")
            
            print(f"Batch composition: {len(image_samples)} image samples, {len(text_only_samples)} text-only")
            
            # Handle different scenarios
            if len(image_samples) > 0 and len(text_only_samples) > 0:
                # Mixed batch - xử lý hybrid
                return self._process_mixed_batch(image_samples, text_only_samples)
            elif len(image_samples) > 0:
                # Pure image batch
                return self._process_image_batch(image_samples)
            elif len(text_only_samples) > 0 and self.handle_text_only:
                # Pure text batch với placeholder images
                return self._process_text_only_batch(text_only_samples)
            else:
                raise ValueError("No valid samples to process or text-only handling disabled!")
                
        except Exception as e:
            print(f"Critical error in hybrid data collator: {e}")
            import traceback
            traceback.print_exc()
            raise e
    
    def _process_image_batch(self, image_samples):
        """Process batch chỉ có image samples."""
        print("Processing pure image batch...")
        
        texts = []
        images_list = []
        
        for idx, example in image_samples:
            conv = example["conversations"]
            images = self._extract_images_from_conversation(conv)
            
            # Generate text
            text = self.processor.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=False
            )
            
            # Validate token count
            image_token_count = text.count('<image>')
            actual_image_count = len(images)
            
            print(f"Image sample {idx}: {actual_image_count} images, {image_token_count} tokens")
            
            # Sync tokens và images
            if image_token_count != actual_image_count:
                if image_token_count < actual_image_count:
                    images = images[:image_token_count] if image_token_count > 0 else images[:1]
                elif image_token_count > actual_image_count:
                    # Thêm placeholder images
                    while len(images) < image_token_count:
                        images.append(self._create_placeholder_image())
            
            texts.append(text)
            images_list.append(images)
        
        return self._create_batch(texts, images_list)
    
    def _process_text_only_batch(self, text_only_samples):
        """Process batch chỉ có text samples với placeholder images."""
        print("Processing text-only batch with placeholder images...")
        
        texts = []
        images_list = []
        
        for idx, example in text_only_samples:
            conv = example["conversations"]
            
            # Convert to text-only format
            text_only_conv = self._create_text_only_conversation(conv)
            
            # Generate text
            text = self.processor.apply_chat_template(
                text_only_conv, tokenize=False, add_generation_prompt=False
            )
            
            # Add 1 placeholder image và corresponding token
            placeholder = self._create_placeholder_image()
            # Use correct Gemma3n token instead of generic <image>
            lines = text.split('\n')
            if len(lines) > 0 and any(token in lines[0] for token in ['<bos>', '<s>', '<|']):
                lines.insert(1, '<image_soft_token>')
            else:
                lines.insert(0, '<image_soft_token>')
            text_with_image_token = '\n'.join(lines)
            
            print(f"Text sample {idx}: Added 1 placeholder image and token")
            
            texts.append(text_with_image_token)
            images_list.append([placeholder])
        
        return self._create_batch(texts, images_list)
    
    def _process_mixed_batch(self, image_samples, text_only_samples):
        """Process mixed batch có cả image và text-only samples."""
        print("Processing mixed batch...")
        
        texts = []
        images_list = []
        
        # Process image samples first
        for idx, example in image_samples:
            conv = example["conversations"]
            images = self._extract_images_from_conversation(conv)
            
            text = self.processor.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=False
            )
            
            image_token_count = text.count('<image>')
            actual_image_count = len(images)
            
            print(f"Mixed - Image sample {idx}: {actual_image_count} images, {image_token_count} tokens")
            
            # Sync tokens và images
            if image_token_count != actual_image_count:
                if image_token_count < actual_image_count:
                    images = images[:image_token_count] if image_token_count > 0 else images[:1]
                elif image_token_count > actual_image_count:
                    while len(images) < image_token_count:
                        images.append(self._create_placeholder_image())
            
            texts.append(text)
            images_list.append(images)
        
        # Process text-only samples with placeholders
        for idx, example in text_only_samples:
            conv = example["conversations"]
            text_only_conv = self._create_text_only_conversation(conv)
            
            text = self.processor.apply_chat_template(
                text_only_conv, tokenize=False, add_generation_prompt=False
            )
            
            # Add placeholder
            placeholder = self._create_placeholder_image()
            text_with_token = self._insert_image_token_strategically(text, num_images=1)
            
            print(f"Mixed - Text sample {idx}: Added placeholder image and token")
            
            texts.append(text_with_token)
            images_list.append([placeholder])
        
        return self._create_batch(texts, images_list)
    
    def _create_batch(self, texts, images_list):
        """Create final batch tensor."""
        print(f"Creating batch: {len(texts)} texts, {len(images_list)} image lists")
        
        # Enhanced validation với detailed logging
        for i, (text, imgs) in enumerate(zip(texts, images_list)):
            token_count = text.count('<image_soft_token>')
            image_count = len(imgs)
            
            print(f"🔍 Sample {i} validation: {token_count} <image_soft_token>, {image_count} images")
            
            if token_count != image_count:
                print(f"⚠️  Sample {i}: Token/image mismatch ({token_count} vs {image_count})")
                print(f"📝 Text preview: {repr(text[:200])}...")
                
                # Fix mismatch với detailed logic
                if token_count > image_count:
                    print(f"🔧 Adding {token_count - image_count} placeholder images")
                    while len(imgs) < token_count:
                        imgs.append(self._create_placeholder_image())
                    images_list[i] = imgs
                elif image_count > token_count:
                    print(f"🔧 Truncating to {token_count} images")
                    images_list[i] = imgs[:token_count] if token_count > 0 else imgs[:1]
                elif token_count == 0 and image_count > 0:
                    # Critical case: No tokens but have images - force add token
                    print(f"🚨 CRITICAL: No image tokens but have {image_count} images - forcing token insertion")
                    texts[i] = '<image_soft_token>\n' + text
                    print(f"🔧 Fixed text preview: {repr(texts[i][:200])}...")
        
        # Final validation trước khi gửi to processor
        print("🔍 Final validation before processor...")
        total_tokens = sum(text.count('<image_soft_token>') for text in texts)
        total_images = sum(len(imgs) for imgs in images_list)
        print(f"📊 Total: {total_tokens} <image_soft_token>, {total_images} images")
        
        if total_tokens != total_images:
            print(f"🚨 FINAL MISMATCH DETECTED: {total_tokens} tokens vs {total_images} images")
            # Emergency fix: Ensure all samples have at least 1 image token
            for i, text in enumerate(texts):
                if '<image_soft_token>' not in text:
                    print(f"🚨 Emergency fix for sample {i}: Adding <image_soft_token>")
                    texts[i] = '<image_soft_token>\n' + text
        
        # Process with processor
        print("Sending to processor...")
        try:
            batch = self.processor(
                text=texts,
                images=images_list,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=CONFIG["max_seq_length"]
            )
        except Exception as e:
            print(f"🚨 PROCESSOR ERROR: {e}")
            # Debug information
            for i, (text, imgs) in enumerate(zip(texts, images_list)):
                print(f"Sample {i}: {text.count('<image_soft_token>')} <image_soft_token>, {len(imgs)} images")
                print(f"Text: {repr(text[:100])}...")
            raise
        
        # Create labels
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        
        print(f"✅ Batch created successfully: {batch.keys()}")
        if "pixel_values" in batch:
            print(f"pixel_values shape: {batch['pixel_values'].shape}")
        print(f"input_ids shape: {batch['input_ids'].shape}")
        
        return batch

# =============================================================================
# MODEL SETUP AND TRAINING
# =============================================================================

def setup_model_and_processor(config: Dict[str, Any]):
    """
    Load and setup Gemma3N model and processor.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (model, processor)
    """
    print("Loading Gemma3N model and processor...")
    
    # Load model and processor
    model, processor = FastVisionModel.from_pretrained(
        config["model_name"],
        max_seq_length=config["max_seq_length"],
        load_in_4bit=config["load_in_4bit"],
        use_gradient_checkpointing="unsloth" if config["use_gradient_checkpointing"] else False,
    )
    
    # Apply LoRA
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        random_state=config["seed"],
        use_rslora=False,
        target_modules="all-linear",
        modules_to_save=["lm_head", "embed_tokens"],
    )
    
    # Setup chat template
    processor = get_chat_template(processor, "gemma-3n")
    
    print("Model and processor setup complete!")
    return model, processor

def test_data_collator(train_dataset, processor, num_samples=2):
    """Test the data collator with a few samples to catch issues early."""
    print(f"\n🧪 Testing data collator with {num_samples} samples...")
    
    try:
        # Create data collator
        collator = HybridVisionDataCollator(processor, handle_text_only=CONFIG["handle_text_only_samples"])
        
        # Test with a small batch
        test_samples = [train_dataset[i] for i in range(min(num_samples, len(train_dataset)))]
        
        print(f"Test samples prepared: {len(test_samples)}")
        
        # Try to collate
        batch = collator(test_samples)
        
        print("✅ Data collator test passed!")
        print(f"Batch keys: {batch.keys()}")
        for key, value in batch.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {type(value)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data collator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def filter_dataset_with_images(dataset):
    """Filter dataset để chỉ giữ samples có real images."""
    print("\n🔍 Filtering dataset for samples with images...")
    
    filtered_samples = []
    total_samples = len(dataset)
    
    for i, sample in enumerate(dataset):
        try:
            conversations = sample["conversations"]
            has_images = False
            
            for message in conversations:
                for content in message.get("content", []):
                    if content.get("type") == "image" and content.get("image") is not None:
                        # Validate image
                        img = content["image"]
                        if hasattr(img, 'convert') and hasattr(img, 'size'):
                            if img.size[0] > 0 and img.size[1] > 0:
                                has_images = True
                                break
                if has_images:
                    break
            
            if has_images:
                filtered_samples.append(sample)
                if i < 10:  # Log first 10
                    print(f"Sample {i}: HAS IMAGES ✅")
            else:
                if i < 10:  # Log first 10
                    print(f"Sample {i}: NO IMAGES ❌ (FILTERED OUT)")
                    
        except Exception as e:
            print(f"Error checking sample {i}: {e}")
            continue
    
    print(f"\n📊 Filtering results:")
    print(f"- Original samples: {total_samples}")
    print(f"- Samples with images: {len(filtered_samples)}")
    print(f"- Filtered out: {total_samples - len(filtered_samples)}")
    print(f"- Retention rate: {len(filtered_samples)/total_samples*100:.1f}%")
    
    if len(filtered_samples) == 0:
        raise ValueError("No samples with images found in dataset!")
    
    # Convert back to Dataset
    from datasets import Dataset
    return Dataset.from_list(filtered_samples)

def create_trainer(model, processor, train_dataset, config: Dict[str, Any]):
    """
    Create optimized SFTTrainer.
    
    Args:
        model: Prepared model
        processor: Model processor
        train_dataset: Training dataset
        config: Configuration dictionary
        
    Returns:
        Configured SFTTrainer
    """
    # Enable training
    FastVisionModel.for_training(model)
    
    # Create data collator với option cho fixed collator
    if config.get("use_fixed_collator", False):
        print("🔧 Using Fixed Gemma3n Data Collator...")
        from fixed_data_collator import FixedGemma3nDataCollator
        data_collator = FixedGemma3nDataCollator(
            processor, 
            handle_text_only=config["handle_text_only_samples"]
        )
    else:
        print("📦 Using Hybrid Vision Data Collator...")
        data_collator = HybridVisionDataCollator(
            processor, 
            handle_text_only=config["handle_text_only_samples"],
            debug_mode=config["debug_data_collator"],
            force_image_tokens=config["force_image_tokens"]
        )
    
    # Training arguments
    training_args = SFTConfig(
        # Basic training settings
        output_dir=config["output_dir"],
        max_steps=config["max_steps"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        
        # Optimization settings
        learning_rate=config["learning_rate"],
        warmup_ratio=config["warmup_ratio"],
        weight_decay=config["weight_decay"],
        optim="adamw_torch_fused",
        lr_scheduler_type="cosine",
        
        # Memory optimization - DISABLED for vision models to avoid CheckpointError
        gradient_checkpointing=False,  # Must be False for Gemma3n vision models
        gradient_checkpointing_kwargs={},  # Empty since not using checkpointing
        max_grad_norm=0.3,
        
        # Logging and saving
        logging_steps=config["logging_steps"],
        save_strategy="steps",
        save_steps=config["save_steps"],
        report_to=config["report_to"],
        
        # Vision-specific settings
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=config["max_seq_length"],
        
        # Reproducibility
        seed=config["seed"],
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        processing_class=processor.tokenizer,
        data_collator=data_collator,
        args=training_args,
    )
    
    return trainer

# =============================================================================
# COMET ML SETUP
# =============================================================================

def setup_comet_ml(config: Dict[str, Any]) -> None:
    """
    Setup Comet ML experiment tracking with full configuration support.
    
    Args:
        config: Configuration dictionary
    """
    if config["report_to"] == "comet_ml":
        try:
            import comet_ml
            
            # Use external config if available
            if COMET_AVAILABLE:
                if not validate_comet_config():
                    print("Falling back to tensorboard logging...")
                    config["report_to"] = "tensorboard"
                    return None
                
                setup_comet_environment()
                comet_settings = COMET_CONFIG
            else:
                # Use config from main CONFIG
                comet_settings = {
                    "workspace": config.get("comet_workspace"),
                    "project": config.get("comet_project", "gemma3n-math-tutor"),
                    "auto_metric_logging": True,
                    "auto_param_logging": True,
                    "auto_histogram_weight_logging": True,
                    "auto_histogram_gradient_logging": True,
                    "auto_histogram_activation_logging": False,
                    "tags": ["gemma3n", "vision-language", "math-tutor", "vietnamese"]
                }
            
            # Initialize Comet experiment
            experiment_kwargs = {
                "workspace": comet_settings.get("workspace"),
                "project_name": comet_settings.get("project"),
                "auto_metric_logging": comet_settings.get("auto_metric_logging", True),
                "auto_param_logging": comet_settings.get("auto_param_logging", True),
                "auto_histogram_weight_logging": comet_settings.get("auto_histogram_weight_logging", True),
                "auto_histogram_gradient_logging": comet_settings.get("auto_histogram_gradient_logging", True),
                "auto_histogram_activation_logging": comet_settings.get("auto_histogram_activation_logging", False),
            }
            
            # Remove None values
            experiment_kwargs = {k: v for k, v in experiment_kwargs.items() if v is not None}
            
            experiment = comet_ml.Experiment(**experiment_kwargs)
            
            # Log configuration
            experiment.log_parameters(config)
            
            # Add tags
            tags = comet_settings.get("tags", ["gemma3n", "vision-language", "math-tutor"])
            for tag in tags:
                experiment.add_tag(tag)
            
            # Log additional metadata
            experiment.log_other("dataset", "ngohongthai/exam-sixth_grade-instruct-dataset")
            experiment.log_other("model_base", "unsloth/gemma-3n-E4B")
            experiment.log_other("task", "sixth-grade-math-tutoring")
            experiment.log_other("language", "vietnamese")
            
            print(f"✅ Comet ML experiment initialized")
            print(f"🔗 Experiment URL: {experiment.url}")
            print(f"📊 Workspace: {comet_settings.get('workspace', 'default')}")
            print(f"📁 Project: {comet_settings.get('project', 'gemma3n-math-tutor')}")
            
            # Set environment variables for transformers integration
            os.environ["COMET_PROJECT_NAME"] = comet_settings.get("project", "gemma3n-math-tutor")
            if comet_settings.get("workspace"):
                os.environ["COMET_WORKSPACE"] = comet_settings["workspace"]
            
            return experiment
            
        except ImportError:
            print("❌ comet_ml not installed. Please install with: pip install comet-ml")
            print("Falling back to tensorboard logging...")
            config["report_to"] = "tensorboard"
            return None
        except Exception as e:
            print(f"❌ Failed to initialize Comet ML: {e}")
            print("Possible causes:")
            print("- Invalid API key or workspace/project names")
            print("- Network connection issues")
            print("- Missing permissions")
            print("Falling back to tensorboard logging...")
            config["report_to"] = "tensorboard"
            return None
    
    return None

# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def main():
    """Main training function."""
    print("🦥 Starting Gemma3N Math Tutor Fine-tuning")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    # Setup Comet ML if specified
    comet_experiment = setup_comet_ml(CONFIG)
    
    try:
        # 1. Setup model and processor
        model, processor = setup_model_and_processor(CONFIG)
        
        # 2. Prepare dataset
        train_dataset = prepare_dataset(CONFIG["dataset_name"], CONFIG["train_split"])
        
        # Print dataset statistics
        print(f"\nDataset Statistics:")
        print(f"- Training samples: {len(train_dataset)}")
        
        # Count samples with images
        samples_with_images = 0
        for sample in train_dataset:
            for conv in sample["conversations"]:
                for content in conv.get("content", []):
                    if content.get("type") == "image":
                        samples_with_images += 1
                        break
                else:
                    continue
                break
        
        print(f"- Samples with images: {samples_with_images}")
        print(f"- Text-only samples: {len(train_dataset) - samples_with_images}")
        
        # 3. Test data collator first (no filtering needed - hybrid collator handles all types)
        if not test_data_collator(train_dataset, processor, num_samples=3):
            print("❌ Data collator test failed, cannot proceed with training")
            return None
        
        # 4. Create trainer
        trainer = create_trainer(model, processor, train_dataset, CONFIG)
        
        # 5. Start training
        print(f"\n🚀 Starting training...")
        print(f"- Output directory: {CONFIG['output_dir']}")
        print(f"- Max steps: {CONFIG['max_steps']}")
        print(f"- Batch size: {CONFIG['per_device_train_batch_size']}")
        print(f"- Gradient accumulation: {CONFIG['gradient_accumulation_steps']}")
        print(f"- Effective batch size: {CONFIG['per_device_train_batch_size'] * CONFIG['gradient_accumulation_steps']}")
        
        # Train the model
        trainer_stats = trainer.train()
        
        # 6. Save final model
        print("\n💾 Saving final model...")
        trainer.save_model()
        processor.save_pretrained(CONFIG["output_dir"])
        
        # Log final metrics to Comet ML
        if comet_experiment:
            try:
                # Log final training stats
                if trainer_stats and trainer_stats.log_history:
                    final_loss = trainer_stats.log_history[-1].get('train_loss')
                    if final_loss:
                        comet_experiment.log_metric("final_train_loss", final_loss)
                
                # Log model artifacts
                comet_experiment.log_model(
                    name="gemma3n-math-tutor",
                    file_or_folder=CONFIG["output_dir"],
                    metadata={
                        "model_type": "gemma3n-vision",
                        "task": "math-tutoring",
                        "language": "vietnamese",
                        "dataset": "sixth-grade-math"
                    }
                )
                
                print(f"📊 Metrics and model logged to Comet ML")
                comet_experiment.end()
                
            except Exception as e:
                print(f"Warning: Failed to log final metrics to Comet ML: {e}")
        
        print(f"\n✅ Training completed successfully!")
        print(f"📁 Model saved to: {CONFIG['output_dir']}")
        
        return trainer_stats
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise e

# =============================================================================
# INFERENCE TESTING
# =============================================================================

def test_inference(model_path: str = None):
    """
    Test the fine-tuned model with inference.
    
    Args:
        model_path: Path to the fine-tuned model (optional)
    """
    if model_path is None:
        model_path = CONFIG["output_dir"]
    
    print(f"\n🧪 Testing inference from: {model_path}")
    
    # Load fine-tuned model
    model, processor = FastVisionModel.from_pretrained(
        model_path,
        max_seq_length=CONFIG["max_seq_length"],
        load_in_4bit=CONFIG["load_in_4bit"],
    )
    
    # Enable inference mode
    FastVisionModel.for_inference(model)
    
    # Test sample
    test_question = "Tính 25 × 4 + 15 × 4"
    
    # Create conversation
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": test_question}]
        }
    ]
    
    # Generate response
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=True,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
    
    response = processor.decode(outputs[0], skip_special_tokens=True)
    print(f"\n📝 Question: {test_question}")
    print(f"🤖 Response: {response}")

if __name__ == "__main__":
    # Run training
    trainer_stats = main()
    
    # Test inference (optional)
    # test_inference()