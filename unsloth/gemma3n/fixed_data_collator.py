#!/usr/bin/env python3
"""
Fixed Data Collator cho Gemma3n - giải quyết vấn đề image token mismatch
"""

import torch
from typing import List, Dict, Any, Optional
from PIL import Image


class FixedGemma3nDataCollator:
    """
    Fixed data collator cho Gemma3n xử lý đúng image tokens.
    
    Approach: Thay vì insert text tokens, ta modify conversation structure 
    để processor tự động handle image tokens correctly.
    """
    
    def __init__(self, processor, handle_text_only=True):
        self.processor = processor
        self.handle_text_only = handle_text_only
        self.placeholder_image = None
        
    def _create_placeholder_image(self):
        """Create a minimal placeholder image."""
        if self.placeholder_image is None:
            # Tạo image nhỏ để minimize memory
            self.placeholder_image = Image.new('RGB', (224, 224), color=(245, 245, 245))
        return self.placeholder_image
    
    def _extract_images_from_conversation(self, conv):
        """Extract all images from conversation."""
        images = []
        for message in conv:
            for content in message.get("content", []):
                if content.get("type") == "image" and "image" in content:
                    img = content["image"]
                    if img is not None and hasattr(img, 'convert'):
                        images.append(img.convert('RGB'))
        return images
    
    def _has_real_images(self, conv):
        """Check if conversation has real images."""
        return len(self._extract_images_from_conversation(conv)) > 0
    
    def _create_conversation_with_placeholder(self, conv):
        """
        Create conversation với placeholder image và ensure image token insertion.
        
        HYBRID APPROACH: Modify conversation structure + manual token insertion
        vì processor không tự động tạo tokens cho placeholder images.
        """
        modified_conv = []
        placeholder_added = False
        
        for message in conv:
            if message["role"] == "user" and not placeholder_added:
                # Add placeholder image vào user message đầu tiên
                new_content = [{"type": "image", "image": self._create_placeholder_image()}]
                
                # Add existing content
                for content in message.get("content", []):
                    new_content.append(content)
                
                modified_message = {
                    "role": "user",
                    "content": new_content
                }
                modified_conv.append(modified_message)
                placeholder_added = True
            else:
                modified_conv.append(message)
        
        return modified_conv
    
    def _force_image_token_insertion(self, text, num_tokens=1):
        """
        Force insert CORRECT image tokens vào text.
        
        Gemma3n sử dụng <image_soft_token>, KHÔNG phải <image>!
        """
        # Check for existing image tokens
        if any(token in text for token in ['<image_soft_token>', '<start_of_image>', '<end_of_image>']):
            return text
            
        lines = text.split('\n')
        
        # Use correct Gemma3n image token
        image_token = '<image_soft_token>'
        
        # Strategy 1: Insert after potential BOS token  
        if len(lines) > 0 and any(token in lines[0] for token in ['<bos>', '<s>', '<|']):
            # Insert sau line đầu tiên
            for i in range(num_tokens):
                lines.insert(1, image_token)
            return '\n'.join(lines)
        
        # Strategy 2: Insert at beginning
        image_tokens = [image_token] * num_tokens
        return '\n'.join(image_tokens) + '\n' + text
    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Main collate function với new approach.
        """
        print(f"🔧 Fixed collator processing {len(examples)} examples...")
        
        texts = []
        images_list = []
        
        for idx, example in enumerate(examples):
            conv = example["conversations"]
            
            if self._has_real_images(conv):
                # Has real images - process normally
                print(f"Sample {idx}: HAS REAL IMAGES ✅")
                images = self._extract_images_from_conversation(conv)
                text = self.processor.apply_chat_template(
                    conv, tokenize=False, add_generation_prompt=False
                )
                
            elif self.handle_text_only:
                # Text-only - hybrid approach: conversation modification + manual token insertion
                print(f"Sample {idx}: TEXT ONLY - hybrid approach ⚡")
                
                # Step 1: Modify conversation structure  
                conv_with_placeholder = self._create_conversation_with_placeholder(conv)
                images = [self._create_placeholder_image()]
                
                # Step 2: Apply template
                text = self.processor.apply_chat_template(
                    conv_with_placeholder, tokenize=False, add_generation_prompt=False
                )
                
                # Step 3: Force token insertion if processor didn't add them
                if '<image_soft_token>' not in text:
                    print(f"  🔧 Processor didn't add <image_soft_token>, forcing insertion...")
                    text = self._force_image_token_insertion(text, num_tokens=1)
                
            else:
                # Skip text-only samples
                print(f"Sample {idx}: TEXT ONLY - SKIPPING")
                continue
            
            texts.append(text)
            images_list.append(images)
            
            # Debug info - check for correct Gemma3n tokens
            image_token_count = text.count('<image_soft_token>')
            print(f"  → {len(images)} images, {image_token_count} <image_soft_token> tokens")
            if image_token_count != len(images):
                print(f"  ⚠️  MISMATCH DETECTED!")
        
        if not texts:
            raise ValueError("No valid samples after processing!")
        
        # Final validation
        print("🔍 Final validation:")
        total_images = sum(len(imgs) for imgs in images_list)
        total_tokens = sum(text.count('<image_soft_token>') for text in texts)
        print(f"  Total: {total_images} images, {total_tokens} <image_soft_token> tokens")
        
        if total_tokens != total_images:
            print(f"🚨 FINAL MISMATCH DETECTED! Applying last resort fixes...")
            # Last resort fixes
            for i, (text, imgs) in enumerate(zip(texts, images_list)):
                token_count = text.count('<image_soft_token>')
                image_count = len(imgs)
                
                if token_count != image_count:
                    print(f"🚨 Sample {i}: {token_count} <image_soft_token> vs {image_count} images")
                    
                    if token_count == 0 and image_count > 0:
                        # Critical: No tokens but have images
                        print(f"  💥 Emergency: Force adding {image_count} <image_soft_token>")
                        texts[i] = self._force_image_token_insertion(text, num_tokens=image_count)
                    elif token_count > image_count:
                        # Too many tokens
                        print(f"  ⚠️  Truncating images to {token_count}")
                        images_list[i] = imgs[:token_count]
                    elif image_count > token_count and token_count > 0:
                        # Too many images
                        print(f"  ⚠️  Truncating images to {token_count}")
                        images_list[i] = imgs[:token_count]
        
        # Process with processor
        print("📤 Sending to processor...")
        try:
            batch = self.processor(
                text=texts,
                images=images_list,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )
        except Exception as e:
            print(f"🚨 PROCESSOR ERROR: {e}")
            # Additional debug info
            for i, (text, imgs) in enumerate(zip(texts, images_list)):
                print(f"Sample {i}: {text.count('<image_soft_token>')} <image_soft_token>, {len(imgs)} images")
                print(f"  Text preview: {repr(text[:150])}...")
            raise
        
        # Create labels
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        
        print(f"✅ Batch created: {batch.keys()}")
        if "pixel_values" in batch:
            print(f"  pixel_values: {batch['pixel_values'].shape}")
        print(f"  input_ids: {batch['input_ids'].shape}")
        
        return batch