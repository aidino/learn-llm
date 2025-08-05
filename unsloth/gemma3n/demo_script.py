#!/usr/bin/env python3
"""
Demo script cho fine-tuning Gemma3N với Unsloth

Script này cung cấp một ví dụ đơn giản về cách sử dụng Unsloth
để fine-tuning model Gemma3N cho vision-language tasks.

Requirements:
- pip install unsloth
- pip install torch torchvision torchaudio
- pip install transformers datasets pillow

Usage:
    python demo_script.py
"""

import torch
from unsloth import FastVisionModel
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import Dataset
from PIL import Image
import numpy as np
import os

def main():
    print("🚀 Bắt đầu demo fine-tuning Gemma3N với Unsloth")
    
    # 1. Load model
    print("📥 Loading Gemma3N model...")
    try:
        model, processor = FastVisionModel.from_pretrained(
            "unsloth/gemma-3n-E4B",
            load_in_4bit=True,
            use_gradient_checkpointing="unsloth",
            max_seq_length=2048,
        )
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # 2. Configure PEFT
    print("⚙️  Configuring PEFT...")
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=32,
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
        target_modules="all-linear",
        modules_to_save=["lm_head", "embed_tokens"],
    )
    
    # 3. Create sample dataset
    print("📊 Creating sample dataset...")
    
    def create_sample_image(color=(255, 0, 0), size=(224, 224)):
        """Create a simple colored image"""
        img_array = np.full((*size, 3), color, dtype=np.uint8)
        return Image.fromarray(img_array)
    
    # Simple dataset with text and images
    sample_data = [
        {
            "text": "<|system|>\nYou are a helpful vision-language assistant.\n<|user|>\nDescribe this red image: <image>\n<|assistant|>\nThis is a red colored image.\n"
        },
        {
            "text": "<|system|>\nYou are a helpful vision-language assistant.\n<|user|>\nWhat color is this image? <image>\n<|assistant|>\nThis image is green in color.\n"
        }
    ]
    
    dataset = Dataset.from_list(sample_data)
    
    # 4. Setup training
    print("🔧 Setting up training...")
    training_args = TrainingArguments(
        output_dir="./demo_output",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        warmup_steps=5,
        max_steps=10,  # Very short for demo
        learning_rate=2e-5,
        logging_steps=1,
        save_steps=5,
        optim="adamw_8bit",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        seed=3407,
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=processor.tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        dataset_num_proc=1,
        packing=False,
        args=training_args,
    )
    
    # 5. Train (short demo)
    print("🏃 Starting demo training...")
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        trainer_stats = trainer.train()
        print("✅ Demo training completed!")
        print(f"Training loss: {trainer_stats.training_loss:.4f}")
        
    except Exception as e:
        print(f"⚠️  Training error: {e}")
        print("This might be due to memory constraints.")
    
    # 6. Save model
    print("💾 Saving model...")
    try:
        os.makedirs("./demo_finetuned", exist_ok=True)
        model.save_pretrained("./demo_finetuned")
        processor.save_pretrained("./demo_finetuned")
        print("✅ Model saved to ./demo_finetuned")
    except Exception as e:
        print(f"⚠️  Save error: {e}")
    
    # 7. Test inference
    print("🧪 Testing inference...")
    try:
        FastVisionModel.for_inference(model)
        
        test_prompt = "<|system|>\nYou are a helpful assistant.\n<|user|>\nHello, how are you?\n<|assistant|>\n"
        
        inputs = processor.tokenizer(
            test_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        response = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("Generated response:")
        print("-" * 50)
        print(response)
        print("-" * 50)
        
    except Exception as e:
        print(f"⚠️  Inference error: {e}")
    
    print("🎉 Demo completed!")
    print("\nNext steps:")
    print("- Replace sample dataset with your real vision-language data")
    print("- Increase max_steps for longer training") 
    print("- Adjust hyperparameters for your specific use case")
    print("- Use the saved model for your applications")

if __name__ == "__main__":
    main()