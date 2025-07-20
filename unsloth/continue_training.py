#!/usr/bin/env python3
"""
Continue Training cho Gemma 3 4B Model
"""

import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
import os

def main():
    print("🔄 Continue Training Gemma 3 4B")
    print("=" * 50)
    
    # Load existing model
    print("📥 Loading existing model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        "./gemma_multimodal_finetuned_merged",
        max_seq_length=512,
        dtype=None,
        load_in_4bit=True,
    )
    
    # Setup LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=4,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=8,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    
    print("✅ Model loaded with LoRA!")
    
    # Handle tokenizer
    if hasattr(tokenizer, 'tokenizer'):
        actual_tokenizer = tokenizer.tokenizer
    else:
        actual_tokenizer = tokenizer
    
    # Load more data
    print("📊 Loading dataset...")
    dataset = load_dataset("ngohongthai/dethivaolop6-montoan", split="train[:200]", trust_remote_code=True)
    
    # Format data with proper field detection
    def format_data(examples):
        texts = []
        
        # Debug: check available fields
        print(f"🔍 Available fields: {list(examples.keys())}")
        
        # Get the main text field (could be 'text', 'question', 'problem')
        text_field = None
        for field in ['text', 'question', 'problem']:
            if field in examples:
                text_field = field
                break
        
        if not text_field:
            print("❌ No text field found!")
            return {"text": ["<start_of_turn>user\nSample question<end_of_turn>\n<start_of_turn>model\nSample answer<end_of_turn>"]}
        
        # Get solution and result fields
        solution_field = None
        for field in ['Solution', 'solution', 'rationale']:
            if field in examples:
                solution_field = field
                break
                
        result_field = None
        for field in ['Result', 'result', 'answer']:
            if field in examples:
                result_field = field
                break
        
        # Process each example
        for i in range(len(examples[text_field])):
            try:
                question = str(examples[text_field][i])
                
                solution = ""
                if solution_field and i < len(examples[solution_field]):
                    solution = str(examples[solution_field][i])
                
                result = ""
                if result_field and i < len(examples[result_field]):
                    result = str(examples[result_field][i])
                
                # Create answer
                if solution and result and solution != result:
                    answer = f"Giải: {solution}\nĐáp án: {result}"
                elif result:
                    answer = f"Đáp án: {result}"
                elif solution:
                    answer = solution
                else:
                    answer = "Cần thêm thông tin"
                
                formatted = f"<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n{answer}<end_of_turn>"
                texts.append(formatted)
                
            except Exception as e:
                print(f"⚠️ Error formatting item {i}: {e}")
                texts.append("<start_of_turn>user\nSample question<end_of_turn>\n<start_of_turn>model\nSample answer<end_of_turn>")
        
        return {"text": texts}
    
    # Process dataset
    dataset = dataset.map(format_data, batched=True, batch_size=10, remove_columns=dataset.column_names)
    
    # Filter valid samples
    dataset = dataset.filter(lambda x: len(x['text']) > 50 and len(x['text']) < 1000)
    
    print(f"📊 Dataset size: {len(dataset)} samples")
    
    # Training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=5,
        max_steps=50,  # More steps
        learning_rate=5e-5,  # Lower learning rate
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="./outputs_continue",
        report_to=None,
        save_steps=25,
        save_total_limit=1,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        dataloader_drop_last=True,
        max_grad_norm=1.0,
        eval_strategy="no",
        tf32=True if torch.cuda.is_available() else False,
        auto_find_batch_size=False,
        group_by_length=False,
        logging_first_step=True,
        save_safetensors=True,
        load_best_model_at_end=False,
        remove_unused_columns=True,
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=actual_tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=512,
        args=training_args,
        packing=False,
        dataset_num_proc=1,
    )
    
    print("🚀 Starting continued training...")
    trainer.train()
    
    # Save model
    print("💾 Saving model...")
    trainer.save_model("./gemma_multimodal_finetuned_continued")
    
    print("🎉 Continued training completed!")

if __name__ == "__main__":
    main() 