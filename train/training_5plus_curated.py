#!/usr/bin/env python3
"""
Training script for 5+ score Lex Fridman dataset
"""
from unsloth import FastLanguageModel

import json
import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer

CONFIG = {
    "model_name": "unsloth/llama-3.2-3b-instruct-bnb-4bit",
    "max_seq_length": 512,
    "data_file": "../data/lex_posts_5plus.json",
    "output_dir": "../models/lex-5plus/lex-5plus-model",
    
    # Training settings
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 10,
    "num_train_epochs": 1,
    "learning_rate": 8e-5,
    "weight_decay": 0.1,
}

def format_training_data(data_file: str):
    """Format 5+ score data for training"""
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    training_data = []
    instruction_templates = [
        "Write a social media post in Lex Fridman's distinctive style.",
        "Create a post that sounds like Lex Fridman would write it.",
        "Write in Lex Fridman's voice and tone.",
        "Generate a thoughtful post in Lex Fridman's style.",
        "Write a post reflecting Lex Fridman's perspective and voice.",
    ]
    
    for i, item in enumerate(data):
        content = item.get('content', '').strip()
        if not content or len(content) < 20:
            continue
        
        instruction = instruction_templates[i % len(instruction_templates)]
        
        formatted_text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an AI trained to write in Lex Fridman's distinctive voice and style. Write authentic, thoughtful posts that reflect his philosophical nature, optimism about technology and humanity, and personal reflection style.<|eot_id|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{content}<|eot_id|>"""

        training_data.append({"text": formatted_text})
    
    print(f"Formatted {len(training_data)} training examples")
    return training_data

def setup_model():
    """Initialize model and tokenizer"""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=CONFIG["model_name"],
        max_seq_length=CONFIG["max_seq_length"],
        dtype=None,
        load_in_4bit=True,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
    )
    
    return model, tokenizer

def train_model():
    """Main training function"""
    print("Loading 5+ score dataset...")
    training_data = format_training_data(CONFIG["data_file"])
    
    if len(training_data) < 50:
        print(f"Warning: Only {len(training_data)} examples. Consider lowering score threshold.")
    
    print("Setting up model...")
    model, tokenizer = setup_model()
    
    dataset = Dataset.from_list(training_data)
    
    training_args = TrainingArguments(
        per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        warmup_steps=CONFIG["warmup_steps"],
        num_train_epochs=CONFIG["num_train_epochs"],
        learning_rate=CONFIG["learning_rate"],
        fp16=False,
        bf16=True,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=CONFIG["weight_decay"],
        lr_scheduler_type="linear",
        output_dir=CONFIG["output_dir"],
        save_steps=100,
        save_total_limit=3,
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=CONFIG["max_seq_length"],
        args=training_args,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving model...")
    model.save_pretrained(CONFIG["output_dir"])
    tokenizer.save_pretrained(CONFIG["output_dir"])
    
    # Save merged model
    FastLanguageModel.for_inference(model)
    model.save_pretrained_merged("../models/lex-5plus/lex-5plus-model_merged", tokenizer, save_method="merged_16bit")
    
    print(f"Training complete! Models saved to {CONFIG['output_dir']}")

if __name__ == "__main__":
    train_model()