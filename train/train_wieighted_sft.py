#!/usr/bin/env python3
"""
Weighted SFT training based on quality scores
"""
from unsloth import FastLanguageModel

import pandas as pd
import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
import json
import numpy as np

class WeightedSFTTrainer(SFTTrainer):
    def __init__(self, quality_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.quality_weights = quality_weights
        self._global_step_count = 0
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Get standard loss
        outputs = model(**inputs)
        loss = outputs.loss
        
        # Apply quality weights if available
        if self.quality_weights is not None and hasattr(self, 'state'):
            # Calculate current batch start index based on global step and batch size
            batch_size = inputs['input_ids'].size(0)
            total_batch_size = batch_size * self.args.gradient_accumulation_steps
            current_batch_start = self.state.global_step * total_batch_size
            
            # Get weights for current batch
            end_idx = min(current_batch_start + batch_size, len(self.quality_weights))
            if current_batch_start < len(self.quality_weights):
                weights = self.quality_weights[current_batch_start:end_idx]
                if weights:
                    # Pad weights if needed
                    while len(weights) < batch_size:
                        weights.append(weights[-1])
                    
                    weight_tensor = torch.tensor(weights[:batch_size], device=loss.device, dtype=loss.dtype)
                    # Apply weights to loss
                    if loss.dim() == 0:  # scalar loss
                        loss = loss * weight_tensor.mean()
                    else:  # per-example loss
                        loss = (loss * weight_tensor).mean()
        
        return (loss, outputs) if return_outputs else loss

def prepare_weighted_dataset(input_file: str):
    """Prepare dataset with quality weights"""
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    training_data = []
    weights = []
    
    for item in data:
        content = item['content']
        score = item.get('quality_score', 5)
        
        # Create training format
        formatted_text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an AI trained to write in Lex Fridman's distinctive voice and style. Write authentic, thoughtful posts that reflect his philosophical nature, optimism about technology and humanity, and personal reflection style.<|eot_id|><|start_header_id|>user<|end_header_id|>

Write a social media post in Lex Fridman's distinctive style.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{content}<|eot_id|>"""

        training_data.append({"text": formatted_text})
        
        # Quality weight: normalize scores 5-10 to weights 0.1-1.0
        weight = max(0.1, (score - 4) / 6)  # score 5 = 0.17, score 10 = 1.0
        weights.append(weight)
    
    return training_data, weights

def train_weighted_model():
    # Load and prepare data
    training_data, weights = prepare_weighted_dataset("../data/lex_posts_5plus.json")
    
    print(f"Training examples: {len(training_data)}")
    print(f"Weight range: {min(weights):.2f} - {max(weights):.2f}")
    print(f"Average weight: {np.mean(weights):.2f}")
    
    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/llama-3.2-3b-instruct-bnb-4bit",
        max_seq_length=512,
        dtype=None,
        load_in_4bit=True,
    )
    
    # Setup LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
    )
    
    # Create dataset
    dataset = Dataset.from_list(training_data)
    
    # Training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        num_train_epochs=1,
        learning_rate=8e-5,
        fp16=False,
        bf16=True,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.1,
        output_dir="../models/lex-weighted/lex-weighted-model",
        save_steps=100,
        save_total_limit=3,
    )
    
    # Setup weighted trainer
    trainer = WeightedSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=512,
        quality_weights=weights,
        args=training_args,
    )
    
    # Train
    trainer.train()
    
    # Save
    model.save_pretrained("../models/lex-weighted/lex-weighted-model")
    tokenizer.save_pretrained("../models/lex-weighted/lex-weighted-model")
    
    # Save merged
    FastLanguageModel.for_inference(model)
    model.save_pretrained_merged("../models/lex-weighted/lex-weighted-model_merged", tokenizer, save_method="merged_16bit")
    
    print("Weighted training complete!")

if __name__ == "__main__":
    train_weighted_model()