# #!/usr/bin/env python3
# """
# Lex Fridman Writing Style Fine-tuning Script
# Trains an LLM to mimic Lex Fridman's posting style using Unsloth
# """
# from unsloth import FastLanguageModel

# import pandas as pd
# import torch
# from datasets import Dataset
# from transformers import TrainingArguments
# from trl import SFTTrainer
# import re
# import json
# from typing import List, Dict
# import os

# # Configuration
# CONFIG = {
#     "model_name": "unsloth/llama-3.2-3b-instruct-bnb-4bit",  # or "unsloth/llama-3.1-8b-instruct-bnb-4bit"
#     "max_seq_length": 512,
#     "dtype": None,
#     "load_in_4bit": True,
#     "load_in_8bit": False,
#     # LoRA settings
#     "lora_r": 64,
#     "lora_alpha": 16,
#     "lora_dropout": 0,
#     "bias": "none",
#     "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    
#     # Training settings
#     "per_device_train_batch_size": 4,
#     "gradient_accumulation_steps": 4,
#     "warmup_steps": 10,
#     "num_train_epochs": 3,
#     "learning_rate": 2e-4,
#     "fp16": False,
#     "bf16": True,
#     "logging_steps": 1,
#     "optim": "adamw_8bit",
#     "weight_decay": 0.01,
#     "lr_scheduler_type": "linear",
#     "seed": 42,
    
#     # Data settings
#     "data_file": "../data/lex_posts.json",  # Your JSON file path
#     "output_dir": "../models/lex-style-wo-curation/lex-style-model",
#     "min_post_length": 10,  # Minimum characters for a valid post
#     "max_post_length": 1000,  # Maximum characters to include
# }

# def clean_text(text: str) -> str:
#     """Clean and normalize post text"""
#     if pd.isna(text) or not isinstance(text, str):
#         return ""
    
#     # Remove excessive whitespace
#     text = re.sub(r'\s+', ' ', text).strip()
    
#     # Remove URLs but keep the text flow natural
#     text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
#     # Clean up extra spaces after URL removal
#     text = re.sub(r'\s+', ' ', text).strip()
    
#     # Keep emojis and special characters (they're part of Lex's style)
#     return text

# def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
#     """Load and preprocess the dataset"""
#     print(f"Loading data from {file_path}...")
    
#     try:
#         # Load JSON data
#         with open(file_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
        
#         # Convert to DataFrame
#         if isinstance(data, list):
#             df = pd.DataFrame(data)
#         else:
#             # If it's a single object, wrap in list
#             df = pd.DataFrame([data])
            
#         print(f"Loaded {len(df)} rows")
#         print(f"Columns: {list(df.columns)}")
        
#         # Check if required columns exist
#         required_cols = ['content']
#         missing_cols = [col for col in required_cols if col not in df.columns]
#         if missing_cols:
#             raise ValueError(f"Missing required columns: {missing_cols}")
        
#         # Clean the content column
#         df['cleaned_text'] = df['content'].apply(clean_text)
        
#         # Filter out empty or very short posts
#         initial_count = len(df)
#         df = df[
#             (df['cleaned_text'].str.len() >= CONFIG['min_post_length']) & 
#             (df['cleaned_text'].str.len() <= CONFIG['max_post_length'])
#         ].copy()
        
#         print(f"Filtered from {initial_count} to {len(df)} posts")
#         print(f"Average post length: {df['cleaned_text'].str.len().mean():.1f} characters")
        
#         # Optional: Show date range if available
#         if 'date_posted' in df.columns:
#             try:
#                 df['date_parsed'] = pd.to_datetime(df['date_posted'])
#                 date_range = f"{df['date_parsed'].min()} to {df['date_parsed'].max()}"
#                 print(f"Date range: {date_range}")
#             except:
#                 print("Could not parse dates")
        
#         return df
        
#     except Exception as e:
#         print(f"Error loading data: {e}")
#         raise

# def format_training_data(df: pd.DataFrame) -> List[Dict[str, str]]:
#     """Format data for training with instruction-following format"""
#     training_data = []
    
#     # Since we're training on Lex Fridman's style, use his name
#     author_name = "Lex Fridman"
    
#     for _, row in df.iterrows():
#         post_text = row['cleaned_text']
#         if not post_text:
#             continue
            
#         # Create instruction-following format
#         instruction = f"Write a social media post in {author_name}'s distinctive style."
        
#         # Format as conversation
#         formatted_text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

# You are an AI trained to write in {author_name}'s distinctive voice and style. Write authentic, thoughtful posts that reflect his philosophical nature, optimism about technology and humanity, and personal reflection style.<|eot_id|><|start_header_id|>user<|end_header_id|>

# {instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

# {post_text}<|eot_id|>"""

#         training_data.append({
#             "text": formatted_text
#         })
    
#     print(f"Formatted {len(training_data)} training examples")
#     return training_data

# def setup_model():
#     """Initialize the model and tokenizer"""
#     print("Loading model...")
    
#     model, tokenizer = FastLanguageModel.from_pretrained(
#         model_name=CONFIG["model_name"],
#         max_seq_length=CONFIG["max_seq_length"],
#         dtype=CONFIG["dtype"],
#         load_in_4bit=CONFIG["load_in_4bit"],
#     )
    
#     # Setup LoRA
#     model = FastLanguageModel.get_peft_model(
#         model,
#         r=CONFIG["lora_r"],
#         target_modules=CONFIG["target_modules"],
#         lora_alpha=CONFIG["lora_alpha"],
#         lora_dropout=CONFIG["lora_dropout"],
#         bias=CONFIG["bias"],
#         use_gradient_checkpointing="unsloth",
#         random_state=CONFIG["seed"],
#     )
    
#     print("Model loaded and LoRA configured")
#     return model, tokenizer

# def train_model(model, tokenizer, training_data: List[Dict[str, str]]):
#     """Train the model"""
#     print("Starting training...")
    
#     # Convert to dataset
#     dataset = Dataset.from_list(training_data)
    
#     # Training arguments
#     training_args = TrainingArguments(
#         per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
#         gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
#         warmup_steps=CONFIG["warmup_steps"],
#         num_train_epochs=CONFIG["num_train_epochs"],
#         learning_rate=CONFIG["learning_rate"],
#         fp16=CONFIG["fp16"],
#         logging_steps=CONFIG["logging_steps"],
#         optim=CONFIG["optim"],
#         weight_decay=CONFIG["weight_decay"],
#         lr_scheduler_type=CONFIG["lr_scheduler_type"],
#         seed=CONFIG["seed"],
#         output_dir=CONFIG["output_dir"],
#         report_to=None,  # Disable wandb
#         save_steps=100,
#         save_total_limit=3,
#         dataloader_num_workers=2,
#     )
    
#     # Setup trainer
#     trainer = SFTTrainer(
#         model=model,
#         tokenizer=tokenizer,
#         train_dataset=dataset,
#         dataset_text_field="text",
#         max_seq_length=CONFIG["max_seq_length"],
#         dataset_num_proc=2,
#         packing=False,
#         args=training_args,
#     )
    
#     # Train
#     trainer.train()
    
#     print("Training completed!")
#     return trainer

# def save_model(model, tokenizer, output_dir: str):
#     """Save the trained model"""
#     print(f"Saving model to {output_dir}...")
    
#     # Save using Unsloth's fast save
#     model.save_pretrained(output_dir)
#     tokenizer.save_pretrained(output_dir)
    
#     # Also save merged model for easier deployment
#     merged_output_dir = f"{output_dir}_merged"
#     print(f"Saving merged model to {merged_output_dir}...")
    
#     # Merge and save
#     FastLanguageModel.for_inference(model)
#     model.save_pretrained_merged(merged_output_dir, tokenizer, save_method="merged_16bit")
    
#     print("Model saved successfully!")

# def test_model(model, tokenizer):
#     """Quick test of the trained model"""
#     print("\nTesting the model...")
    
#     FastLanguageModel.for_inference(model)
    
#     test_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

# You are an AI trained to write in Lex Fridman's distinctive voice and style. Write authentic, thoughtful posts that reflect his philosophical nature, optimism about technology and humanity, and personal reflection style.<|eot_id|><|start_header_id|>user<|end_header_id|>

# Write a social media post in Lex Fridman's distinctive style about the beauty of human connection.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

# """
    
#     inputs = tokenizer([test_prompt], return_tensors="pt").to("cuda")
    
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=150,
#             use_cache=True,
#             temperature=0.7,
#             do_sample=True,
#             top_p=0.9,
#         )
    
#     generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     response = generated_text.split("assistant<|end_header_id|>")[-1].strip()
    
#     print(f"Generated post:\n{response}\n")

# def main():
#     """Main training pipeline"""
#     try:
#         # Check if CUDA is available
#         if not torch.cuda.is_available():
#             print("Warning: CUDA not available. Training will be very slow.")
#         else:
#             print(f"Using GPU: {torch.cuda.get_device_name()}")
#             print(f"VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
#         # Load and preprocess data
#         if not os.path.exists(CONFIG["data_file"]):
#             print(f"Error: Data file {CONFIG['data_file']} not found!")
#             print("Please place your JSON file in the same directory as this script.")
#             return
            
#         df = load_and_preprocess_data(CONFIG["data_file"])
        
#         if len(df) == 0:
#             print("No valid training data found!")
#             return
            
#         # Format training data
#         training_data = format_training_data(df)
        
#         if len(training_data) == 0:
#             print("No formatted training data created!")
#             return
        
#         # Setup model
#         model, tokenizer = setup_model()
        
#         # Train
#         trainer = train_model(model, tokenizer, training_data)
        
#         # Save
#         save_model(model, tokenizer, CONFIG["output_dir"])
        
#         # Test
#         test_model(model, tokenizer)
        
#         print(f"""
# Training completed successfully!

# Model saved to: {CONFIG['output_dir']}
# Merged model saved to: {CONFIG['output_dir']}_merged

# To use the model:
# 1. Load the merged model for inference
# 2. Use the same prompt format as shown in the test
# 3. Adjust temperature (0.7-1.0) for creativity vs consistency
#         """)
        
#     except Exception as e:
#         print(f"Training failed: {e}")
#         raise

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
"""
Lex Fridman Writing Style Fine-tuning Script
Trains an LLM to mimic Lex Fridman's posting style using Unsloth
"""
from unsloth import FastLanguageModel

import pandas as pd
import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
import re
import json
from typing import List, Dict
import os

# Configuration
CONFIG = {
    "model_name": "unsloth/llama-3.2-3b-instruct-bnb-4bit",  # or "unsloth/llama-3.1-8b-instruct-bnb-4bit"
    "max_seq_length": 512,
    "dtype": None,
    "load_in_4bit": True,
    "load_in_8bit": False,
    # LoRA settings
    "lora_r": 64,
    "lora_alpha": 16,
    "lora_dropout": 0,
    "bias": "none",
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    
    # Training settings
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 10,
    "num_train_epochs": 1,  # What worked with 340 examples
    "learning_rate": 5e-5,  # What worked with 340 examples
    "fp16": False,
    "bf16": True,
    "logging_steps": 1,
    "optim": "adamw_8bit",
    "weight_decay": 0.1,  # What worked with 340 examples
    "lr_scheduler_type": "linear",
    "seed": 42,
    
    # Data settings
    "data_file": "../data/lex_posts.json",  # Your JSON file path
    "output_dir": "../models/lex-style-wo-curation/lex-style-model",
    "min_post_length": 10,  # Minimum characters for a valid post
    "max_post_length": 1000,  # Maximum characters to include
}

def clean_text(text: str) -> str:
    """Clean and normalize post text"""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove URLs but keep the text flow natural
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Clean up extra spaces after URL removal
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Keep emojis and special characters (they're part of Lex's style)
    return text

def is_podcast_post(text: str) -> bool:
    """Check if post is a podcast promotion"""
    if pd.isna(text) or not isinstance(text, str):
        return False
    
    text_lower = text.lower()
    
    # Podcast-related keywords that indicate promotional posts
    podcast_keywords = [
    'podcast', 'conversation with', 'here is my conversation', 
    'Here\'s my conversation', 'here\'s my conversation with',
    'watch the full conversation', 'listen to', 'episode', 
    'guest', 'interview', 'check out', 'new episode', 'here\'s my conversation', 'here is my conversation', 'here s my conversation',  'my conversation with'
]
    
    # Check if any podcast keywords are present
    for keyword in podcast_keywords:
        if keyword in text_lower:
            return True
    
    return False

def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """Load and preprocess the dataset"""
    print(f"Loading data from {file_path}...")
    
    try:
        # Load JSON data
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to DataFrame
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            # If it's a single object, wrap in list
            df = pd.DataFrame([data])
            
        print(f"Loaded {len(df)} rows")
        print(f"Columns: {list(df.columns)}")
        
        # Check if required columns exist
        required_cols = ['content']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Clean the content column
        df['cleaned_text'] = df['content'].apply(clean_text)
        
        # Filter out podcast promotion posts
        initial_count = len(df)
        df['is_podcast'] = df['content'].apply(is_podcast_post)
        df = df[~df['is_podcast']].copy()
        
        podcast_filtered_count = len(df)
        print(f"Filtered out {initial_count - podcast_filtered_count} podcast promotion posts")
        
        # Filter out empty or very short posts
        df = df[
            (df['cleaned_text'].str.len() >= CONFIG['min_post_length']) & 
            (df['cleaned_text'].str.len() <= CONFIG['max_post_length'])
        ].copy()
        
        final_count = len(df)
        print(f"Final dataset: {initial_count} → {podcast_filtered_count} → {final_count} posts")
        print(f"Average post length: {df['cleaned_text'].str.len().mean():.1f} characters")
        
        # Optional: Show date range if available
        if 'date_posted' in df.columns:
            try:
                df['date_parsed'] = pd.to_datetime(df['date_posted'])
                date_range = f"{df['date_parsed'].min()} to {df['date_parsed'].max()}"
                print(f"Date range: {date_range}")
            except:
                print("Could not parse dates")
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def format_training_data(df: pd.DataFrame) -> List[Dict[str, str]]:
    """Format data for training with instruction-following format"""
    training_data = []
    
    # Since we're training on Lex Fridman's style, use his name
    author_name = "Lex Fridman"
    
    # Variety of instruction prompts to prevent memorization
    instruction_templates = [
        f"Write a social media post in {author_name}'s distinctive style.",
        f"Create a post that sounds like {author_name} would write it.",
        f"Write in {author_name}'s voice and tone.",
        f"Generate a thoughtful post in {author_name}'s style.",
        f"Write a post reflecting {author_name}'s perspective and voice.",
    ]
    
    for i, row in df.iterrows():
        post_text = row['cleaned_text']
        if not post_text:
            continue
            
        # Rotate through different instruction templates
        instruction = instruction_templates[i % len(instruction_templates)]
        
        # Format as conversation
        formatted_text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an AI trained to write in {author_name}'s distinctive voice and style. Write authentic, thoughtful posts that reflect his philosophical nature, optimism about technology and humanity, and personal reflection style.<|eot_id|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{post_text}<|eot_id|>"""

        training_data.append({
            "text": formatted_text
        })
    
    print(f"Formatted {len(training_data)} training examples")
    return training_data

def setup_model():
    """Initialize the model and tokenizer"""
    print("Loading model...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=CONFIG["model_name"],
        max_seq_length=CONFIG["max_seq_length"],
        dtype=CONFIG["dtype"],
        load_in_4bit=CONFIG["load_in_4bit"],
    )
    
    # Setup LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=CONFIG["lora_r"],
        target_modules=CONFIG["target_modules"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"],
        bias=CONFIG["bias"],
        use_gradient_checkpointing="unsloth",
        random_state=CONFIG["seed"],
    )
    
    print("Model loaded and LoRA configured")
    return model, tokenizer

def train_model(model, tokenizer, training_data: List[Dict[str, str]]):
    """Train the model"""
    print("Starting training...")
    
    # Convert to dataset
    dataset = Dataset.from_list(training_data)
    
    # Training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        warmup_steps=CONFIG["warmup_steps"],
        num_train_epochs=CONFIG["num_train_epochs"],
        learning_rate=CONFIG["learning_rate"],
        fp16=CONFIG["fp16"],
        bf16=CONFIG["bf16"],
        logging_steps=CONFIG["logging_steps"],
        optim=CONFIG["optim"],
        weight_decay=CONFIG["weight_decay"],
        lr_scheduler_type=CONFIG["lr_scheduler_type"],
        seed=CONFIG["seed"],
        output_dir=CONFIG["output_dir"],
        report_to=None,  # Disable wandb
        save_steps=100,
        save_total_limit=3,
        dataloader_num_workers=2,
    )
    
    # Setup trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=CONFIG["max_seq_length"],
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )
    
    # Train
    trainer.train()
    
    print("Training completed!")
    return trainer

def save_model(model, tokenizer, output_dir: str):
    """Save the trained model"""
    print(f"Saving model to {output_dir}...")
    
    # Save using Unsloth's fast save
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Also save merged model for easier deployment
    merged_output_dir = "../models/lex-style-wo-curation/lex-style-model_merged"
    print(f"Saving merged model to {merged_output_dir}...")
    
    # Merge and save
    FastLanguageModel.for_inference(model)
    model.save_pretrained_merged(merged_output_dir, tokenizer, save_method="merged_16bit")
    
    print("Model saved successfully!")

def test_model(model, tokenizer):
    """Quick test of the trained model"""
    print("\nTesting the model...")
    
    FastLanguageModel.for_inference(model)
    
    test_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an AI trained to write in Lex Fridman's distinctive voice and style. Write authentic, thoughtful posts that reflect his philosophical nature, optimism about technology and humanity, and personal reflection style.<|eot_id|><|start_header_id|>user<|end_header_id|>

Write a social media post in Lex Fridman's distinctive style about the beauty of human connection.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    inputs = tokenizer([test_prompt], return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            use_cache=True,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated_text.split("assistant<|end_header_id|>")[-1].strip()
    
    print(f"Generated post:\n{response}\n")

def main():
    """Main training pipeline"""
    try:
        # Check if CUDA is available
        if not torch.cuda.is_available():
            print("Warning: CUDA not available. Training will be very slow.")
        else:
            print(f"Using GPU: {torch.cuda.get_device_name()}")
            print(f"VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Load and preprocess data
        if not os.path.exists(CONFIG["data_file"]):
            print(f"Error: Data file {CONFIG['data_file']} not found!")
            print("Please place your JSON file in the same directory as this script.")
            return
            
        df = load_and_preprocess_data(CONFIG["data_file"])
        
        if len(df) == 0:
            print("No valid training data found!")
            return
            
        # Format training data
        training_data = format_training_data(df)
        
        if len(training_data) == 0:
            print("No formatted training data created!")
            return
        
        # Setup model
        model, tokenizer = setup_model()
        
        # Train
        trainer = train_model(model, tokenizer, training_data)
        
        # Save
        save_model(model, tokenizer, CONFIG["output_dir"])
        
        # Test
        test_model(model, tokenizer)
        
        print(f"""
Training completed successfully!

Model saved to: {CONFIG['output_dir']}
Merged model saved to: {CONFIG['output_dir']}_merged

To use the model:
1. Load the merged model for inference
2. Use the same prompt format as shown in the test
3. Adjust temperature (0.7-1.0) for creativity vs consistency
        """)
        
    except Exception as e:
        print(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()