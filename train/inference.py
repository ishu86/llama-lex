#!/usr/bin/env python3
"""
Lex Fridman Style Inference Script
Load and test your trained model for generating posts
"""
from unsloth import FastLanguageModel

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

class LexStyleGenerator:
    def __init__(self, model_path: str, use_merged: bool = True):
        """Initialize the generator with trained model"""
        self.model_path = model_path
        self.use_merged = use_merged
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        print(f"Loading model from {self.model_path}...")
        
        if self.use_merged:
            # Load merged model (easier, faster)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        else:
            # Load with Unsloth (if you want to use LoRA weights)
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_path,
                max_seq_length=512,
                dtype=None,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(self.model)
        
        print("Model loaded successfully!")
    
    def create_prompt(self, topic: str, include_context: str = "") -> str:
        """Create a formatted prompt for generation"""
        instruction = f"Write a social media post in Lex Fridman's distinctive style about {topic}."
        if include_context:
            instruction += f" Context: {include_context}"
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an AI trained to write in Lex Fridman's distinctive voice and style. Write authentic, thoughtful posts that reflect his philosophical nature, optimism about technology and humanity, and personal reflection style.<|eot_id|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return prompt
    
    def generate_post(self, 
                     topic: str, 
                     context: str = "",
                     max_length: int = 200,
                     temperature: float = 0.7,
                     top_p: float = 0.9,
                     do_sample: bool = True) -> str:
        """Generate a post in Lex's style"""
        
        prompt = self.create_prompt(topic, context)
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )
        
        # Decode and extract just the response
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the assistant's response
        if "assistant<|end_header_id|>" in generated_text:
            response = generated_text.split("assistant<|end_header_id|>")[-1].strip()
            # Remove any trailing tokens
            if "<|eot_id|>" in response:
                response = response.split("<|eot_id|>")[0].strip()
        else:
            response = generated_text.split(prompt)[-1].strip()
        
        # Clean up any remaining system artifacts
        if response.startswith("system"):
            lines = response.split('\n')
            # Find where actual content starts
            for i, line in enumerate(lines):
                if not any(keyword in line.lower() for keyword in ['system', 'user', 'assistant', 'you are an ai']):
                    response = '\n'.join(lines[i:]).strip()
                    break
        
        return response
    
    def interactive_mode(self):
        """Interactive mode for testing"""
        print("\n🤖 Lex Style Generator - Interactive Mode")
        print("Type 'quit' to exit, 'settings' to adjust parameters")
        print("-" * 50)
        
        # Default settings
        settings = {
            'temperature': 0.6,
            'top_p': 0.9,
            'max_length': 500
        }
        
        while True:
            try:
                user_input = input("\nEnter a topic or prompt: ").strip()
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'settings':
                    self.adjust_settings(settings)
                    continue
                elif not user_input:
                    continue
                
                print("\nGenerating...")
                response = self.generate_post(
                    topic=user_input,
                    temperature=settings['temperature'],
                    top_p=settings['top_p'],
                    max_length=settings['max_length']
                )
                
                print(f"\n📝 Generated Post:\n{response}")
                print(f"\n⚙️  Settings: temp={settings['temperature']}, top_p={settings['top_p']}, max_len={settings['max_length']}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("\nGoodbye! 👋")
    
    def adjust_settings(self, settings):
        """Adjust generation settings"""
        print("\nCurrent settings:")
        for key, value in settings.items():
            print(f"  {key}: {value}")
        
        print("\nPress Enter to keep current value")
        
        try:
            temp = input(f"Temperature (0.1-2.0, current: {settings['temperature']}): ")
            if temp.strip():
                settings['temperature'] = float(temp)
            
            top_p = input(f"Top-p (0.1-1.0, current: {settings['top_p']}): ")
            if top_p.strip():
                settings['top_p'] = float(top_p)
            
            max_len = input(f"Max length (50-500, current: {settings['max_length']}): ")
            if max_len.strip():
                settings['max_length'] = int(max_len)
                
        except ValueError:
            print("Invalid input, keeping current settings")

def run_preset_tests(generator: LexStyleGenerator):
    """Run some preset tests"""
    test_topics = [
        "the beauty of human connection",
        "artificial intelligence and consciousness", 
        "a meaningful conversation you had recently",
        "the future of technology",
        "what you learned from failure",
        "gratitude and perspective"
    ]
    
    print("\n🧪 Running preset tests...")
    print("=" * 60)
    
    for i, topic in enumerate(test_topics, 1):
        print(f"\n{i}. Topic: {topic}")
        print("-" * 40)
        
        response = generator.generate_post(topic)
        print(f"Generated: {response}")
        
        # Wait for user input to continue
        input("\nPress Enter for next test...")

def main():
    parser = argparse.ArgumentParser(description="Test Lex Fridman style generator")
    parser.add_argument("--model_path", "-m", required=True, 
                       help="Path to your trained model (use _merged version)")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Run in interactive mode")
    parser.add_argument("--test", "-t", action="store_true",
                       help="Run preset tests")
    parser.add_argument("--prompt", "-p", type=str,
                       help="Generate single post with this prompt")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = LexStyleGenerator(args.model_path)
    
    if args.interactive:
        generator.interactive_mode()
    elif args.test:
        run_preset_tests(generator)
    elif args.prompt:
        response = generator.generate_post(args.prompt)
        print(f"\nGenerated Post:\n{response}")
    else:
        print("Please specify --interactive, --test, or --prompt")
        print("Example: python inference.py -m ../models/lex-style-wo-curation/lex-style-model_merged --interactive")

if __name__ == "__main__":
    main()