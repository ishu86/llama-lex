#!/usr/bin/env python3
"""
Helper script to run training from the root directory
"""
import os
import sys
import subprocess

def run_training_script(script_name):
    """Run a training script from the train/ directory"""
    script_path = os.path.join("train", script_name)
    
    if not os.path.exists(script_path):
        print(f"Error: {script_path} not found!")
        return False
    
    print(f"Running {script_name}...")
    print(f"Working directory: {os.getcwd()}")
    print(f"Script path: {script_path}")
    print("-" * 50)
    
    try:
        # Run the script
        result = subprocess.run([sys.executable, script_path], 
                              cwd=os.getcwd(),
                              check=True)
        print(f"\n{script_name} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n{script_name} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"\nError running {script_name}: {e}")
        return False

def main():
    """Main function to show available training scripts"""
    print("Available training scripts:")
    print("1. training_5plus_curated.py - Train on 5+ score curated data")
    print("2. train_wieighted_sft.py - Train with quality-weighted SFT")
    print("3. train_lex_style.py - Train on full dataset without curation")
    print("4. filter_dataset.py - Filter dataset by quality score")
    print("5. inference.py - Run inference on trained models")
    print()
    
    choice = input("Enter script number (1-5) or 'q' to quit: ").strip()
    
    if choice == 'q':
        return
    
    script_map = {
        '1': 'training_5plus_curated.py',
        '2': 'train_wieighted_sft.py', 
        '3': 'train_lex_style.py',
        '4': 'filter_dataset.py',
        '5': 'inference.py'
    }
    
    if choice in script_map:
        script_name = script_map[choice]
        run_training_script(script_name)
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main() 