#!/usr/bin/env python3
"""
Advanced LLM-powered dataset curator for Lex Fridman style learning
"""

import json
import time
import os
from openai import OpenAI
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def score_post_quality(content: str) -> dict:
    """Score a post for Lex Fridman style learning value"""
    
    prompt = f"""You are curating training data for an AI that needs to learn Lex Fridman's authentic writing style. The goal is to train a model that can convert user drafts into Lex's distinctive voice.

AUTHENTIC LEX CHARACTERISTICS:
- Personal storytelling: "Just had lunch with...", "Today I was thinking..."
- Philosophical reflections on simple moments
- Genuine gratitude and optimism about humanity
- Deep questions about consciousness, AI, life
- Humble intellectual curiosity
- Simple, profound observations
- Stories about conversations that changed his perspective

VALUABLE FOR TRAINING (Score 7-10):
✓ Personal reflections with philosophical depth
✓ Stories about meaningful conversations
✓ Gratitude and perspective posts
✓ Questions about consciousness, technology, humanity
✓ Simple moments that reveal deeper truths
✓ Authentic vulnerability and growth

HARMFUL FOR TRAINING (Score 1-3):
✗ Podcast/interview promotions ("Here's my conversation with...")
✗ Pure marketing content
✗ Generic motivational quotes
✗ Technical explanations without personal touch
✗ Event announcements
✗ Link sharing without personal commentary

BORDERLINE (Score 4-6):
~ Technical content with some personal insight
~ Brief philosophical thoughts without stories
~ Posts that are "Lex-adjacent" but missing authenticity

Post to evaluate: "{content}"

Consider: Will this post help an AI learn Lex's distinctive voice and philosophical perspective? Can it teach the model how to transform generic content into Lex's authentic style?

Respond with JSON only:
{{
    "score": 1-10,
    "category": "personal_story" | "philosophical" | "gratitude" | "technical_personal" | "promotional" | "generic" | "borderline",
    "reasoning": "specific explanation of why this helps/hurts style learning",
    "keep_for_training": true/false,
    "keep_as_negative": true/false,
    "style_elements": ["list", "of", "specific", "lex", "traits", "present"]
}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.1
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
        
    except Exception as e:
        print(f"Error processing post: {e}")
        return {
            "score": 0,
            "category": "error", 
            "reasoning": f"API error: {str(e)}",
            "keep_for_training": False,
            "keep_as_negative": False,
            "style_elements": []
        }

def curate_dataset(input_file: str, output_file: str, min_positive_score: int = 7):
    """Curate dataset with positive and negative examples"""
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    positive_examples = []
    negative_examples = []
    all_scores = []
    
    print(f"Processing {len(data)} posts...")
    
    for i, post in enumerate(data):
        content = post.get('content', '').strip()
        
        if len(content) < 20:
            continue
            
        score_result = score_post_quality(content)
        score = score_result.get('score', 0)
        
        # Enhanced post data
        enhanced_post = {
            **post,
            'quality_score': score,
            'category': score_result.get('category', 'unknown'),
            'reasoning': score_result.get('reasoning', ''),
            'style_elements': score_result.get('style_elements', []),
            'training_value': 'positive' if score >= min_positive_score else 'negative' if score <= 3 else 'excluded'
        }
        
        # Collect examples
        if score >= min_positive_score and score_result.get('keep_for_training'):
            positive_examples.append(enhanced_post)
        elif score <= 3 and score_result.get('keep_as_negative'):
            negative_examples.append(enhanced_post)
        
        all_scores.append(enhanced_post)
        
        status = "POSITIVE" if score >= min_positive_score else "NEGATIVE" if score <= 3 else "EXCLUDED"
        print(f"{i+1}/{len(data)} - Score: {score} - {status} - {score_result.get('category', '')}")
        
        time.sleep(0.1)  # Rate limiting
    
    # Create training dataset with both positive and select negative examples
    training_dataset = positive_examples + negative_examples[:50]  # Limit negatives
    
    # Save curated training dataset
    with open(output_file, 'w') as f:
        json.dump(training_dataset, f, indent=2, ensure_ascii=False)
    
    # Save full analysis
    analysis_file = output_file.replace('.json', '_analysis.json')
    with open(analysis_file, 'w') as f:
        json.dump({
            'summary': {
                'total_processed': len(data),
                'positive_examples': len(positive_examples),
                'negative_examples': len(negative_examples),
                'training_dataset_size': len(training_dataset),
                'average_positive_score': sum(p['quality_score'] for p in positive_examples) / len(positive_examples) if positive_examples else 0,
                'categories': {}
            },
            'all_scores': all_scores
        }, f, indent=2, ensure_ascii=False)
    
    # Category breakdown
    categories = {}
    for item in all_scores:
        cat = item['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    print(f"\n=== CURATION COMPLETE ===")
    print(f"Original posts: {len(data)}")
    print(f"Positive examples: {len(positive_examples)}")
    print(f"Negative examples: {len(negative_examples)}")
    print(f"Final training dataset: {len(training_dataset)}")
    print(f"Average positive score: {sum(p['quality_score'] for p in positive_examples) / len(positive_examples):.1f}")
    print(f"\nCategories: {categories}")
    
    return training_dataset

def create_training_format(curated_file: str, output_file: str):
    """Convert curated data to training format with quality indicators"""
    
    with open(curated_file, 'r') as f:
        data = json.load(f)
    
    formatted_data = []
    
    for item in data:
        content = item['content']
        score = item['quality_score']
        reasoning = item['reasoning']
        
        if item['training_value'] == 'positive':
            # High quality example
            formatted_item = {
                "content": content,
                "quality_score": score,
                "reasoning": reasoning,
                "style_elements": item['style_elements'],
                "training_type": "positive_example"
            }
        else:
            # Negative example (what not to do)
            formatted_item = {
                "content": content,
                "quality_score": score,
                "reasoning": f"NEGATIVE EXAMPLE - {reasoning}",
                "style_elements": [],
                "training_type": "negative_example"
            }
        
        formatted_data.append(formatted_item)
    
    with open(output_file, 'w') as f:
        json.dump(formatted_data, f, indent=2, ensure_ascii=False)
    
    print(f"Training format saved to {output_file}")

if __name__ == "__main__":
    input_file = "lex_posts.json"
    curated_file = "lex_posts_curated.json"
    training_file = "lex_training_data.json"
    
    # Step 1: Curate dataset
    training_data = curate_dataset(input_file, curated_file, min_positive_score=7)
    
    # Step 2: Create training format
    create_training_format(curated_file, training_file)
    
    print(f"\n=== FILES CREATED ===")
    print(f"Curated dataset: {curated_file}")
    print(f"Analysis: {curated_file.replace('.json', '_analysis.json')}")
    print(f"Training format: {training_file}")