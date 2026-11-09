#!/usr/bin/env python3
"""
Filter dataset for 5+ score examples
"""

import json

def filter_by_score(input_file: str, output_file: str, min_score: int = 5):
    """Filter dataset to include only examples with score >= min_score"""
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Handle analysis file structure
    if isinstance(data, dict) and 'all_scores' in data:
        items = data['all_scores']
    else:
        items = data
    
    # Filter positive examples (5+ score)
    filtered_data = [
        item for item in items 
        if item.get('quality_score', 0) >= min_score
    ]
    
    print(f"Original: {len(items)} examples")
    print(f"Filtered (score {min_score}+): {len(filtered_data)} examples")
    
    # Show score distribution
    scores = [item['quality_score'] for item in filtered_data]
    print(f"Score range: {min(scores)}-{max(scores)}")
    print(f"Average score: {sum(scores)/len(scores):.1f}")
    
    with open(output_file, 'w') as f:
        json.dump(filtered_data, f, indent=2)
    
    return filtered_data

if __name__ == "__main__":
    filter_by_score("../data/lex_posts_curated_analysis.json", "../data/lex_posts_5plus.json", min_score=5)