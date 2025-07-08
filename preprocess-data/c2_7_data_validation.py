
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd


def validate_cleaned_data(file_path, sample_size=100):
    df = pd.read_csv(file_path)
    
    # Basic statistics
    print(f"Total samples: {len(df)}")
    print(f"Average text length: {df['text'].str.len().mean():.2f}")
    print(f"Unique samples: {df['text'].nunique()}")
    
    # Check for empty or very short texts
    short_texts = df[df['text'].str.len() < 10]
    print(f"Texts shorter than 10 characters: {len(short_texts)}")
    
    # Sample for manual review
    sample = df.sample(n=min(sample_size, len(df)))
    print("\nSample for manual review:")
    print(sample['text'].head())

    # Check for common issues
    common_issues = {
        'special_chars': df['text'].str.contains(r'[^a-zA-Z0-9\s]'),
        'numbers': df['text'].str.contains(r'\d'),
        'all_caps': df['text'].str.isupper()
    }
    
    for issue, mask in common_issues.items():
        print(f"Samples with {issue}: {mask.sum()}")
    
    # Evaluate impact on model perplexity
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    def calculate_perplexity(text):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=1024)
        with torch.no_grad():
            outputs = model(inputs, labels=inputs['input_ids'])
        return torch.exp(outputs.loss).item()
    
    sample_perplexities = sample['text'].apply(calculate_perplexity)
    print(f"\nAverage perplexity on sample: {sample_perplexities.mean():.2f}")

# Example usage
validate_cleaned_data('cleaned_data.csv')