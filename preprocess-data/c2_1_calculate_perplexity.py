import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def calculate_perplexity(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return torch.exp(outputs.loss).item()

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Example texts
clean_text = "The quick brown fox jumps over the lazy dog."
noisy_text = "Th3 qu1ck br0wn f0x jumps 0ver th3 l@zy d0g."

# Calculate perplexity
clean_perplexity = calculate_perplexity(model, tokenizer, clean_text)
noisy_perplexity = calculate_perplexity(model, tokenizer, noisy_text)

print(f"Clean text perplexity: {clean_perplexity:.2f}")
print(f"Noisy text perplexity: {noisy_perplexity:.2f}")

# Output:
# Clean text perplexity: 162.47
# Noisy text perplexity: 587.93

# Giá trị perplexity thấp hơn cho thấy mô hình tự tin hơn vào các dự đoán của mình 
# và xem văn bản đó là có khả năng xảy ra cao hơn hoặc “bình thường” hơn