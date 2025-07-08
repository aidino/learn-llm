from langdetect import detect
from unidecode import unidecode
from nltk import word_tokenize
import nltk
# Download required NLTK data
nltk.download('punkt')

def handle_multilingual_text(text):
    # Detect language
    try:
        lang = detect(text)
    except:
        lang = 'unknown'
    
    # Script normalization: Converts text to a consistent script (e.g., transliteration - phiên âm)
    transliterated_text = unidecode(text)
    
    # Tokenize (using NLTK for simplicity, but consider language-specific tokenizers)
    tokens = word_tokenize(transliterated_text)
    
    return {
        'original': text,
        'language': lang,
        'transliterated': transliterated_text,
        'tokens': tokens
    }

# Example usage
texts = [
    "This is English text.",
    "Dies ist deutscher Text.", # German
    "これは日本語のテキストです。", # Japanese
    "This is mixed language text avec un peu de français." # French
]

for text in texts:
    result = handle_multilingual_text(text)
    print(f"Original: {result['original']}")
    print(f"Detected Language: {result['language']}")
    print(f"Transliterated: {result['transliterated']}")
    print(f"Tokens: {result['tokens']}\n")