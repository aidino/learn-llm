import unicodedata
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    
    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Join tokens back into text
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

# Example usage
raw_text = "This is an EXAMPLE of text preprocessing... It's quite useful!"
cleaned_text = preprocess_text(raw_text)
print(f"Original: {raw_text}")
print(f"Preprocessed: {cleaned_text}")