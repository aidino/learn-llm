
# • Data ingestion: Efficiently load and parse large text corpora.
# • Quality assessment: Automatically detect and flag data quality issues.
# • Preprocessing: Apply text cleaning and normalization techniques.
# • Deduplication: Remove exact and near-duplicate content.
# • Filtering: Remove low-quality or irrelevant samples based on predefined criteria.
# • Validation: Ensure the cleaned data meets quality standards.
# • Output: Save the cleaned data in an appropriate format for LLM training.

import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
# Download required NLTK data
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

class DataCleaningPipeline:
    def __init__(self, similarity_threshold=0.9, min_length=10, max_length=1000):
        self.similarity_threshold = similarity_threshold
        self.min_length = min_length
        self.max_length = max_length
        self.vectorizer = TfidfVectorizer(stop_words='english')
    
    def preprocess(self, text):
        # Basic preprocessing
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = [word for word in text.split() if word not in stop_words]
        return ' '.join(tokens)
    
    def filter_by_length(self, df):
        return df[(df['text'].str.len() >= self.min_length) & (df['text'].str.len() <= self.max_length)]
    
    def deduplicate(self, df):
        tfidf_matrix = self.vectorizer.fit_transform(df['text'])
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        duplicates = set()
        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                if similarity_matrix[i, j] > self.similarity_threshold:
                    duplicates.add(j)
        
        return df.drop(df.index[list(duplicates)])
    
    def clean(self, input_file, output_file):
        # Read data
        df = pd.read_csv(input_file)
        
        # Preprocess
        df['text'] = df['text'].apply(self.preprocess)
        
        # Filter by length
        df = self.filter_by_length(df)
        
        # Deduplicate
        df = self.deduplicate(df)
        
        # Save cleaned data
        df.to_csv(output_file, index=False)
        
        print(f"Cleaned data saved to {output_file}")

# Example usage
pipeline = DataCleaningPipeline()
pipeline.clean('input_data.csv', 'cleaned_data.csv')