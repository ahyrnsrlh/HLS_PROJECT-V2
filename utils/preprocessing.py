"""
Data preprocessing utilities for SDG classification project
"""

import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk

# Download required NLTK data
def ensure_nltk_data():
    """Ensure all required NLTK data is downloaded."""
    resources = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('corpora/stopwords', 'stopwords')
    ]
    
    for resource_path, download_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            print(f"Downloading NLTK resource: {download_name}")
            nltk.download(download_name)

# Initialize NLTK data
ensure_nltk_data()

class DataPreprocessor:
    def __init__(self):
        self.mlb = MultiLabelBinarizer()
        self.scaler = StandardScaler()
        self.tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and stem
        tokens = [self.stemmer.stem(word) for word in tokens 
                 if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(tokens)
    
    def parse_sdgs(self, sdg_string):
        """Parse SDG labels from string format to list"""
        if pd.isna(sdg_string):
            return []
        
        # Extract SDG numbers using regex
        sdg_matches = re.findall(r'SDG (\d+)', str(sdg_string))
        return [f"SDG_{num}" for num in sdg_matches]
    
    def preprocess_data(self, df):
        """Complete preprocessing pipeline"""
        df_processed = df.copy()
        
        # Handle missing values
        df_processed['Authors'] = df_processed['Authors'].fillna('Unknown')
        df_processed['Title'] = df_processed['Title'].fillna('')
        df_processed['Year'] = df_processed['Year'].fillna(df_processed['Year'].median())
        df_processed['Cited'] = df_processed['Cited'].fillna(0)
        
        # Clean title text
        df_processed['Title_cleaned'] = df_processed['Title'].apply(self.clean_text)
        
        # Parse SDG labels
        df_processed['SDG_labels'] = df_processed['SDGs'].apply(self.parse_sdgs)
        
        # Remove rows with no SDG labels
        df_processed = df_processed[df_processed['SDG_labels'].apply(len) > 0]
        
        return df_processed
    
    def create_feature_matrix(self, df_processed, fit_transform=True):
        """Create feature matrix combining text and numerical features"""
        # Text features using TF-IDF
        if fit_transform:
            tfidf_features = self.tfidf.fit_transform(df_processed['Title_cleaned'])
        else:
            tfidf_features = self.tfidf.transform(df_processed['Title_cleaned'])
        
        # Numerical features
        numerical_features = df_processed[['Year', 'Cited']].values
        
        # Normalize numerical features
        if fit_transform:
            numerical_features = self.scaler.fit_transform(numerical_features)
        else:
            numerical_features = self.scaler.transform(numerical_features)
        
        # Combine features
        from scipy.sparse import hstack, csr_matrix
        combined_features = hstack([tfidf_features, csr_matrix(numerical_features)])
        
        return combined_features
    
    def create_target_matrix(self, df_processed, fit_transform=True):
        """Create multi-label target matrix"""
        if fit_transform:
            y_multilabel = self.mlb.fit_transform(df_processed['SDG_labels'])
        else:
            y_multilabel = self.mlb.transform(df_processed['SDG_labels'])
        
        return y_multilabel
    
    def get_feature_names(self):
        """Get feature names for interpretability"""
        tfidf_names = self.tfidf.get_feature_names_out().tolist()
        numerical_names = ['Year', 'Cited']
        return tfidf_names + numerical_names
    
    def get_target_names(self):
        """Get target class names"""
        return self.mlb.classes_
