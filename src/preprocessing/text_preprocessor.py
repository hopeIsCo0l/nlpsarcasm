"""
Text preprocessing module for sarcasm detection.
Handles cleaning, tokenization, and feature extraction from headlines.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class TextPreprocessor:
    """
    A comprehensive text preprocessor for sarcasm detection.
    """
    
    def __init__(self, 
                 remove_stopwords: bool = True,
                 lemmatize: bool = True,
                 stem: bool = False,
                 remove_punctuation: bool = True,
                 lowercase: bool = True,
                 remove_numbers: bool = False,
                 min_word_length: int = 2):
        """
        Initialize the text preprocessor.
        
        Args:
            remove_stopwords: Whether to remove stop words
            lemmatize: Whether to lemmatize words
            stem: Whether to stem words (overrides lemmatize if True)
            remove_punctuation: Whether to remove punctuation
            lowercase: Whether to convert to lowercase
            remove_numbers: Whether to remove numbers
            min_word_length: Minimum word length to keep
        """
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize and not stem
        self.stem = stem
        self.remove_punctuation = remove_punctuation
        self.lowercase = lowercase
        self.remove_numbers = remove_numbers
        self.min_word_length = min_word_length
        
        # Initialize NLTK components
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
        self.lemmatizer = WordNetLemmatizer() if lemmatize else None
        self.stemmer = PorterStemmer() if stem else None
        
        # Custom stop words for sarcasm detection
        self.custom_stop_words = {
            'said', 'says', 'according', 'reports', 'report', 'reported',
            'announced', 'announcement', 'breaking', 'news', 'update'
        }
        self.stop_words.update(self.custom_stop_words)
        
        # Sarcasm indicators (words that might indicate sarcasm)
        self.sarcasm_indicators = {
            'obviously', 'clearly', 'apparently', 'supposedly', 'allegedly',
            'surprisingly', 'shockingly', 'amazingly', 'incredibly', 'unbelievably',
            'finally', 'at last', 'wow', 'oh', 'well', 'right', 'sure', 'yeah'
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def clean_text(self, text: str) -> str:
        """
        Clean the input text by removing unwanted characters and normalizing.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase if specified
        if self.lowercase:
            text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove numbers if specified
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Remove punctuation if specified
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize the text into words.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens
        """
        # Use simple word tokenization instead of NLTK's punkt to avoid punkt_tab issues
        return text.split()
    
    def remove_stop_words(self, tokens: List[str]) -> List[str]:
        """
        Remove stop words from tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Tokens with stop words removed
        """
        if not self.remove_stopwords:
            return tokens
        
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def apply_stemming_or_lemmatization(self, tokens: List[str]) -> List[str]:
        """
        Apply stemming or lemmatization to tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Tokens with stemming/lemmatization applied
        """
        if self.stem and self.stemmer:
            return [self.stemmer.stem(token) for token in tokens]
        elif self.lemmatize and self.lemmatizer:
            return [self.lemmatizer.lemmatize(token) for token in tokens]
        else:
            return tokens
    
    def filter_by_length(self, tokens: List[str]) -> List[str]:
        """
        Filter tokens by minimum length.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Tokens with minimum length filter applied
        """
        return [token for token in tokens if len(token) >= self.min_word_length]
    
    def preprocess_text(self, text: str) -> str:
        """
        Complete text preprocessing pipeline.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Preprocessed text as a single string
        """
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(cleaned_text)
        
        # Remove stop words
        tokens = self.remove_stop_words(tokens)
        
        # Apply stemming or lemmatization
        tokens = self.apply_stemming_or_lemmatization(tokens)
        
        # Filter by length
        tokens = self.filter_by_length(tokens)
        
        # Join tokens back into text
        return ' '.join(tokens)
    
    def extract_features(self, text: str) -> Dict[str, Any]:
        """
        Extract various features from the text that might be useful for sarcasm detection.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of extracted features
        """
        original_text = text
        preprocessed_text = self.preprocess_text(text)
        
        # Basic text features
        features = {
            'original_length': len(original_text),
            'preprocessed_length': len(preprocessed_text),
            'word_count': len(original_text.split()),
            'preprocessed_word_count': len(preprocessed_text.split()),
            'avg_word_length': np.mean([len(word) for word in original_text.split()]) if original_text.split() else 0,
            'char_count': len(original_text.replace(' ', '')),
            'sentence_count': len(original_text.split('.')),
        }
        
        # Punctuation features
        features.update({
            'exclamation_count': original_text.count('!'),
            'question_count': original_text.count('?'),
            'quote_count': original_text.count('"') + original_text.count("'"),
            'capital_letter_count': sum(1 for c in original_text if c.isupper()),
            'all_caps_ratio': sum(1 for word in original_text.split() if word.isupper()) / max(len(original_text.split()), 1)
        })
        
        # Sarcasm indicator features
        sarcasm_indicator_count = sum(1 for indicator in self.sarcasm_indicators 
                                    if indicator.lower() in original_text.lower())
        features['sarcasm_indicator_count'] = sarcasm_indicator_count
        features['sarcasm_indicator_ratio'] = sarcasm_indicator_count / max(len(original_text.split()), 1)
        
        # Sentiment-like features
        features.update({
            'has_obviously': 'obviously' in original_text.lower(),
            'has_clearly': 'clearly' in original_text.lower(),
            'has_supposedly': 'supposedly' in original_text.lower(),
            'has_allegedly': 'allegedly' in original_text.lower(),
            'has_finally': 'finally' in original_text.lower(),
            'has_wow': 'wow' in original_text.lower(),
            'has_oh': ' oh ' in original_text.lower(),
            'has_well': ' well ' in original_text.lower(),
        })
        
        return features
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of preprocessed texts
        """
        return [self.preprocess_text(text) for text in texts]
    
    def extract_features_batch(self, texts: List[str]) -> pd.DataFrame:
        """
        Extract features from a batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            DataFrame with extracted features
        """
        features_list = [self.extract_features(text) for text in texts]
        return pd.DataFrame(features_list)
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the preprocessing configuration.
        
        Returns:
            Dictionary with preprocessing settings
        """
        return {
            'remove_stopwords': self.remove_stopwords,
            'lemmatize': self.lemmatize,
            'stem': self.stem,
            'remove_punctuation': self.remove_punctuation,
            'lowercase': self.lowercase,
            'remove_numbers': self.remove_numbers,
            'min_word_length': self.min_word_length,
            'stop_words_count': len(self.stop_words),
            'sarcasm_indicators_count': len(self.sarcasm_indicators)
        } 