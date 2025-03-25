from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import joblib
import os

class FeatureExtractor:
    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.label_encoder = LabelEncoder()
        
    def fit_transform(self, texts, labels=None):
        """Fit and transform the text data to TF-IDF features"""
        X = self.vectorizer.fit_transform(texts)
        if labels is not None:
            y = self.label_encoder.fit_transform(labels)
            return X, y
        return X
    
    def transform(self, texts):
        """Transform new text data using fitted vectorizer"""
        return self.vectorizer.transform(texts)
    
    def encode_labels(self, labels):
        """Encode labels to numerical values"""
        return self.label_encoder.transform(labels)
    
    def decode_labels(self, encoded_labels):
        """Decode numerical values back to label names"""
        return self.label_encoder.inverse_transform(encoded_labels)
    
    def save(self, directory='models'):
        """Save the fitted vectorizer and label encoder"""
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self.vectorizer, os.path.join(directory, 'vectorizer.pkl'))
        joblib.dump(self.label_encoder, os.path.join(directory, 'label_encoder.pkl'))
    
    def load(self, directory='models'):
        """Load saved vectorizer and label encoder"""
        self.vectorizer = joblib.load(os.path.join(directory, 'vectorizer.pkl'))
        self.label_encoder = joblib.load(os.path.join(directory, 'label_encoder.pkl'))