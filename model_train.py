from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import joblib
import os

from data_preprocessing import TextPreprocessor
from feature_engineering import FeatureExtractor

class EmotionClassifier:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.feature_extractor = FeatureExtractor()
        self.model = LogisticRegression(max_iter=1000, multi_class='multinomial')
        
    def prepare_data(self, texts, labels):
        """Preprocess text data and extract features"""
        # Preprocess texts
        processed_texts = [self.preprocessor.preprocess(text) for text in texts]
        # Extract features and encode labels
        X, y = self.feature_extractor.fit_transform(processed_texts, labels)
        return X, y
    
    def train(self, X, y, test_size=0.2, random_state=42):
        """Train the emotion classification model"""
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        
        # Return evaluation metrics
        return {
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'test_data': (X_test, y_test)
        }
    
    def predict(self, texts):
        """Predict emotions for new texts"""
        # Preprocess texts
        processed_texts = [self.preprocessor.preprocess(text) for text in texts]
        # Transform texts to features
        X = self.feature_extractor.transform(processed_texts)
        # Make predictions
        predictions = self.model.predict(X)
        # Decode predictions to emotion labels
        return self.feature_extractor.decode_labels(predictions)
    
    def save(self, directory='models'):
        """Save the trained model and preprocessing components"""
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self.model, os.path.join(directory, 'emotion_classifier.pkl'))
        self.feature_extractor.save(directory)
    
    def load(self, directory='models'):
        """Load the trained model and preprocessing components"""
        self.model = joblib.load(os.path.join(directory, 'emotion_classifier.pkl'))
        self.feature_extractor.load(directory)