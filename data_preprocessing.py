import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """Remove URLs, mentions, hashtags, and special characters"""
        # Convert to lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|\#\w+', '', text)
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def remove_stopwords(self, text):
        """Remove stop words from text"""
        words = word_tokenize(text)
        return ' '.join([word for word in words if word not in self.stop_words])
    
    def lemmatize_text(self, text):
        """Lemmatize text to reduce words to their base form"""
        words = word_tokenize(text)
        return ' '.join([self.lemmatizer.lemmatize(word) for word in words])
    
    def preprocess(self, text):
        """Apply all preprocessing steps"""
        text = self.clean_text(text)
        text = self.remove_stopwords(text)
        text = self.lemmatize_text(text)
        return text