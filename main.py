try:
    import pandas as pd
    from sklearn.datasets import make_classification
    from model_train import EmotionClassifier
    import os
    import sys
    
    def create_sample_data(n_samples=100):  # Reduced sample size for testing
        """Create sample data for demonstration"""
        print("Generating sample training data...")
        
        # Create simple sample texts and emotions
        texts = [
            "I'm so happy today!",
            "Feeling really down and blue",
            "This makes me so angry!",
            "I'm really worried about tomorrow",
            "Wow, I can't believe what just happened!"
        ] * 20  # Repeat 20 times to get 100 samples
        
        emotions = ['joy', 'sadness', 'anger', 'fear', 'surprise'] * 20
        
        # Create DataFrame
        df = pd.DataFrame({
            'text': texts,
            'emotion': emotions
        })
        print(f"Generated {len(df)} training examples")
        return df
    
    def main():
        print("\n=== Starting Emotion Recognition Model Training ===\n")
        
        print("Step 1: Loading dataset...")
        df = create_sample_data()
        
        print("\nStep 2: Initializing classifier...")
        classifier = EmotionClassifier()
        
        print("\nStep 3: Preparing data and training model...")
        X, y = classifier.prepare_data(df['text'], df['emotion'])
        
        metrics = classifier.train(X, y)
        
        print("\nStep 4: Model Performance Metrics:")
        print("=" * 50)
        print(metrics['classification_report'])
        
        print("\nStep 5: Saving model...")
        os.makedirs('models', exist_ok=True)
        classifier.save()
        print("Model saved successfully!")
        
        print("\nTesting the model with example texts...")
        test_texts = [
            "I'm so excited about this!",
            "This is terrible news...",
            "I can't believe it!"
        ]
        
        predictions = classifier.predict(test_texts)
        for text, emotion in zip(test_texts, predictions):
            print(f"\nText: '{text}'")
            print(f"Predicted emotion: {emotion}")
        
        print("\n=== Training Complete! ===")
        print("You can now run 'streamlit run app.py' to use the model.")

    if __name__ == "__main__":
        main()

except ImportError as e:
    print(f"\nError: Missing required package: {str(e)}")
    print("\nPlease install the required packages using:")
    print("pip install pandas scikit-learn nltk")
    sys.exit(1)
except Exception as e:
    print(f"\nAn error occurred: {str(e)}")
    sys.exit(1)