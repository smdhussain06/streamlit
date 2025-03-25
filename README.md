# Emotion Recognition from Social Media Text

A Python-based project that performs emotion recognition on social media text using machine learning. The system preprocesses text data, extracts features using TF-IDF vectorization, and classifies emotions using a machine learning model.

## Features

- Text preprocessing (URL removal, tokenization, lemmatization)
- TF-IDF feature extraction
- Emotion classification using Logistic Regression
- Interactive web interface using Streamlit
- Model persistence and loading
- Sample data generation for testing

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

Run the main script to train and test the model:
```bash
python main.py
```

This will:
- Generate sample data (replace with your own dataset in production)
- Train the emotion classification model
- Save the trained model
- Show example predictions

### Using the Web Interface

Start the Streamlit web interface:
```bash
streamlit run app.py
```

This will open a web browser where you can:
- Enter text to analyze
- View predicted emotions
- Get explanations for predictions

## Project Structure

- `data_preprocessing.py`: Text cleaning and preprocessing functions
- `feature_engineering.py`: TF-IDF vectorization and feature extraction
- `model_train.py`: Machine learning model training and evaluation
- `app.py`: Streamlit web interface
- `main.py`: Example usage and model training
- `requirements.txt`: Project dependencies

## Model Details

The system uses:
- NLTK for text preprocessing
- TF-IDF vectorization for feature extraction
- Logistic Regression for classification
- 5 emotion classes: joy, sadness, anger, fear, surprise

## Future Improvements

- Add support for custom datasets
- Implement more advanced models (LSTM, Transformer)
- Add confidence scores for predictions
- Expand emotion categories
- Add data visualization for model performance

## License

This project is open source and available under the MIT License.