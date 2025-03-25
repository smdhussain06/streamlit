import streamlit as st
from model_train import EmotionClassifier

def main():
    st.title('Emotion Recognition from Text')
    st.write('Enter text to analyze its emotional content')
    
    # Initialize the classifier
    classifier = EmotionClassifier()
    
    try:
        # Load the pre-trained model
        classifier.load()
        model_status = "Model loaded successfully!"
    except:
        model_status = "No pre-trained model found. Please train the model first by running main.py"
    
    st.write(model_status)
    
    # Text input
    text_input = st.text_area("Enter your text here:", height=100)
    
    if st.button('Analyze Emotion'):
        if text_input.strip() == "":
            st.warning("Please enter some text to analyze")
        else:
            try:
                # Make prediction
                emotion = classifier.predict([text_input])[0]
                
                # Display results
                st.subheader("Results:")
                st.write(f"Detected emotion: **{emotion}**")
                
                # Add emoji based on emotion
                emoji_dict = {
                    'joy': '😊',
                    'sadness': '😢',
                    'anger': '😠',
                    'fear': '😨',
                    'surprise': '😮'
                }
                if emotion.lower() in emoji_dict:
                    st.write(f"Emotion emoji: {emoji_dict[emotion.lower()]}")
                
            except Exception as e:
                st.error(f"Error analyzing text: {str(e)}")
    
    # Add information about the project
    st.sidebar.header("About")
    st.sidebar.info("""
    This app uses machine learning to detect emotions in text.
    It can recognize five basic emotions:
    - Joy 😊
    - Sadness 😢
    - Anger 😠
    - Fear 😨
    - Surprise 😮
    """)

if __name__ == '__main__':
    main()