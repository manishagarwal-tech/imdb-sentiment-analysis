import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

#load the model
model = load_model('models/simple_rnn_imdb.keras')

# load IMDB database word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}


# helper function to decoding review,
def review_decode(encoded_review):
    decoded_review = ' '.join([reverse_word_index.get(i-3, '?') for i in encoded_review])
    return decoded_review

#pre-processing the data
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

#prediction function
# def predict_sentiment(review):
#     preprocessed_input = preprocess_text(review)
#     predection = model.predict(preprocessed_input)
#     sentiment = 'Positive' if predection[0][0] > 0.5 else 'Negative'
#     return sentiment, predection[0][0]


# streamlit app
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to classify as negative or positive")

user_input = st.text_area('Movie Review')
if st.button('Classify'):
    processed_input = preprocess_text(user_input)
    prediction = model.predict(processed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    st.write(f"Review sentiment is: {sentiment}")
    st.write(f"Prediction Score is: {prediction[0][0]:.2f}")
else:
    st.write("Please enter a movie review to predict")

