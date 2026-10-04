import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Get the folder where app.py is located
BASE_DIR = Path(__file__).resolve().parent

# Load the LSTM model
model = load_model(BASE_DIR / "lstm_model.h5")

# Load the tokenizer
with open(BASE_DIR / "tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)


def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]

    if len(token_list) >= max_sequence_len:
        token_list = token_list[-max_sequence_len:]

    token_list = pad_sequences(
        [token_list],
        maxlen=max_sequence_len,
        padding="pre"
    )

    predicted = model.predict(token_list, verbose=0)

    predicted_word_index = np.argmax(predicted, axis=-1)[0]

    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word

    return None


# Streamlit app
st.title("Next Word Prediction with LSTM")

input_text = st.text_input(
    "Enter the sequence of words",
    "To be or not to"
)

if st.button("Predict Next Word"):
    next_word = predict_next_word(
        model,
        tokenizer,
        input_text,
        max_sequence_len=10
    )

    st.write(f"Predicted next word: {next_word}")