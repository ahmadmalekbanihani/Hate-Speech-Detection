import re
import string
import json
import pickle
import numpy as np
import streamlit as st
import tensorflow as tf

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL_PATH = "best_hate_speech_model.keras"
TOKENIZER_PATH = "tokenizer.pkl"
CONFIG_PATH = "model_config.json"

stemmer = PorterStemmer()
stop_words = set(ENGLISH_STOP_WORDS)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    tokens = re.findall(r"[a-z]+", text)
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    return model, tokenizer, config

model, tokenizer, config = load_assets()
MAX_LEN = int(config["max_len"])
CLASS_NAMES = {int(k): v for k, v in config["class_names"].items()}

st.set_page_config(page_title="Hate Speech Detection", page_icon="🛡️")
st.title("🛡️ Hate Speech Detection")
st.write("Enter a sentence or tweet and the deep-learning model will classify it.")

user_text = st.text_area("Enter text:", height=150, placeholder="Type a sentence here...")

if st.button("Predict"):
    if not user_text.strip():
        st.warning("Please enter some text first.")
    else:
        cleaned = clean_text(user_text)
        sequence = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(sequence, maxlen=MAX_LEN, padding="post", truncating="post")
        probabilities = model.predict(padded, verbose=0)[0]
        predicted_class = int(np.argmax(probabilities))
        predicted_label = CLASS_NAMES[predicted_class]
        confidence = float(probabilities[predicted_class])

        st.subheader("Prediction")
        st.success(predicted_label)
        st.write(f"Confidence: {confidence:.2%}")

        st.subheader("Class probabilities")
        for class_id, class_name in CLASS_NAMES.items():
            st.write(f"{class_name}: {float(probabilities[class_id]):.2%}")
