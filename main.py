"""Streamlit interface for a small, intent-based chatbot."""

from __future__ import annotations

import json
import random
from hashlib import sha256
from pathlib import Path
from time import sleep

import nltk
import numpy as np
import streamlit as st
import tensorflow as tf
from nltk.stem.lancaster import LancasterStemmer


APP_DIR = Path(__file__).resolve().parent
INTENTS_PATH = APP_DIR / "intent.json"
MODEL_PATH = APP_DIR / "chatbot_model.h5"
MODEL_METADATA_PATH = APP_DIR / "chatbot_model.meta.json"
CONFIDENCE_THRESHOLD = 0.80
stemmer = LancasterStemmer()


def ensure_nltk_resources() -> None:
    """Download the tokenizers required by word_tokenize when missing."""
    resources = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
    }
    for package, resource_path in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            if not nltk.download(package, quiet=True):
                raise RuntimeError(
                    f"Could not download NLTK resource '{package}'. "
                    "Connect to the internet and try again."
                )


@st.cache_data(show_spinner=False)
def load_intents() -> dict:
    """Load and minimally validate the chatbot's training data."""
    with INTENTS_PATH.open(encoding="utf-8") as file:
        data = json.load(file)

    intents = data.get("intents")
    if not isinstance(intents, list) or not intents:
        raise ValueError("intent.json must contain a non-empty 'intents' list.")

    for intent in intents:
        if not all(key in intent for key in ("tag", "patterns", "responses")):
            raise ValueError("Each intent must include tag, patterns, and responses.")
        if not intent["responses"]:
            raise ValueError(f"Intent '{intent['tag']}' must include a response.")

    return data


def prepare_training_data(data: dict) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    """Turn intent examples into bag-of-words feature and target arrays."""
    words: list[str] = []
    labels: list[str] = []
    documents: list[tuple[list[str], str]] = []

    for intent in data["intents"]:
        tag = intent["tag"]
        if intent["patterns"] and tag not in labels:
            labels.append(tag)
        for pattern in intent["patterns"]:
            tokens = nltk.word_tokenize(pattern)
            words.extend(tokens)
            documents.append((tokens, tag))

    if not documents:
        raise ValueError("intent.json must contain at least one training pattern.")

    words = sorted({stemmer.stem(word.lower()) for word in words if word != "?"})
    training: list[list[int]] = []
    output: list[list[int]] = []

    for tokens, tag in documents:
        token_stems = {stemmer.stem(token.lower()) for token in tokens}
        training.append([int(word in token_stems) for word in words])
        target = [0] * len(labels)
        target[labels.index(tag)] = 1
        output.append(target)

    return words, labels, np.array(training), np.array(output)


def training_signature(words: list[str], labels: list[str]) -> str:
    """Return a stable identifier for the vocabulary and class ordering."""
    payload = json.dumps({"words": words, "labels": labels}, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def has_matching_model_metadata(signature: str) -> bool:
    """Check that a saved model was trained with the current feature mapping."""
    try:
        with MODEL_METADATA_PATH.open(encoding="utf-8") as file:
            metadata = json.load(file)
        return metadata.get("training_signature") == signature
    except (OSError, json.JSONDecodeError):
        return False


def save_model_metadata(signature: str) -> None:
    """Record the training signature alongside the saved Keras model."""
    with MODEL_METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump({"training_signature": signature}, file, indent=2)
        file.write("\n")


def build_and_train_model(
    training: np.ndarray, output: np.ndarray, signature: str
) -> tf.keras.Model:
    """Build a compact classifier and save it for subsequent application starts."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(training.shape[1],)),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(output.shape[1], activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=25, restore_best_weights=True
    )
    model.fit(
        training,
        output,
        epochs=300,
        batch_size=8,
        verbose=0,
        callbacks=[early_stopping],
    )
    model.save(MODEL_PATH)
    save_model_metadata(signature)
    return model


def load_or_train_model(
    training: np.ndarray, output: np.ndarray, signature: str
) -> tf.keras.Model:
    """Load a compatible saved model, or create a replacement when needed."""
    if MODEL_PATH.exists() and has_matching_model_metadata(signature):
        try:
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            if (
                model.input_shape[-1] == training.shape[1]
                and model.output_shape[-1] == output.shape[1]
            ):
                return model
        except (OSError, ValueError, TypeError):
            pass

    return build_and_train_model(training, output, signature)


def bag_of_words(text: str, words: list[str]) -> np.ndarray:
    """Vectorize a message using the vocabulary from the training examples."""
    token_stems = {stemmer.stem(token.lower()) for token in nltk.word_tokenize(text)}
    return np.array([int(word in token_stems) for word in words])


def choose_reply(
    text: str,
    model: tf.keras.Model,
    words: list[str],
    labels: list[str],
    intents: list[dict],
) -> str:
    """Classify a message and return a response only when confidence is sufficient."""
    predictions = model.predict(np.array([bag_of_words(text, words)]), verbose=0)[0]
    confidence = float(np.max(predictions))
    if confidence < CONFIDENCE_THRESHOLD:
        return "I’m sorry, I didn’t understand that. Could you rephrase it?"

    tag = labels[int(np.argmax(predictions))]
    responses = next(
        (intent["responses"] for intent in intents if intent["tag"] == tag),
        [],
    )
    return random.choice(responses) if responses else "I’m sorry, I can’t answer that yet."


def apply_theme(theme_choice: str) -> None:
    """Apply the selected light or dark colour palette."""
    if theme_choice == "Dark Mode":
        bg_color, text_color = "#111111", "#FFFFFF"
    else:
        bg_color, text_color = "#FFFFFF", "#111111"

    st.markdown(
        f"""
        <style>
            .stApp {{ background-color: {bg_color}; color: {text_color}; }}
            .gemini-title {{
                background: linear-gradient(70deg, #4285f4, #9b51e0, #e24a8a);
                background-size: 200% 200%;
                animation: geminiGlow 6s ease infinite;
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-family: 'Google Sans', sans-serif;
                font-weight: 500;
                font-size: 2.8rem;
                margin-bottom: 20px;
            }}
            .stChatMessage {{ animation: fadeIn 0.4s ease-in-out; padding: 10px 0; }}
            .stChatMessage p, .stChatMessage span, .stChatMessage div {{ color: {text_color}; }}
            .stChatInputContainer textarea {{
                background-color: {bg_color}; color: {text_color};
                border-radius: 28px; border: 1px solid rgba(128,128,128,0.2);
            }}
            @keyframes fadeIn {{
                from {{ opacity: 0; transform: translateY(5px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Gemini AI Assistant", page_icon="✨", layout="centered")

    with st.sidebar:
        st.markdown("### 🪐 Appearance")
        theme_choice = st.radio("Choose mode", ["Dark Mode", "Light Mode"])
    apply_theme(theme_choice)

    st.markdown('<h1 class="gemini-title">✨ Gemini AI Assistant</h1>', unsafe_allow_html=True)
    st.write("Ask a simple question in English or Arabic.")

    ensure_nltk_resources()
    data = load_intents()
    words, labels, training, output = prepare_training_data(data)
    model = load_or_train_model(training, output, training_signature(words, labels))

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    if user_input := st.chat_input("Ask the assistant"):
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking…"):
                sleep(0.4)
                bot_reply = choose_reply(
                    user_input, model, words, labels, data["intents"]
                )
                st.markdown(bot_reply)
        st.session_state.messages.append({"role": "assistant", "content": bot_reply})


if __name__ == "__main__":
    main()
