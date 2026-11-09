# Gemini AI Assistant

A compact, bilingual Streamlit chatbot that classifies messages into predefined intents using a neural-network bag-of-words model. It supports common English and Arabic greetings, questions, confirmations, basic programming prompts, and more.

> This is an intent-classification demo, not a generative LLM and not an official Google Gemini product.

## Highlights

- Streamlit chat interface with light and dark themes.
- English and Arabic example phrases.
- Tokenization and Lancaster stemming with NLTK.
- TensorFlow/Keras classifier with a confidence threshold for unknown requests.
- A saved model is reused only when its feature and class mapping match the current intent data.
- Training data is easy to extend in `intent.json`.

## Project structure

```text
ChatBot/
├── main.py              # Streamlit application and model lifecycle
├── intent.json          # Intent tags, example patterns, and replies
├── requirements.txt     # Python dependencies
├── chatbot_model.h5     # Pre-trained Keras model
├── chatbot_model.meta.json # Generated model compatibility metadata
├── data.pickle          # Legacy training-data cache from earlier versions
└── explantion/          # Background documentation (Arabic PDFs)
```

## Requirements

- Python 3.10–3.11 is recommended.
- `pip` and an internet connection are required for the first dependency install and, if missing, the first NLTK tokenizer download.

## Quick start

Create and activate a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies and start the application:

```powershell
pip install -r requirements.txt
streamlit run main.py
```

Streamlit will print a local URL, normally `http://localhost:8501`.

## How it works

1. `intent.json` supplies example phrases and possible replies for each intent.
2. The app tokenizes and stems each phrase, then converts it to a bag-of-words vector.
3. The TensorFlow model predicts an intent for the vector.
4. A reply is returned only when the top prediction meets the 0.80 confidence threshold; otherwise the assistant asks the user to rephrase.

On startup, the app checks whether the saved model matches the current vocabulary and intent ordering. If it does not, it retrains and replaces `chatbot_model.h5` automatically. The accompanying metadata file is created after the first training run.

## Customising intents

Edit `intent.json` to add examples or replies. Each intent follows this shape:

```json
{
  "tag": "greeting",
  "patterns": ["hello", "مرحبا"],
  "responses": ["Hello!", "Welcome!"]
}
```

Add at least one pattern and one response for every trainable intent. Restart the app after editing; it will detect a vocabulary or label-order change and retrain as needed. If you only change response text, no retraining is necessary.

## Limitations

- The assistant selects only from the intents defined in `intent.json`; it does not browse the web, retain knowledge, or generate open-ended answers.
- Its small dataset makes it best suited to demos and learning projects, not production customer support.
- The `time`, `date`, and `weather` intents intentionally do not access live services.

## Troubleshooting

- **`py` reports that no Python is installed:** install Python 3.10 or 3.11 from [python.org](https://www.python.org/downloads/) and enable the option to add Python to PATH.
- **NLTK tokenizer download fails:** connect to the internet, then rerun the app. The app downloads `punkt` and `punkt_tab` when they are absent.
- **TensorFlow installation fails:** confirm that the active interpreter is a supported 64-bit Python version, then upgrade pip with `python -m pip install --upgrade pip` before reinstalling.

## License

No license has been supplied. Add one before distributing or reusing this project publicly.
