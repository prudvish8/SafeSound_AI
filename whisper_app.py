# --- Contents of whisper_app.py ---
import torch
from transformers import pipeline
from flask import Flask, request, jsonify
import tempfile
import os
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Load the model once when the server starts
logging.info("Loading Whisper model for transcription service...")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
WHISPER_MODEL = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",
    chunk_length_s=30,
    device=device,
)
logging.info("Whisper model loaded.")

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    
    with tempfile.NamedTemporaryFile(delete=False) as temp_audio_file:
        file.save(temp_audio_file.name)
        temp_filename = temp_audio_file.name

    try:
        logging.info(f"Transcribing file: {temp_filename}")
        result = WHISPER_MODEL(temp_filename)
        logging.info(f"Transcription result: {result['text']}")
        return jsonify(result)
    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

if __name__ == '__main__':
    # Run on a different port, e.g., 9000
    app.run(host='0.0.0.0', port=9000)