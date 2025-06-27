#!/usr/bin/env python3
"""
Whisper transcription service for the WhatsApp chatbot.
"""

import os
import tempfile
import whisper
from logging_config import get_logger

logger = get_logger(__name__)

# Load Whisper model
try:
    logger.info("Loading Whisper model for transcription service...")
    model = whisper.load_model("base")
    logger.info("Whisper model loaded.")
except Exception as e:
    logger.exception("Failed to load Whisper model")
    model = None

def transcribe_audio_file(audio_file_path: str) -> str:
    """
    Transcribe an audio file using Whisper.
    
    Args:
        audio_file_path (str): Path to the audio file
        
    Returns:
        str: Transcribed text or empty string if transcription fails
    """
    if model is None:
        logger.error("Whisper model not loaded")
        return ""
    
    try:
        logger.info(f"Transcribing file: {audio_file_path}")
        result = model.transcribe(audio_file_path)
        transcription = result["text"].strip()
        logger.info(f"Transcription result: {transcription}")
        return transcription
    except Exception as e:
        logger.exception("Error during transcription")
        return ""

def transcribe_audio_bytes(audio_bytes: bytes, file_extension: str = ".wav") -> str:
    """
    Transcribe audio bytes using Whisper.
    
    Args:
        audio_bytes (bytes): Audio data as bytes
        file_extension (str): File extension for the temporary file
        
    Returns:
        str: Transcribed text or empty string if transcription fails
    """
    if model is None:
        logger.error("Whisper model not loaded")
        return ""
    
    try:
        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_filename = temp_file.name
        
        transcription = transcribe_audio_file(temp_filename)
        
        # Clean up temporary file
        os.unlink(temp_filename)
        
        return transcription
    except Exception as e:
        logger.exception("Error during audio bytes transcription")
        return ""