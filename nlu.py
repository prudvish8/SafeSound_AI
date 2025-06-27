#!/usr/bin/env python3
"""
Natural Language Understanding (NLU) module for the WhatsApp chatbot.

This module handles:
- Profile extraction from user text using Google Gemini
- Intent classification for confirmation responses
- Language detection and processing
"""

import os
import json
import google.generativeai as genai
from typing import Dict, Any, Optional
from logging_config import get_logger

# Get logger for this module
logger = get_logger(__name__)

# Configure Google Generative AI
try:
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")
    genai.configure(api_key=GOOGLE_API_KEY)
    logger.info("Google Generative AI configured successfully.")
except Exception as e:
    logger.exception("Failed to configure Google Generative AI")

def extract_profile_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract user profile information from text using Google Gemini.
    
    Args:
        text (str): User input text
        
    Returns:
        dict: Extracted profile data or None if extraction fails
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
        Extract the following information from this text and return as JSON:
        - age (number)
        - gender (string: male/female/other)
        - occupation (string)
        - income (number - monthly income in rupees)
        - caste (string)
        - ownership (boolean - whether they own a house)
        - target (string: self/other)
        - relation (string - if target is other, what relation)
        - language (string: telugu/english)
        
        Text: {text}
        
        Return only valid JSON, no other text.
        """
        
        logger.info(f"Sending text to Gemini for NLU extraction: '{text[:100]}...'")
        response = model.generate_content(prompt)
        
        # Clean the response to extract JSON
        cleaned_response = response.text.strip()
        if cleaned_response.startswith('```json'):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.endswith('```'):
            cleaned_response = cleaned_response[:-3]
        
        logger.info(f"Gemini raw response: {cleaned_response}")
        
        # Parse JSON response
        extracted_data = json.loads(cleaned_response.strip())
        logger.info(f"Successfully extracted profile data: {extracted_data}")
        
        return extracted_data
        
    except Exception as e:
        logger.exception("Error during Gemini API call or JSON parsing")
        return None

def get_confirmation_intent(text: str) -> str:
    """
    Determine if the user's response is a confirmation or rejection.
    
    Args:
        text (str): User's response text
        
    Returns:
        str: 'confirm', 'reject', or 'unknown'
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
        Classify this response as either 'confirm', 'reject', or 'unknown':
        
        Response: {text}
        
        Rules:
        - 'confirm' for: yes, అవును, correct, right, ok, okay, proceed, continue
        - 'reject' for: no, కాదు, wrong, incorrect, stop, cancel
        - 'unknown' for: anything else
        
        Return only the classification word, no other text.
        """
        
        logger.info(f"get_confirmation_intent: Received text: '{text}'")
        response = model.generate_content(prompt)
        
        # Clean and classify the response
        intent = response.text.strip().lower()
        
        logger.info(f"Gemini raw intent response: '{response.text.strip()}'")
        logger.info(f"NLU classified confirmation intent as: {intent}")
        
        if intent in ['confirm', 'reject', 'unknown']:
            return intent
        else:
            logger.warning(f"NLU returned unexpected intent: {intent}")
            return 'unknown'
            
    except Exception as e:
        logger.exception("Error during intent classification")
        return 'unknown'

# --- END OF nlu.py ---