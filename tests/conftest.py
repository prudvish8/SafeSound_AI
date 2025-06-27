#!/usr/bin/env python3
"""
Shared pytest configuration and fixtures for the WhatsApp chatbot backend tests.

This file provides common mock configurations and fixtures that can be used
across all test files to ensure consistent mocking of external APIs.
"""

import pytest
from unittest.mock import MagicMock, patch
import os


@pytest.fixture(scope="session", autouse=True)
def mock_environment_variables():
    """Mock all required environment variables for testing."""
    env_vars = {
        'TWILIO_ACCOUNT_SID': 'test_account_sid',
        'TWILIO_AUTH_TOKEN': 'test_auth_token',
        'WHISPER_REMOTE_URL': 'http://test-whisper.com/transcribe',
        'GOOGLE_API_KEY': 'test_google_api_key',
        'TWILIO_TEMPLATE_SID_EN_CONFIRM': 'test_template_en_confirm',
        'TWILIO_TEMPLATE_SID_TE_CONFIRM': 'test_template_te_confirm',
        'TWILIO_TEMPLATE_SID_TE_SCHEME_DETAILS': 'test_template_te_scheme',
        'TWILIO_TEMPLATE_SID_EN_SCHEME_DETAILS': 'test_template_en_scheme'
    }
    
    with patch.dict(os.environ, env_vars):
        yield env_vars


@pytest.fixture
def mock_twilio_client():
    """Mock Twilio client to prevent real API calls."""
    with patch('app.twilio_client') as mock_client:
        mock_message = MagicMock()
        mock_client.messages.create.return_value = mock_message
        yield mock_client


@pytest.fixture
def mock_whisper_api():
    """Mock Whisper API calls to prevent real transcription requests."""
    with patch('app.requests.post') as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {"text": "test transcription result"}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        yield mock_post


@pytest.fixture
def mock_nlu_extraction():
    """Mock NLU extraction to prevent real LLM API calls."""
    with patch('nlu.extract_profile_from_text') as mock_nlu:
        mock_nlu.return_value = {
            'target': 'self',
            'age': 30,
            'gender': 'male',
            'occupation': 'farmer',
            'income': 5000,
            'ownership': True,
            'caste': 'BC',
            'language': 'telugu'
        }
        yield mock_nlu


@pytest.fixture
def mock_google_genai():
    """Mock Google Generative AI to prevent real API calls."""
    with patch('google.generativeai.GenerativeModel.generate_content') as mock_genai:
        mock_response = MagicMock()
        mock_response.text = '{"target": "self", "age": 30, "gender": "male"}'
        mock_genai.return_value = mock_response
        yield mock_genai


@pytest.fixture
def mock_background_processing():
    """Mock background processing functions to prevent real execution."""
    with patch('app.process_voice_message_in_background') as mock_voice, \
         patch('app.process_text_in_background') as mock_text:
        
        mock_voice.return_value = None
        mock_text.return_value = None
        
        yield {
            'voice': mock_voice,
            'text': mock_text
        }


@pytest.fixture
def sample_profile_data():
    """Sample profile data for testing."""
    return {
        'target': 'self',
        'age': 35,
        'gender': 'female',
        'occupation': 'teacher',
        'income': 8000,
        'ownership': False,
        'caste': 'OC',
        'language': 'english'
    }


@pytest.fixture
def sample_webhook_data():
    """Sample webhook data that Twilio would send."""
    return {
        'From': 'whatsapp:+1234567890',
        'To': 'whatsapp:+0987654321',
        'Body': 'Hello, this is a test message',
        'MessageSid': 'SM1234567890abcdef',
        'AccountSid': 'AC1234567890abcdef'
    }


@pytest.fixture
def sample_voice_webhook_data():
    """Sample voice message webhook data."""
    return {
        'From': 'whatsapp:+1234567890',
        'To': 'whatsapp:+0987654321',
        'MediaUrl0': 'https://api.twilio.com/2010-04-01/Accounts/AC123/Messages/MM123/Media/ME123',
        'MessageSid': 'SM1234567890abcdef'
    } 