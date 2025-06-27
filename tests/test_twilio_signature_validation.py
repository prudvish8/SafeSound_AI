#!/usr/bin/env python3
"""
Tests for Twilio Signature Validation Security Feature

This module tests the critical security feature that validates X-Twilio-Signature
headers to ensure webhook requests come from Twilio and not malicious actors.
"""

import pytest
from unittest.mock import patch, MagicMock
from flask import Flask
from twilio.request_validator import RequestValidator

# Import the app after mocking to avoid initialization issues
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock environment variables before importing app
with patch.dict(os.environ, {
    'TWILIO_ACCOUNT_SID': 'test_account_sid',
    'TWILIO_AUTH_TOKEN': 'test_auth_token',
    'WHISPER_REMOTE_URL': 'http://test-whisper.com/transcribe',
    'GOOGLE_API_KEY': 'test_google_api_key'
}):
    from app import app

@pytest.fixture(autouse=True)
def mock_all_external_apis():
    """Mock all external API calls to prevent network requests during testing."""
    with patch('app.twilio_client') as mock_twilio, \
         patch('app.requests.post') as mock_requests_post, \
         patch('nlu.extract_profile_from_text') as mock_nlu, \
         patch('app.process_voice_message_in_background') as mock_voice_bg, \
         patch('app.process_text_in_background') as mock_text_bg:
        
        # Mock Twilio client
        mock_twilio.messages.create.return_value = MagicMock()
        
        # Mock requests.post for Whisper API
        mock_response = MagicMock()
        mock_response.json.return_value = {"text": "test transcription"}
        mock_response.raise_for_status.return_value = None
        mock_requests_post.return_value = mock_response
        
        # Mock NLU extraction
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
        
        # Mock background processing to prevent real execution
        mock_voice_bg.return_value = None
        mock_text_bg.return_value = None
        
        yield {
            'twilio': mock_twilio,
            'requests_post': mock_requests_post,
            'nlu': mock_nlu,
            'voice_bg': mock_voice_bg,
            'text_bg': mock_text_bg
        }

class TestTwilioSignatureValidation:
    """Test suite for Twilio signature validation security feature."""
    
    @pytest.fixture
    def client(self):
        """Create a test client for the Flask app."""
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    @pytest.fixture
    def sample_webhook_data(self):
        """Sample webhook data that Twilio would send."""
        return {
            'From': 'whatsapp:+1234567890',
            'To': 'whatsapp:+0987654321',
            'Body': 'Hello, this is a test message',
            'MessageSid': 'SM1234567890abcdef',
            'AccountSid': 'AC1234567890abcdef'
        }
    
    @patch('app.validator')
    def test_valid_twilio_signature_accepts_request(self, mock_validator, client, sample_webhook_data):
        """Test that requests with valid Twilio signatures are accepted."""
        # Mock validator to return True (valid signature)
        mock_validator.validate.return_value = True
        
        # Create request with valid signature header
        headers = {
            'X-Twilio-Signature': 'valid_signature_here',
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        response = client.post('/whatsapp', data=sample_webhook_data, headers=headers)
        
        # Should accept the request (not return 403)
        assert response.status_code != 403
        # Updated to expect the processing message instead of welcome message
        assert 'Thank you. Processing your text message' in response.get_data(as_text=True)
        
        # Verify validator was called
        mock_validator.validate.assert_called_once()
    
    @patch('app.validator')
    def test_invalid_twilio_signature_rejects_request(self, mock_validator, client, sample_webhook_data):
        """Test that requests with invalid Twilio signatures are rejected."""
        # Mock validator to return False (invalid signature)
        mock_validator.validate.return_value = False
        
        headers = {
            'X-Twilio-Signature': 'invalid_signature_here',
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        response = client.post('/whatsapp', data=sample_webhook_data, headers=headers)
        
        # Should reject the request with 403 Forbidden
        assert response.status_code == 403
        assert response.get_data(as_text=True) == 'Forbidden'
    
    def test_missing_signature_header_rejects_request(self, client, sample_webhook_data):
        """Test that requests without X-Twilio-Signature header are rejected."""
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded'
            # No X-Twilio-Signature header
        }
        
        response = client.post('/whatsapp', data=sample_webhook_data, headers=headers)
        
        # Should reject the request with 403 Forbidden
        assert response.status_code == 403
        assert response.get_data(as_text=True) == 'Forbidden'
    
    @patch('app.validator')
    def test_empty_signature_header_rejects_request(self, mock_validator, client, sample_webhook_data):
        """Test that requests with empty signature header are rejected."""
        headers = {
            'X-Twilio-Signature': '',
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        response = client.post('/whatsapp', data=sample_webhook_data, headers=headers)
        
        # Should reject the request with 403 Forbidden
        assert response.status_code == 403
        assert response.get_data(as_text=True) == 'Forbidden'
        
        # Validator should not be called with empty signature
        mock_validator.validate.assert_not_called()
    
    @patch('app.validator')
    def test_validator_exception_handling(self, mock_validator, client, sample_webhook_data):
        """Test that exceptions in signature validation are handled gracefully."""
        # Mock validator to raise an exception
        mock_validator.validate.side_effect = Exception("Validation error")
        
        headers = {
            'X-Twilio-Signature': 'signature_here',
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        response = client.post('/whatsapp', data=sample_webhook_data, headers=headers)
        
        # Should reject the request on validation error with 403 Forbidden
        assert response.status_code == 403
        assert response.get_data(as_text=True) == 'Forbidden'
    
    def test_validator_not_initialized_rejects_request(self, client, sample_webhook_data):
        """Test that requests are rejected when validator is not initialized."""
        # Mock the validator to be None (simulating initialization failure)
        with patch('app.validator', None):
            headers = {
                'X-Twilio-Signature': 'signature_here',
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            response = client.post('/whatsapp', data=sample_webhook_data, headers=headers)
            
            # Should reject the request with 403 Forbidden
            assert response.status_code == 403
            assert response.get_data(as_text=True) == 'Forbidden'
    
    @patch('app.validator')
    def test_validator_called_with_correct_parameters(self, mock_validator, client, sample_webhook_data):
        """Test that the validator is called with the correct parameters."""
        mock_validator.validate.return_value = True
        
        headers = {
            'X-Twilio-Signature': 'test_signature_123',
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        response = client.post('/whatsapp', data=sample_webhook_data, headers=headers)
        
        # Verify validator was called with correct parameters
        mock_validator.validate.assert_called_once()
        call_args = mock_validator.validate.call_args
        
        # Should be called with (url, post_data, signature)
        assert len(call_args[0]) == 3
        url, post_data, signature = call_args[0]
        
        # Check parameters
        assert 'whatsapp' in url  # URL should contain the endpoint
        # Updated: post_data should be a MultiDict (request.form)
        from werkzeug.datastructures import MultiDict
        assert isinstance(post_data, MultiDict)  # POST data should be MultiDict
        assert signature == 'test_signature_123'  # Signature should match
    
    @patch('app.validator')
    def test_voice_message_with_valid_signature(self, mock_validator, client):
        """Test that voice messages with valid signatures are accepted."""
        mock_validator.validate.return_value = True
        
        voice_data = {
            'From': 'whatsapp:+1234567890',
            'To': 'whatsapp:+0987654321',
            'MediaUrl0': 'https://api.twilio.com/2010-04-01/Accounts/AC123/Messages/MM123/Media/ME123',
            'MessageSid': 'SM1234567890abcdef'
        }
        
        headers = {
            'X-Twilio-Signature': 'valid_signature_here',
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        response = client.post('/whatsapp', data=voice_data, headers=headers)
        
        # Should accept the request (not return 403)
        assert response.status_code != 403
        assert 'Thank you. Processing your voice message' in response.get_data(as_text=True)
    
    @patch('app.validator')
    def test_multiple_requests_with_different_signatures(self, mock_validator, client, sample_webhook_data):
        """Test that multiple requests with different signatures are handled correctly."""
        # First request with valid signature
        mock_validator.validate.return_value = True
        headers1 = {
            'X-Twilio-Signature': 'valid_signature_1',
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        response1 = client.post('/whatsapp', data=sample_webhook_data, headers=headers1)
        assert response1.status_code != 403
        
        # Second request with invalid signature
        mock_validator.validate.return_value = False
        headers2 = {
            'X-Twilio-Signature': 'invalid_signature_2',
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        response2 = client.post('/whatsapp', data=sample_webhook_data, headers=headers2)
        assert response2.status_code == 403
        
        # Verify validator was called for both requests
        assert mock_validator.validate.call_count == 2
    
    @patch('app.validator')
    def test_request_with_extra_headers(self, mock_validator, client, sample_webhook_data):
        """Test that requests with extra headers are handled correctly."""
        mock_validator.validate.return_value = True
        
        headers = {
            'X-Twilio-Signature': 'valid_signature_here',
            'Content-Type': 'application/x-www-form-urlencoded',
            'X-Custom-Header': 'custom_value',
            'User-Agent': 'TestBot/1.0'
        }
        
        response = client.post('/whatsapp', data=sample_webhook_data, headers=headers)
        
        # Should accept the request despite extra headers
        assert response.status_code != 403
        assert 'Thank you. Processing your text message' in response.get_data(as_text=True)
    
    @patch('app.validator')
    def test_request_with_unicode_data(self, mock_validator, client):
        """Test that requests with unicode data are handled correctly."""
        mock_validator.validate.return_value = True
        
        unicode_data = {
            'From': 'whatsapp:+1234567890',
            'To': 'whatsapp:+0987654321',
            'Body': 'నమస్కారం! ఎలా ఉన్నారు?',  # Telugu text
            'MessageSid': 'SM1234567890abcdef'
        }
        
        headers = {
            'X-Twilio-Signature': 'valid_signature_here',
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        response = client.post('/whatsapp', data=unicode_data, headers=headers)
        
        # Should accept the request with unicode data
        assert response.status_code != 403
        assert 'Thank you. Processing your text message' in response.get_data(as_text=True)


class TestTwilioSignatureValidationIntegration:
    """Integration tests for Twilio signature validation."""
    
    def test_real_validator_initialization(self):
        """Test that the real validator can be initialized with test credentials."""
        validator = RequestValidator('test_auth_token')
        assert validator is not None
    
    def test_real_validator_with_fake_data(self):
        """Test that the real validator works with fake data."""
        validator = RequestValidator('test_auth_token')
        # This should not raise an exception even with fake data
        result = validator.validate('http://test.com', {}, 'fake_signature')
        assert isinstance(result, bool)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
