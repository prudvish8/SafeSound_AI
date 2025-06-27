# Test Suite Documentation

## Overview

This test suite has been refactored to run completely offline without making any real network requests. All external API calls are properly mocked to ensure fast, reliable, and clean test execution.

## Test Files

### `test_twilio_signature_validation.py`
Tests the critical security feature that validates X-Twilio-Signature headers to ensure webhook requests come from Twilio and not malicious actors.

**Key Features:**
- Tests valid/invalid signature validation
- Tests missing signature headers
- Tests exception handling
- Tests validator initialization
- Tests voice message handling
- Tests multiple request scenarios
- Tests unicode data handling

### `test_utils_state_management.py`
Tests the database-backed session management functions for user state persistence.

**Key Features:**
- Tests state retrieval and storage
- Tests database error handling
- Tests JSON serialization/deserialization
- Tests complex data structures
- Tests connection cleanup
- Tests edge cases and error conditions

## Mocking Strategy

### External APIs Mocked

1. **Twilio API**
   - `app.twilio_client.messages.create` - Prevents real WhatsApp message sending
   - Returns mock response objects

2. **Whisper API**
   - `app.requests.post` - Prevents real speech-to-text transcription
   - Returns mock transcription results

3. **Google Gemini API**
   - `nlu.extract_profile_from_text` - Prevents real LLM profile extraction
   - `google.generativeai.GenerativeModel.generate_content` - Prevents real LLM calls
   - Returns mock profile data

4. **Background Processing**
   - `app.process_voice_message_in_background` - Prevents real background processing
   - `app.process_text_in_background` - Prevents real background processing

### Shared Configuration

The `conftest.py` file provides shared fixtures and mock configurations:
- Environment variable mocking
- Common mock responses
- Reusable test data
- Consistent mock behavior across all tests

## Test Execution

### Running All Tests
```bash
pytest -v
```

### Running with Coverage
```bash
pytest --cov=app --cov=utils --cov=nlu --cov-report=term-missing
```

### Running Specific Test Files
```bash
pytest tests/test_twilio_signature_validation.py -v
pytest tests/test_utils_state_management.py -v
```

## Benefits Achieved

1. **Fast Execution**: Tests run in ~1-2 seconds instead of waiting for network timeouts
2. **Clean Output**: No authentication errors or API failure messages
3. **Reliable**: Tests don't depend on external service availability
4. **Isolated**: Each test runs independently without side effects
5. **Consistent**: Mock responses provide predictable test data

## Test Coverage

Current coverage focuses on:
- **Security validation** (Twilio signature verification)
- **State management** (database operations)
- **Error handling** (network failures, database errors)
- **Edge cases** (invalid data, missing headers)

## Future Improvements

1. **Add integration tests** for the complete user flow
2. **Add performance tests** for database operations
3. **Add load tests** for concurrent user handling
4. **Add API endpoint tests** for all webhook scenarios
5. **Improve coverage** for business logic functions

## Mock Data Examples

### Sample Profile Data
```python
{
    'target': 'self',
    'age': 30,
    'gender': 'male',
    'occupation': 'farmer',
    'income': 5000,
    'ownership': True,
    'caste': 'BC',
    'language': 'telugu'
}
```

### Sample Webhook Data
```python
{
    'From': 'whatsapp:+1234567890',
    'To': 'whatsapp:+0987654321',
    'Body': 'Hello, this is a test message',
    'MessageSid': 'SM1234567890abcdef',
    'AccountSid': 'AC1234567890abcdef'
}
```

## Best Practices

1. **Always mock external APIs** - Never make real network calls in tests
2. **Use descriptive mock data** - Make test data realistic and meaningful
3. **Test error conditions** - Ensure error handling works correctly
4. **Keep tests isolated** - Each test should be independent
5. **Use shared fixtures** - Leverage conftest.py for common configurations 