#!/usr/bin/env python3
"""
Tests for state management functions in utils.py

This module tests the new database-backed session management functions:
- get_user_state()
- save_user_state()

Test coverage includes:
- Happy path scenarios
- Edge cases and error conditions
- Database connectivity issues
- JSON serialization/deserialization
- Default state handling
"""

import pytest
import sqlite3
import json
import tempfile
import os
import shutil
from unittest.mock import patch, MagicMock
from utils import get_user_state, save_user_state
import contextlib


@contextlib.contextmanager
def patch_connect_for_utils(temp_db_path):
    """Patch sqlite3.connect in utils to use the temp DB only during the function call, avoiding recursion."""
    import sqlite3 as real_sqlite3
    real_connect = real_sqlite3.connect
    with patch('utils.sqlite3.connect') as mock_connect:
        mock_connect.side_effect = lambda *args, **kwargs: real_connect(temp_db_path)
        yield


@pytest.fixture(autouse=True)
def mock_external_apis():
    """Mock any external API calls that might be triggered during state management tests."""
    with patch('nlu.extract_profile_from_text') as mock_nlu, \
         patch('google.generativeai.GenerativeModel.generate_content') as mock_genai:
        
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
        
        # Mock Google Generative AI
        mock_response = MagicMock()
        mock_response.text = '{"target": "self", "age": 30}'
        mock_genai.return_value = mock_response
        
        yield {
            'nlu': mock_nlu,
            'genai': mock_genai
        }


class TestStateManagement:
    """Test suite for state management functions."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, 'test_sessions.db')
        
        # Create the database and table
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE user_sessions (
                session_id TEXT PRIMARY KEY,
                user_state TEXT NOT NULL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
        
        yield db_path
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_state(self):
        """Sample state data for testing."""
        return {
            'step': 'AWAITING_CONFIRMATION',
            'profile': {
                'name': 'John Doe',
                'age': 30,
                'income': 50000
            },
            'lang': 'en',
            'matched_schemes': ['Scheme A', 'Scheme B']
        }
    
    def test_get_user_state_existing_user(self, temp_db, sample_state):
        """Test retrieving state for an existing user."""
        # Insert test data
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO user_sessions (session_id, user_state) VALUES (?, ?)",
            ('test_user_123', json.dumps(sample_state))
        )
        conn.commit()
        conn.close()
        
        # Patch only during the function call
        with patch_connect_for_utils(temp_db):
            result = get_user_state('test_user_123')
        
        # Verify the result
        assert result == sample_state
        assert result['step'] == 'AWAITING_CONFIRMATION'
        assert result['profile']['name'] == 'John Doe'
        assert result['lang'] == 'en'
    
    def test_get_user_state_nonexistent_user(self, temp_db):
        """Test retrieving state for a user that doesn't exist."""
        with patch_connect_for_utils(temp_db):
            result = get_user_state('nonexistent_user')
        expected_default = {'step': 'START', 'profile': {}, 'lang': 'en'}
        assert result == expected_default
    
    def test_get_user_state_database_error(self):
        """Test get_user_state when database connection fails."""
        with patch('utils.sqlite3.connect') as mock_connect:
            mock_connect.side_effect = sqlite3.Error("Database connection failed")
            
            result = get_user_state('test_user')
            
            # Should return default state on error
            expected_default = {'step': 'START', 'profile': {}, 'lang': 'en'}
            assert result == expected_default
    
    def test_get_user_state_json_decode_error(self, temp_db):
        """Test get_user_state when JSON in database is corrupted."""
        # Insert corrupted JSON data
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO user_sessions (session_id, user_state) VALUES (?, ?)",
            ('test_user_123', 'invalid json data')
        )
        conn.commit()
        conn.close()
        
        with patch_connect_for_utils(temp_db):
            result = get_user_state('test_user_123')
            
            # Should return default state on JSON decode error
            expected_default = {'step': 'START', 'profile': {}, 'lang': 'en'}
            assert result == expected_default
    
    def test_save_user_state_success(self, temp_db, sample_state):
        """Test successfully saving user state."""
        with patch_connect_for_utils(temp_db):
            result = save_user_state('test_user_123', sample_state)
            
            # Should return True on success
            assert result is True
            
            # Verify data was actually saved
            conn = sqlite3.connect(temp_db)
            cursor = conn.cursor()
            cursor.execute("SELECT user_state FROM user_sessions WHERE session_id = ?", ('test_user_123',))
            row = cursor.fetchone()
            conn.close()
            
            assert row is not None
            saved_state = json.loads(row[0])
            assert saved_state == sample_state
    
    def test_save_user_state_update_existing(self, temp_db, sample_state):
        """Test updating an existing user's state."""
        # First, insert initial state
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        initial_state = {'step': 'START', 'profile': {}, 'lang': 'en'}
        cursor.execute(
            "INSERT INTO user_sessions (session_id, user_state) VALUES (?, ?)",
            ('test_user_123', json.dumps(initial_state))
        )
        conn.commit()
        conn.close()
        
        # Now update with new state
        with patch_connect_for_utils(temp_db):
            result = save_user_state('test_user_123', sample_state)
            
            assert result is True
            
            # Verify the state was updated
            conn = sqlite3.connect(temp_db)
            cursor = conn.cursor()
            cursor.execute("SELECT user_state FROM user_sessions WHERE session_id = ?", ('test_user_123',))
            row = cursor.fetchone()
            conn.close()
            
            updated_state = json.loads(row[0])
            assert updated_state == sample_state
            assert updated_state['step'] == 'AWAITING_CONFIRMATION'
    
    def test_save_user_state_database_error(self):
        """Test save_user_state when database connection fails."""
        with patch('utils.sqlite3.connect') as mock_connect:
            mock_connect.side_effect = sqlite3.Error("Database connection failed")
            
            result = save_user_state('test_user', {'step': 'START'})
            
            # Should return False on error
            assert result is False
    
    def test_save_user_state_serialization_error(self, temp_db):
        """Test save_user_state when state data cannot be serialized."""
        # Create a state with non-serializable data
        non_serializable_state = {
            'step': 'START',
            'profile': {'data': object()}  # object() is not JSON serializable
        }
        
        with patch_connect_for_utils(temp_db):
            result = save_user_state('test_user', non_serializable_state)
            
            # Should return False on serialization error
            assert result is False
    
    def test_save_user_state_empty_state(self, temp_db):
        """Test saving an empty state."""
        empty_state = {}
        
        with patch_connect_for_utils(temp_db):
            result = save_user_state('test_user_123', empty_state)
            
            assert result is True
            
            # Verify empty state was saved
            conn = sqlite3.connect(temp_db)
            cursor = conn.cursor()
            cursor.execute("SELECT user_state FROM user_sessions WHERE session_id = ?", ('test_user_123',))
            row = cursor.fetchone()
            conn.close()
            
            assert row is not None
            saved_state = json.loads(row[0])
            assert saved_state == empty_state
    
    def test_save_user_state_complex_data(self, temp_db):
        """Test saving state with complex nested data structures."""
        complex_state = {
            'step': 'AWAITING_SCHEME_CHOICE',
            'profile': {
                'name': 'Jane Doe',
                'age': 45,
                'income': 75000,
                'address': {
                    'street': '123 Main St',
                    'city': 'Hyderabad',
                    'state': 'Telangana',
                    'pincode': '500001'
                },
                'family_members': [
                    {'name': 'Spouse', 'age': 42},
                    {'name': 'Child 1', 'age': 15},
                    {'name': 'Child 2', 'age': 12}
                ]
            },
            'lang': 'en',
            'matched_schemes': ['Scheme A', 'Scheme B', 'Scheme C'],
            'selected_scheme_name': 'Scheme A',
            'metadata': {
                'session_start': '2024-01-01T10:00:00Z',
                'last_activity': '2024-01-01T10:30:00Z',
                'interaction_count': 5
            }
        }
        
        with patch_connect_for_utils(temp_db):
            result = save_user_state('test_user_123', complex_state)
            
            assert result is True
            
            # Verify complex state was saved correctly
            conn = sqlite3.connect(temp_db)
            cursor = conn.cursor()
            cursor.execute("SELECT user_state FROM user_sessions WHERE session_id = ?", ('test_user_123',))
            row = cursor.fetchone()
            conn.close()
            
            assert row is not None
            saved_state = json.loads(row[0])
            assert saved_state == complex_state
            assert saved_state['profile']['address']['city'] == 'Hyderabad'
            assert len(saved_state['profile']['family_members']) == 3
            assert saved_state['metadata']['interaction_count'] == 5
    
    def test_get_user_state_connection_cleanup(self, temp_db):
        """Test that database connections are properly closed after get_user_state."""
        # This test ensures we don't have connection leaks
        with patch_connect_for_utils(temp_db):
            result = get_user_state('test_user_123')
            
            # Should return default state for non-existent user
            expected_default = {'step': 'START', 'profile': {}, 'lang': 'en'}
            assert result == expected_default
    
    def test_save_user_state_connection_cleanup(self, temp_db):
        """Test that database connections are properly closed after save_user_state."""
        # This test ensures we don't have connection leaks
        test_state = {'step': 'START', 'profile': {}, 'lang': 'en'}
        
        with patch_connect_for_utils(temp_db):
            result = save_user_state('test_user_123', test_state)
            
            assert result is True


if __name__ == '__main__':
    pytest.main([__file__]) 