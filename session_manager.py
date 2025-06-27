#!/usr/bin/env python3
"""
Session Manager Module

This module provides database-based session management for the WhatsApp chatbot.
It replaces the file-based user_states.json with a more secure and performant
SQLite database solution.

Key Features:
- Thread-safe session operations
- Automatic timestamp management
- Session cleanup and expiration
- Error handling and logging
- Backward compatibility with existing state structure
"""

import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import threading

# Configure logging
logger = logging.getLogger(__name__)

class SessionManager:
    """
    Database-based session manager for user conversation states.
    """
    
    def __init__(self, db_path: str = "sessions.db"):
        """
        Initialize the session manager.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._lock = threading.Lock()
        
    def _get_connection(self) -> Optional[sqlite3.Connection]:
        """
        Get a database connection with proper configuration.
        
        Returns:
            sqlite3.Connection or None if connection fails
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.Error as e:
            logger.error(f"Failed to connect to database: {e}")
            return None
    
    def get_user_state(self, session_id: str) -> Dict[str, Any]:
        """
        Retrieve user session state from the database.
        
        Args:
            session_id: WhatsApp sender ID (e.g., "whatsapp:+1234567890")
            
        Returns:
            Dict containing the user's session state, or default state if not found
        """
        default_state = {
            'step': 'START',
            'profile': {},
            'lang': 'en'
        }
        
        with self._lock:
            conn = self._get_connection()
            if not conn:
                logger.error(f"Could not connect to database for session {session_id}")
                return default_state
            
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT user_state FROM user_sessions WHERE session_id = ?",
                    (session_id,)
                )
                result = cursor.fetchone()
                
                if result:
                    try:
                        state = json.loads(result['user_state'])
                        logger.debug(f"Retrieved state for {session_id}: {state}")
                        return state
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON for session {session_id}: {e}")
                        return default_state
                else:
                    logger.debug(f"No existing state found for {session_id}, returning default")
                    return default_state
                    
            except sqlite3.Error as e:
                logger.error(f"Database error retrieving state for {session_id}: {e}")
                return default_state
            finally:
                conn.close()
    
    def save_user_state(self, session_id: str, state: Dict[str, Any]) -> bool:
        """
        Save or update user session state in the database.
        
        Args:
            session_id: WhatsApp sender ID
            state: Dictionary containing the session state
            
        Returns:
            bool: True if successful, False otherwise
        """
        with self._lock:
            conn = self._get_connection()
            if not conn:
                logger.error(f"Could not connect to database for session {session_id}")
                return False
            
            try:
                # Serialize state to JSON
                state_json = json.dumps(state, ensure_ascii=False)
                
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO user_sessions (session_id, user_state, last_updated)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                """, (session_id, state_json))
                
                conn.commit()
                logger.debug(f"Saved state for {session_id}: {state}")
                return True
                
            except sqlite3.Error as e:
                logger.error(f"Database error saving state for {session_id}: {e}")
                return False
            except json.JSONEncodeError as e:
                logger.error(f"JSON encoding error for session {session_id}: {e}")
                return False
            finally:
                conn.close()
    
    def clear_user_state(self, session_id: str) -> bool:
        """
        Clear user session state from the database.
        
        Args:
            session_id: WhatsApp sender ID
            
        Returns:
            bool: True if successful, False otherwise
        """
        with self._lock:
            conn = self._get_connection()
            if not conn:
                logger.error(f"Could not connect to database for session {session_id}")
                return False
            
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM user_sessions WHERE session_id = ?",
                    (session_id,)
                )
                
                conn.commit()
                logger.info(f"Cleared state for session {session_id}")
                return True
                
            except sqlite3.Error as e:
                logger.error(f"Database error clearing state for {session_id}: {e}")
                return False
            finally:
                conn.close()
    
    def cleanup_old_sessions(self, days_old: int = 7) -> int:
        """
        Remove sessions older than the specified number of days.
        
        Args:
            days_old: Number of days after which sessions should be considered old
            
        Returns:
            int: Number of sessions removed
        """
        with self._lock:
            conn = self._get_connection()
            if not conn:
                logger.error("Could not connect to database for cleanup")
                return 0
            
            try:
                cursor = conn.cursor()
                cutoff_date = datetime.now() - timedelta(days=days_old)
                cutoff_str = cutoff_date.strftime('%Y-%m-%d %H:%M:%S')
                
                cursor.execute("""
                    DELETE FROM user_sessions 
                    WHERE last_updated < ?
                """, (cutoff_str,))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                if deleted_count > 0:
                    logger.info(f"Cleaned up {deleted_count} old sessions (older than {days_old} days)")
                else:
                    logger.debug(f"No old sessions found to clean up (older than {days_old} days)")
                
                return deleted_count
                
            except sqlite3.Error as e:
                logger.error(f"Database error during cleanup: {e}")
                return 0
            finally:
                conn.close()
    
    def get_session_count(self) -> int:
        """
        Get the total number of active sessions.
        
        Returns:
            int: Number of active sessions
        """
        with self._lock:
            conn = self._get_connection()
            if not conn:
                logger.error("Could not connect to database for session count")
                return 0
            
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) as count FROM user_sessions")
                result = cursor.fetchone()
                return result['count'] if result else 0
                
            except sqlite3.Error as e:
                logger.error(f"Database error getting session count: {e}")
                return 0
            finally:
                conn.close()
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a session including timestamps.
        
        Args:
            session_id: WhatsApp sender ID
            
        Returns:
            Dict containing session info or None if not found
        """
        with self._lock:
            conn = self._get_connection()
            if not conn:
                logger.error(f"Could not connect to database for session info {session_id}")
                return None
            
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT session_id, user_state, last_updated, created_at
                    FROM user_sessions WHERE session_id = ?
                """, (session_id,))
                
                result = cursor.fetchone()
                if result:
                    return {
                        'session_id': result['session_id'],
                        'user_state': json.loads(result['user_state']),
                        'last_updated': result['last_updated'],
                        'created_at': result['created_at']
                    }
                return None
                
            except sqlite3.Error as e:
                logger.error(f"Database error getting session info for {session_id}: {e}")
                return None
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error for session {session_id}: {e}")
                return None
            finally:
                conn.close()
    
    def migrate_from_json(self, json_file_path: str = "user_states.json") -> int:
        """
        Migrate existing sessions from user_states.json to the database.
        
        Args:
            json_file_path: Path to the existing user_states.json file
            
        Returns:
            int: Number of sessions migrated
        """
        import os
        
        if not os.path.exists(json_file_path):
            logger.warning(f"JSON file not found: {json_file_path}")
            return 0
        
        try:
            with open(json_file_path, 'r') as f:
                json_data = json.load(f)
            
            migrated_count = 0
            for session_id, state in json_data.items():
                if self.save_user_state(session_id, state):
                    migrated_count += 1
                    logger.debug(f"Migrated session: {session_id}")
                else:
                    logger.error(f"Failed to migrate session: {session_id}")
            
            logger.info(f"Successfully migrated {migrated_count} sessions from {json_file_path}")
            return migrated_count
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON file {json_file_path}: {e}")
            return 0
        except Exception as e:
            logger.error(f"Unexpected error during migration: {e}")
            return 0

# Global session manager instance
session_manager = SessionManager()

# Convenience functions for backward compatibility
def get_user_state(session_id: str) -> Dict[str, Any]:
    """Get user state using the global session manager."""
    return session_manager.get_user_state(session_id)

def save_user_state(session_id: str, state: Dict[str, Any]) -> bool:
    """Save user state using the global session manager."""
    return session_manager.save_user_state(session_id, state)

def clear_user_state(session_id: str) -> bool:
    """Clear user state using the global session manager."""
    return session_manager.clear_user_state(session_id) 