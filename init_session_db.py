#!/usr/bin/env python3
"""
Session Database Initialization Script

This script creates the sessions.db SQLite database with the user_sessions table
to replace the file-based user_states.json for better security and performance.

Schema:
- session_id (TEXT, PRIMARY KEY) - WhatsApp sender ID
- user_state (TEXT) - JSON serialized session data
- last_updated (TIMESTAMP) - Auto-updating timestamp
- created_at (TIMESTAMP) - Session creation timestamp
"""

import sqlite3
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sessions_database():
    """
    Creates the sessions.db database with the user_sessions table.
    
    Returns:
        bool: True if successful, False otherwise
    """
    db_path = "sessions.db"
    
    try:
        # Connect to database (creates it if it doesn't exist)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create the user_sessions table
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS user_sessions (
            session_id TEXT PRIMARY KEY,
            user_state TEXT NOT NULL,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        cursor.execute(create_table_sql)
        
        # Create an index on last_updated for efficient cleanup queries
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_user_sessions_last_updated 
        ON user_sessions(last_updated)
        """)
        
        # Create a trigger to automatically update last_updated timestamp
        cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS update_user_sessions_timestamp
        AFTER UPDATE ON user_sessions
        BEGIN
            UPDATE user_sessions SET last_updated = CURRENT_TIMESTAMP
            WHERE session_id = NEW.session_id;
        END
        """)
        
        conn.commit()
        logger.info(f"Successfully created sessions database: {db_path}")
        
        # Verify the table was created
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_sessions'")
        if cursor.fetchone():
            logger.info("user_sessions table created successfully")
            
            # Show table schema for verification
            cursor.execute("PRAGMA table_info(user_sessions)")
            columns = cursor.fetchall()
            logger.info("Table schema:")
            for col in columns:
                logger.info(f"  {col[1]} ({col[2]}) - {'PRIMARY KEY' if col[5] else 'NOT NULL' if col[3] else 'NULL'}")
        else:
            logger.error("Failed to create user_sessions table")
            return False
            
        conn.close()
        return True
        
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False

def verify_database():
    """
    Verifies that the database and table exist and are accessible.
    
    Returns:
        bool: True if verification passes, False otherwise
    """
    db_path = "sessions.db"
    
    if not os.path.exists(db_path):
        logger.error(f"Database file not found: {db_path}")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_sessions'")
        if not cursor.fetchone():
            logger.error("user_sessions table not found")
            return False
        
        # Check table structure
        cursor.execute("PRAGMA table_info(user_sessions)")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]
        
        required_columns = ['session_id', 'user_state', 'last_updated', 'created_at']
        missing_columns = [col for col in required_columns if col not in column_names]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        # Test insert and select operations
        test_session_id = "test_session_123"
        test_state = '{"step": "START", "profile": {}, "lang": "en"}'
        
        # Insert test record
        cursor.execute(
            "INSERT OR REPLACE INTO user_sessions (session_id, user_state) VALUES (?, ?)",
            (test_session_id, test_state)
        )
        
        # Retrieve test record
        cursor.execute("SELECT user_state FROM user_sessions WHERE session_id = ?", (test_session_id,))
        result = cursor.fetchone()
        
        if not result or result[0] != test_state:
            logger.error("Test insert/select operations failed")
            return False
        
        # Clean up test record
        cursor.execute("DELETE FROM user_sessions WHERE session_id = ?", (test_session_id,))
        
        conn.commit()
        conn.close()
        
        logger.info("Database verification completed successfully")
        return True
        
    except sqlite3.Error as e:
        logger.error(f"Database verification error: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected verification error: {e}")
        return False

def main():
    """
    Main function to initialize the sessions database.
    """
    logger.info("Starting sessions database initialization...")
    
    # Create the database
    if not create_sessions_database():
        logger.error("Failed to create sessions database")
        return False
    
    # Verify the database
    if not verify_database():
        logger.error("Database verification failed")
        return False
    
    logger.info("Sessions database initialization completed successfully!")
    logger.info("You can now use the session_manager.py module for database-based session management.")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 