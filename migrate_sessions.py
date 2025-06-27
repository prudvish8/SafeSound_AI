#!/usr/bin/env python3
"""
Session Migration Script

This script migrates existing user session data from user_states.json to the new
sessions.db database. It ensures data integrity and provides a backup of the original file.

Usage:
    python migrate_sessions.py
"""

import os
import json
import shutil
import logging
from datetime import datetime
from session_manager import SessionManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def backup_json_file(json_file_path: str = "user_states.json") -> str:
    """
    Create a backup of the original JSON file.
    
    Args:
        json_file_path: Path to the original JSON file
        
    Returns:
        str: Path to the backup file
    """
    if not os.path.exists(json_file_path):
        logger.warning(f"JSON file not found: {json_file_path}")
        return ""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{json_file_path}.backup_{timestamp}"
    
    try:
        shutil.copy2(json_file_path, backup_path)
        logger.info(f"Created backup: {backup_path}")
        return backup_path
    except Exception as e:
        logger.error(f"Failed to create backup: {e}")
        return ""

def validate_json_data(json_data: dict) -> bool:
    """
    Validate the structure of the JSON data.
    
    Args:
        json_data: The parsed JSON data
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not isinstance(json_data, dict):
        logger.error("JSON data is not a dictionary")
        return False
    
    for session_id, state in json_data.items():
        if not isinstance(session_id, str):
            logger.error(f"Invalid session ID type: {type(session_id)}")
            return False
        
        if not isinstance(state, dict):
            logger.error(f"Invalid state type for session {session_id}: {type(state)}")
            return False
        
        # Check for required fields
        if 'step' not in state:
            logger.warning(f"Session {session_id} missing 'step' field")
        
        if 'profile' not in state:
            logger.warning(f"Session {session_id} missing 'profile' field")
    
    return True

def migrate_sessions(json_file_path: str = "user_states.json") -> bool:
    """
    Migrate sessions from JSON file to database.
    
    Args:
        json_file_path: Path to the JSON file
        
    Returns:
        bool: True if migration successful, False otherwise
    """
    if not os.path.exists(json_file_path):
        logger.error(f"JSON file not found: {json_file_path}")
        return False
    
    # Create backup
    backup_path = backup_json_file(json_file_path)
    if not backup_path:
        logger.error("Failed to create backup, aborting migration")
        return False
    
    try:
        # Read JSON data
        with open(json_file_path, 'r') as f:
            json_data = json.load(f)
        
        logger.info(f"Loaded {len(json_data)} sessions from {json_file_path}")
        
        # Validate data
        if not validate_json_data(json_data):
            logger.error("JSON data validation failed")
            return False
        
        # Initialize session manager
        session_manager = SessionManager()
        
        # Migrate each session
        migrated_count = 0
        failed_count = 0
        
        for session_id, state in json_data.items():
            try:
                if session_manager.save_user_state(session_id, state):
                    migrated_count += 1
                    logger.debug(f"Migrated session: {session_id}")
                else:
                    failed_count += 1
                    logger.error(f"Failed to migrate session: {session_id}")
            except Exception as e:
                failed_count += 1
                logger.error(f"Error migrating session {session_id}: {e}")
        
        logger.info(f"Migration completed:")
        logger.info(f"  - Successfully migrated: {migrated_count}")
        logger.info(f"  - Failed migrations: {failed_count}")
        logger.info(f"  - Backup created: {backup_path}")
        
        if failed_count == 0:
            logger.info("All sessions migrated successfully!")
            return True
        else:
            logger.warning(f"{failed_count} sessions failed to migrate")
            return False
            
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON file: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during migration: {e}")
        return False

def verify_migration(json_file_path: str = "user_states.json") -> bool:
    """
    Verify that the migration was successful by comparing data.
    
    Args:
        json_file_path: Path to the original JSON file
        
    Returns:
        bool: True if verification passes, False otherwise
    """
    if not os.path.exists(json_file_path):
        logger.warning(f"JSON file not found for verification: {json_file_path}")
        return True
    
    try:
        # Read original JSON data
        with open(json_file_path, 'r') as f:
            json_data = json.load(f)
        
        # Initialize session manager
        session_manager = SessionManager()
        
        # Verify each session
        verified_count = 0
        failed_count = 0
        
        for session_id, original_state in json_data.items():
            try:
                # Get state from database
                db_state = session_manager.get_user_state(session_id)
                
                # Compare states with more flexible matching
                # Check if all original fields match in database state
                verification_passed = True
                
                for key, original_value in original_state.items():
                    if key not in db_state or db_state[key] != original_value:
                        verification_passed = False
                        logger.error(f"Field mismatch for session {session_id}, key '{key}':")
                        logger.error(f"  Original: {original_value}")
                        logger.error(f"  Database: {db_state.get(key, 'MISSING')}")
                        break
                
                if verification_passed:
                    verified_count += 1
                    logger.debug(f"Verified session: {session_id}")
                else:
                    failed_count += 1
                    logger.error(f"Verification failed for session: {session_id}")
                    
            except Exception as e:
                failed_count += 1
                logger.error(f"Error verifying session {session_id}: {e}")
        
        logger.info(f"Verification completed:")
        logger.info(f"  - Verified sessions: {verified_count}")
        logger.info(f"  - Failed verifications: {failed_count}")
        
        return failed_count == 0
        
    except Exception as e:
        logger.error(f"Error during verification: {e}")
        return False

def main():
    """
    Main migration function.
    """
    logger.info("Starting session migration from JSON to database...")
    
    # Check if database exists
    if not os.path.exists("sessions.db"):
        logger.error("Sessions database not found. Please run init_session_db.py first.")
        return False
    
    # Perform migration
    if not migrate_sessions():
        logger.error("Migration failed")
        return False
    
    # Verify migration
    if not verify_migration():
        logger.error("Migration verification failed")
        return False
    
    logger.info("Session migration completed successfully!")
    logger.info("You can now safely remove user_states.json if desired.")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 