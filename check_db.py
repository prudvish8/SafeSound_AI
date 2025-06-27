#!/usr/bin/env python3
"""
Check database schema and structure.
"""

import sqlite3
from logging_config import get_logger

logger = get_logger(__name__)

DB_FILE = "welfare_schemes.db"

def check_schema():
    """Check the schema of the database tables."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        logger.info(f"--- Checking schema for {DB_FILE} ---")
        
        # Check schemes table
        cursor.execute("PRAGMA table_info(schemes)")
        columns = cursor.fetchall()
        
        if columns:
            logger.info("\n[INFO] Schema for 'schemes' table:")
            for col in columns:
                logger.info(f"  Column Name: {col[1]}, Type: {col[2]}")
        else:
            logger.warning("Could not find a table named 'schemes'.")
            
        conn.close()
        
    except Exception as e:
        logger.exception("An error occurred while checking schema")

if __name__ == "__main__":
    check_schema()