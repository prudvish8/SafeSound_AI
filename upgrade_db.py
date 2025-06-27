#!/usr/bin/env python3
"""
Upgrade database schema to add missing columns.
"""

import sqlite3
from logging_config import get_logger

logger = get_logger(__name__)

def add_column_if_not_exists(conn, table_name, column_name, column_definition):
    """Add a column to a table if it doesn't already exist."""
    try:
        cursor = conn.cursor()
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}")
        logger.info(f"Added column '{column_name}' to table '{table_name}'.")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            logger.info(f"Column '{column_name}' already exists in table '{table_name}'.")
        else:
            raise

def upgrade_schema():
    """Upgrade the database schema to the final version."""
    try:
        logger.info("--- Upgrading database schema to final version ---")
        
        conn = sqlite3.connect("welfare_schemes.db")
        
        # Add missing columns to schemes table
        add_column_if_not_exists(conn, "schemes", "application_procedure", "TEXT")
        add_column_if_not_exists(conn, "schemes", "benefit_amount", "TEXT")
        add_column_if_not_exists(conn, "schemes", "contact_info", "TEXT")
        add_column_if_not_exists(conn, "schemes", "website", "TEXT")
        add_column_if_not_exists(conn, "schemes", "last_updated", "TEXT")
        
        # Add missing columns to eligibility_rules table
        add_column_if_not_exists(conn, "eligibility_rules", "rule_description", "TEXT")
        add_column_if_not_exists(conn, "eligibility_rules", "priority", "INTEGER DEFAULT 1")
        
        conn.commit()
        conn.close()
        
        logger.info("--- Schema upgrade complete ---")
        logger.warning("\nIMPORTANT: You must now manually add the data for these new columns using DB Browser for SQLite.")
        
    except Exception as e:
        logger.exception("An error occurred during schema upgrade")

if __name__ == "__main__":
    upgrade_schema()