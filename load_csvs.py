#!/usr/bin/env python3
"""
Load CSV data into SQLite database for welfare schemes.
"""

import pandas as pd
import sqlite3
from logging_config import get_logger

logger = get_logger(__name__)

def load_csv_to_table(csv_file, table_name, db_file="welfare_schemes.db"):
    """Load CSV data into a SQLite table, replacing existing data."""
    try:
        df = pd.read_csv(csv_file)
        conn = sqlite3.connect(db_file)
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        conn.close()
        logger.info(f"{table_name} table refreshed from {csv_file}")
    except Exception as e:
        logger.exception(f"Error loading {csv_file} into {table_name} table")

def verify_data_integrity():
    """Verify that all referenced schemes exist in the schemes table."""
    try:
        conn = sqlite3.connect("welfare_schemes.db")
        
        # Get all scheme IDs from schemes table
        schemes_df = pd.read_sql_query("SELECT scheme_id FROM schemes", conn)
        schemes_ids = set(schemes_df['scheme_id'].tolist())
        
        # Get all scheme IDs referenced in eligibility rules
        rules_df = pd.read_sql_query("SELECT scheme_id FROM eligibility_rules", conn)
        rules_ids = set(rules_df['scheme_id'].tolist())
        
        logger.info(f'Schemes in schemes_updated.csv: {schemes_ids}')
        logger.info(f'Schemes referenced in eligibility_rules_updated.csv: {rules_ids}')
        
        # Check for rules referencing non-existent schemes
        missing_in_schemes = rules_ids - schemes_ids
        if missing_in_schemes:
            logger.warning(f'Rules referencing non-existent schemes: {missing_in_schemes}')
        else:
            logger.info('All rules reference valid schemes.')
        
        # Check for schemes with no eligibility rules
        missing_in_rules = schemes_ids - rules_ids
        if missing_in_rules:
            logger.warning(f'Schemes with NO eligibility rules: {missing_in_rules}')
        else:
            logger.info('All schemes have eligibility rules.')
            
        conn.close()
    except Exception as e:
        logger.exception("Error during data integrity verification")

if __name__ == "__main__":
    # Load all CSV files
    load_csv_to_table("schemes_updated.csv", "schemes")
    load_csv_to_table("eligibility_rules_updated.csv", "eligibility_rules")
    load_csv_to_table("documents_required.csv", "documents")
    load_csv_to_table("scheme_categories_updated.csv", "scheme_categories")
    
    # Verify data integrity
    verify_data_integrity()