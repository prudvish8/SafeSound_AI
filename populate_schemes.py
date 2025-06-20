# --- Contents of the new, intelligent populate_schemes.py ---
import sqlite3
import logging
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

DB_FILE = "welfare_schemes.db"

def populate_schemes_from_rules():
    """
    Reads the eligibility_rules table and intelligently populates all the
    denormalized columns in the schemes table.
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        # Use a dictionary cursor to easily access columns by name
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get all rules, ordered by scheme for efficient processing
        cursor.execute("SELECT * FROM eligibility_rules ORDER BY scheme_id")
        all_rules = cursor.fetchall()
        
        # Group rules by scheme_id
        rules_by_scheme = {}
        for rule in all_rules:
            scheme_id = rule['scheme_id']
            if scheme_id not in rules_by_scheme:
                rules_by_scheme[scheme_id] = []
            rules_by_scheme[scheme_id].append(dict(rule))

        logging.info(f"Found rules for {len(rules_by_scheme)} unique schemes.")

        # Process each scheme's rules
        for scheme_id, rules in rules_by_scheme.items():
            logging.info(f"Processing Scheme ID: {scheme_id}")
            
            update_data = {}
            for rule in rules:
                param = rule['parameter']
                op = rule['operator']
                val = rule['value']

                # This is the "intelligent" translation logic
                if param == 'age' and op == '>=':
                    update_data['min_age'] = int(val)
                elif param == 'age' and op == '<=':
                    update_data['max_age'] = int(val)
                elif param == 'income' and op == '<=':
                    update_data['max_income'] = int(val)
                elif param == 'gender':
                    update_data['gender'] = val
                elif param == 'occupation':
                    update_data['occupation'] = val
                elif param == 'caste':
                    update_data['caste'] = val
                elif param == 'ownership' and val == 'own_house':
                    update_data['own_house'] = 'yes'
                elif param == 'ownership' and val == 'no_house':
                    update_data['own_house'] = 'no'
                # Add translations for 'is_student', 'is_employed', etc. if needed

            if not update_data:
                logging.warning(f"  -> No translatable rules found for this scheme. Skipping.")
                continue

            # Build and execute the UPDATE query
            set_clauses = [f"{key} = ?" for key in update_data.keys()]
            params = list(update_data.values())
            params.append(scheme_id)
            
            update_query = f"UPDATE schemes SET {', '.join(set_clauses)} WHERE scheme_id = ?"
            
            cursor.execute(update_query, params)
            logging.info(f"  -> Successfully updated database with: {update_data}")

        conn.commit()
        logging.info("\nDatabase population complete! All schemes have been updated.")

    except sqlite3.Error as e:
        logging.error(f"A database error occurred: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
    finally:
        if 'conn' in locals() and conn:
            conn.close()

if __name__ == '__main__':
    populate_schemes_from_rules()