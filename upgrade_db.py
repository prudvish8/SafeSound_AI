# --- Contents of upgrade_db.py ---
import sqlite3

DB_FILE = "welfare_schemes.db"

def add_column_if_not_exists(cursor, table_name, column_name, column_type):
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [info[1] for info in cursor.fetchall()]
    if column_name not in columns:
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
        print(f"Added column '{column_name}' to table '{table_name}'.")
    else:
        print(f"Column '{column_name}' already exists in table '{table_name}'.")

print("--- Upgrading database schema to final version ---")
try:
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # The complete list of filtering columns
    add_column_if_not_exists(cursor, "schemes", "min_age", "INTEGER")
    add_column_if_not_exists(cursor, "schemes", "max_age", "INTEGER")
    add_column_if_not_exists(cursor, "schemes", "gender", "TEXT")
    add_column_if_not_exists(cursor, "schemes", "max_income", "INTEGER")
    add_column_if_not_exists(cursor, "schemes", "caste", "TEXT")
    add_column_if_not_exists(cursor, "schemes", "religion", "TEXT")
    add_column_if_not_exists(cursor, "schemes", "district", "TEXT")
    add_column_if_not_exists(cursor, "schemes", "occupation", "TEXT")
    add_column_if_not_exists(cursor, "schemes", "own_house", "TEXT")
    add_column_if_not_exists(cursor, "schemes", "documents_required", "TEXT")
    add_column_if_not_exists(cursor, "schemes", "application_procedure", "TEXT")    



    conn.commit()
    conn.close()
    print("--- Schema upgrade complete ---")
    print("\nIMPORTANT: You must now manually add the data for these new columns using DB Browser for SQLite.")
except Exception as e:
    print(f"\n[ERROR] An error occurred: {e}")