# --- Contents of check_db.py ---
import sqlite3

DB_FILE = "welfare_schemes.db" # Make sure this is your DB file name

print(f"--- Checking schema for {DB_FILE} ---")
try:
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    print("\n[INFO] Schema for 'schemes' table:")
    cursor.execute("PRAGMA table_info(schemes);")
    columns = cursor.fetchall()
    
    if not columns:
        print("Could not find a table named 'schemes'.")
    else:
        for col in columns:
            print(f"  Column Name: {col[1]}, Type: {col[2]}")

    conn.close()
except Exception as e:
    print(f"\n[ERROR] An error occurred: {e}")