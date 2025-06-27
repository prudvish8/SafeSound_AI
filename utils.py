# --- START OF NEW, CLEANED-UP utils.py ---

import sqlite3
import logging
import pandas as pd
import json
from typing import Any, Dict
from logging_config import get_logger

# Get logger for this module
logger = get_logger(__name__)

# ==============================================================================
# ========================== ESSENTIAL FUNCTIONS TO KEEP =======================
# ==============================================================================

def get_db_connection():
    """
    Establishes a connection to the SQLite database.
    Returns a database connection object.
    """
    try:
        # Ensure the DB file path is correct, consider using an environment variable
        conn = sqlite3.connect("welfare_schemes.db")
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        logger.exception("Database connection error")
        return None

def profile_summary_te(profile: dict) -> str:
    """
    Generates a human-readable profile summary in Telugu from the NLU output.
    """
    summary_lines = []
    # Use .get() to safely access keys that might not exist
    if profile.get("target") == "other" and profile.get("relation"):
        summary_lines.append(f"లబ్ధిదారుని సంబంధం: {profile.get('relation', 'N/A')}")
    
    summary_lines.append(f"వయస్సు: {profile.get('age', 'తెలియదు')}")
    summary_lines.append(f"లింగం: {profile.get('gender', 'తెలియదు')}")
    summary_lines.append(f"వృత్తి: {profile.get('occupation', 'తెలియదు')}")
    summary_lines.append(f"నెలసరి ఆదాయం: రూ. {profile.get('income', 'తెలియదు')}")
    
    ownership = profile.get('ownership')
    ownership_text = "ఉంది" if ownership is True else "లేదు" if ownership is False else "తెలియదు"
    summary_lines.append(f"సొంత ఇల్లు: {ownership_text}")
    
    return "\n".join(summary_lines)

def profile_summary_en(profile: dict) -> str:
    """
    Generates a human-readable profile summary in English from the NLU output.
    """
    summary_lines = []
    
    # Use .get() to safely access keys that might not exist in the profile
    if profile.get("target") == "other" and profile.get("relation"):
        summary_lines.append(f"Beneficiary Relation: {profile.get('relation', 'N/A').title()}")

    summary_lines.append(f"Age: {profile.get('age', 'Not specified')}")
    summary_lines.append(f"Gender: {profile.get('gender', 'Not specified').title()}")
    summary_lines.append(f"Occupation: {profile.get('occupation', 'Not specified').title()}")
    summary_lines.append(f"Caste: {profile.get('caste', 'Not specified').upper()}")
    summary_lines.append(f"Monthly Income: Rs. {profile.get('income', 'Not specified')}")
    
    ownership = profile.get('ownership')
    ownership_text = "Yes" if ownership is True else "No" if ownership is False else "Not specified"
    summary_lines.append(f"Owns a House: {ownership_text}")

    return "\n".join(summary_lines)

# --- In utils.py, replace the old match_schemes function with this one ---

def match_schemes(profile: dict) -> pd.DataFrame:
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()

    try:
        query = "SELECT name, description FROM schemes WHERE 1=1"
        params = []

        # Age Filtering
        if profile.get('age'):
            query += " AND (? >= min_age OR min_age IS NULL)"
            params.append(profile['age'])
            query += " AND (? <= max_age OR max_age IS NULL)"
            params.append(profile['age'])
        
        # Gender Filtering
        if profile.get('gender'):
            query += " AND (gender = ? OR gender = 'any' OR gender IS NULL)"
            params.append(profile['gender'])

        # --- THE INCOME LOGIC FIX ---
        # Income Filtering (Converts monthly to annual before comparing)
        if profile.get('income'):
            annual_income = profile['income'] * 12
            logger.info(f"Calculated annual income: {annual_income} for matching.")
            query += " AND (? <= max_income OR max_income IS NULL)"
            params.append(annual_income)
        
        # Occupation Filtering
        if profile.get('occupation'):
            query += " AND (occupation LIKE ? OR occupation = 'any' OR occupation IS NULL)"
            params.append(f"%{profile['occupation']}%")

        # Caste Filtering
        if profile.get('caste'):
            query += " AND (caste LIKE ? OR caste = 'any' OR caste IS NULL)"
            params.append(f"%{profile['caste']}%")
            
        # Ownership Filtering
        if profile.get('ownership') is not None:
            required_ownership = 'yes' if profile.get('ownership') else 'no'
            query += " AND (own_house = ? OR own_house = 'any' OR own_house IS NULL)"
            params.append(required_ownership)

        df = pd.read_sql_query(query, conn, params=params)
        df.rename(columns={'name': 'SchemeName', 'description': 'Description'}, inplace=True)
        return df

    except Exception as e:
        logger.exception("Error during scheme matching")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

# --- In utils.py, add this new function ---

def get_scheme_details_by_name(scheme_name: str) -> dict | None:
    """
    Retrieves all details for a specific scheme from the database.
    """
    conn = get_db_connection()
    if not conn:
        return None
    try:
        # Use a case-insensitive search to be more user-friendly
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM schemes WHERE name LIKE ?", (f'%{scheme_name}%',))
        details = cursor.fetchone() # Use fetchone since we expect a single match
        return dict(details) if details else None
    except Exception as e:
        logger.exception(f"Error fetching details for {scheme_name}")
        return None
    finally:
        if conn:
            conn.close()
def get_required_documents(scheme_name: str) -> list:
    """
    Retrieves the list of required documents for a given scheme name.
    """
    conn = get_db_connection()
    if not conn:
        return []

    try:
        # This query assumes you have a 'documents' table or similar. Adapt as needed.
        cursor = conn.cursor()
        cursor.execute("SELECT document_name FROM documents WHERE scheme_name = ?", (scheme_name,))
        documents = [row['document_name'] for row in cursor.fetchall()]
        return documents
    except Exception as e:
        logger.exception(f"Error fetching documents for {scheme_name}")
        return []
    finally:
        if conn:
            conn.close()

# --- Remove old file-based state management functions (if present) ---
# (No-op if not present)

def get_user_state(session_id: str) -> Dict[str, Any]:
    """
    Retrieves the user's session state from the sessions.db database.

    Args:
        session_id (str): The WhatsApp sender ID.

    Returns:
        dict: The user's session state as a dictionary. If not found, returns the default starting state.
    """
    default_state = {'step': 'START', 'profile': {}, 'lang': 'en'}
    try:
        conn = sqlite3.connect('sessions.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT user_state FROM user_sessions WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
        if row:
            try:
                state = json.loads(row['user_state'])
                return state
            except json.JSONDecodeError as e:
                logger.exception(f"Error decoding JSON for session {session_id}")
                return default_state
        else:
            return default_state
    except sqlite3.Error as e:
        logger.exception(f"Database error in get_user_state for {session_id}")
        return default_state
    finally:
        try:
            conn.close()
        except Exception:
            pass

def save_user_state(session_id: str, state_data: Dict[str, Any]) -> bool:
    """
    Saves the user's session state to the sessions.db database.

    Args:
        session_id (str): The WhatsApp sender ID.
        state_data (dict): The user's session state to save.

    Returns:
        bool: True if the operation was successful, False otherwise.
    """
    try:
        conn = sqlite3.connect('sessions.db')
        cursor = conn.cursor()
        state_json = json.dumps(state_data, ensure_ascii=False)
        cursor.execute(
            "INSERT OR REPLACE INTO user_sessions (session_id, user_state, last_updated) VALUES (?, ?, CURRENT_TIMESTAMP)",
            (session_id, state_json)
        )
        conn.commit()
        return True
    except sqlite3.Error as e:
        logger.exception(f"Database error in save_user_state for {session_id}")
        return False
    except (TypeError, ValueError) as e:
        logger.exception(f"Error serializing state_data for {session_id}")
        return False
    finally:
        try:
            conn.close()
        except Exception:
            pass

# --- END OF NEW, CLEANED-UP utils.py ---