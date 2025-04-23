import pandas as pd
import speech_recognition as sr
import unicodedata
import torch
from transformers import pipeline
import os
import sqlite3 # Import SQLite
from gtts import gTTS
import playsound
import tempfile
import warnings
import numpy as np # Needed if processing audio numpy arrays (though pipeline handles it here)
import time # For potential delays
import traceback

# --- Configuration & Setup ---

DB_FILE = "welfare_schemes.db" # SQLite database file name

# --- Global Definitions ---
# Define garbage lists and minimum length globally
GARBAGE_PHRASES_TE = [
    "సరే సరే", "ఓకే ఓకే", "థాంక్యూ", "ధన్యవాదాలు", "హమ్ హమ్", "హ్మ్ హ్మ్",
    "ఉహు ఉహు", "నమస్కారం", "హలో", "హాయ్", "ఓయ్", "ఏయ్", "ఏరా", "ఏంటి",
    "హు హు", "హూ హూ", "హు"
]

GARBAGE_PHRASES_EN = [
    "okay okay", "thank you", "thanks", "hello", "hi", "hey",
    "what", "alright", "right", "great", "fine", "well", "so"
]

# Short sounds that should only be filtered if they appear alone (not part of valid words)
SHORT_SOUNDS_TE = ["ఆహ్", "ఉమ్", "సరే", "ఆహా", "ఉహు", "ఆం", "ఆమ్", "హుఁ", "హూఁ", "ఊఁ", "అయ్యో", "ఓ", "ఆఁ"]
SHORT_SOUNDS_EN = ["uh", "um", "hmm", "ok", "aha", "uh huh", "uh uh", "oh", "ah", "mm hmm", "mmm", "huh"]

# Increased for more permissive matching
MIN_MEANINGFUL_LENGTH = 2

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.models.whisper")
warnings.filterwarnings("ignore", message="The attention mask is not set.*")
warnings.filterwarnings("ignore", message="torch.utils._pytree._register_pytree_node is deprecated.*")


# --- Function to Setup SQLite Database (Run Once) ---
def setup_database():
    """
    Creates SQLite database and tables, populates them from CSV files.
    Should ideally be run once, or checks if tables/data already exist.
    """
    required_csvs = ["schemes.csv", "eligibility_rules.csv", "documents_required.csv", "scheme_categories.csv"]
    if not all(os.path.exists(f) for f in required_csvs):
        print(f"Error: One or more required CSV files not found: {required_csvs}")
        print("Please ensure they are in the same directory for initial setup.")
        return False # Indicate setup failure

    print(f"Setting up database '{DB_FILE}'...")
    conn = None # Initialize connection variable
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # Create Tables (IF NOT EXISTS prevents errors on re-run)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS schemes (
            scheme_id INTEGER PRIMARY KEY,
            name TEXT,
            description TEXT,
            department TEXT,
            benefit_type TEXT,
            amount TEXT,
            application_url TEXT,
            contact_info TEXT,
            state TEXT,
            language TEXT,
            active_status BOOLEAN,
            last_updated TEXT
        )''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS eligibility_rules (
            rule_id INTEGER PRIMARY KEY,
            scheme_id INTEGER,
            parameter TEXT,
            operator TEXT,
            value TEXT,
            description TEXT,
            FOREIGN KEY (scheme_id) REFERENCES schemes (scheme_id)
        )''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents_required (
            doc_id INTEGER PRIMARY KEY,
            scheme_id INTEGER,
            document_name TEXT,
            is_mandatory BOOLEAN,
            FOREIGN KEY (scheme_id) REFERENCES schemes (scheme_id)
        )''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS scheme_categories (
            category_id INTEGER PRIMARY KEY,
            name TEXT
        )''')

        # --- Populate Tables from CSVs ---
        cursor.execute("SELECT COUNT(*) FROM schemes")
        if cursor.fetchone()[0] == 0:
            print("Populating tables from CSVs...")
            try:
                schemes_df = pd.read_csv("schemes.csv")
                if 'active_status' in schemes_df.columns:
                    # Ensure boolean conversion handles potential NaNs or non-boolean strings gracefully
                    schemes_df['active_status'] = schemes_df['active_status'].fillna(False).astype(str).str.lower().map({'true': True, '1': True, 'yes': True}).fillna(False)
                schemes_df.to_sql("schemes", conn, if_exists="append", index=False)

                eligibility_rules_df = pd.read_csv("eligibility_rules.csv")
                eligibility_rules_df.to_sql("eligibility_rules", conn, if_exists="append", index=False)

                documents_required_df = pd.read_csv("documents_required.csv")
                if 'is_mandatory' in documents_required_df.columns:
                    documents_required_df['is_mandatory'] = documents_required_df['is_mandatory'].fillna(False).astype(str).str.lower().map({'true': True, '1': True, 'yes': True}).fillna(False)
                documents_required_df.to_sql("documents_required", conn, if_exists="append", index=False)

                scheme_categories_df = pd.read_csv("scheme_categories.csv")
                scheme_categories_df.to_sql("scheme_categories", conn, if_exists="append", index=False)

                conn.commit()
                print("Database tables created and populated successfully.")
            except pd.errors.EmptyDataError as e:
                 print(f"Error reading CSV: {e}. Make sure CSV files are not empty.")
                 conn.rollback()
                 return False
            except Exception as e:
                 print(f"Error during data population: {e}")
                 conn.rollback()
                 return False
        else:
            print("Database tables already exist and seem populated. Skipping population.")

        return True # Indicate setup success

    except sqlite3.Error as e:
        print(f"SQLite error during setup: {e}")
        if conn: conn.rollback()
        return False
    except Exception as e:
        print(f"An unexpected error occurred during database setup: {e}")
        return False
    finally:
        if conn:
            conn.close()
            # print("Database connection closed after setup.") # Optional


# --- Whisper Model Loading ---
print("Loading Whisper model...")
try:
    # Set threads for PyTorch if needed (can sometimes help performance on CPU)
    # torch.set_num_threads(4) # Example: Limit to 4 threads
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device selected: {device}")
    whisper_pipeline = pipeline(
        task="automatic-speech-recognition",
        model="vasista22/whisper-telugu-base", # Fine-tuned Telugu model
        chunk_length_s=30,
        device=device
    )
    print("Whisper model loaded successfully.")
except Exception as e:
    print(f"Fatal Error: Could not load Whisper model: {e}")
    print("Ensure transformers and torch are installed correctly and the model is accessible.")
    exit() # Exit if core component fails


# --- START: REPLACE ONLY THIS FUNCTION IN telugu_protype_sql.py ---
# --- Helper function to parse Telugu numbers (V7 - Forced Normalization) ---
import unicodedata # *** IMPORT THIS AT THE TOP OF YOUR MAIN SCRIPT ***

def telugu_to_number(text):
    # --- Using NFC normalized keys for the map ---
    def normalize_nfc(s):
         # Normalize to NFC form, which is standard for comparisons
         return unicodedata.normalize('NFC', s)

    telugu_map_raw = { # Raw dictionary definition
        "సున్నా": 0, "ఒకటి": 1, "రెండు": 2, "మూడు": 3, "నాలుగు": 4, "ఐదు": 5,
        "ఆరు": 6, "ఏడు": 7, "ఎనిమిది": 8, "తొమ్మిది": 9, "పది": 10,
        "పదకొండు": 11, "పన్నెండు": 12, "పదమూడు": 13, "పద్నాలుగు": 14, "పదిహేను": 15,
        "పదహారు": 16, "పదిహేడు": 17, "పద్దెనిమిది": 18, "పందొమ్మిది": 19,
        "ఇరవై": 20, "ముప్పై": 30, "నలభై": 40, "యాభై": 50, "అరవై": 60,
        "డెబ్బై": 70, "ఎనభై": 80, "తొంభై": 90
    }
    # Create a map with normalized keys
    telugu_map = {normalize_nfc(k): v for k, v in telugu_map_raw.items()}

    normalizer_map_raw = {
        "కోట్ల": "కోటి", "కోట్లు": "కోటి",
        "లక్షల": "లక్ష", "లక్షా": "లక్ష", "లక్షలు": "లక్ష",
        "వేల": "వేలు", "వేలా": "వేలు", "వేయి": "వేలు", "వెయ్యి": "వేలు", "వేలాండ్": "వేలు",
        "వందల": "వంద", "వందలు": "వంద",
        "ఒక": "ఒకటి", "రెండ": "రెండు",
        "ఎనిమిదండి": "ఎనిమిది", "మూడండి": "మూడు", "నాలుగండి": "నాలుగు",
    }
    # Normalize the normalizer map's keys and values where they are number words
    normalizer_map = {normalize_nfc(k): normalize_nfc(v) for k, v in normalizer_map_raw.items()}


    multipliers_raw = {"కోటి": 10000000, "లక్ష": 100000, "వేలు": 1000, "వంద": 100}
    # Create normalized multipliers map and list
    multipliers = {normalize_nfc(k): v for k, v in multipliers_raw.items()}
    multiplier_order = [normalize_nfc(k) for k in ["కోటి", "లక్ష", "వేలు", "వంద"]]


    # --- 1. Input Cleaning and Normalization (with NFC) ---
    print(f"\n[telugu_to_number V7 DEBUG] Input: '{text}'")
    text = normalize_nfc(text.replace(",", "").replace("రూపాయలు", "").replace("సంవత్సరాలు", "").strip().lower())
    if not text: return 0
    if text.isdigit():
        try: return int(text)
        except ValueError: pass

    words = text.split()
    normalized_words = []
    for word in words:
        # Always normalize word before any checks
        nfc_word = normalize_nfc(word)

        if nfc_word in normalizer_map:
             cleaned_word = normalizer_map[nfc_word] # Get normalized value
        else:
            cleaned_word = nfc_word # Start with NFC word
            suffixes_to_remove = [normalize_nfc(s) for s in ["ండి", "అండి"]]
            for suffix in suffixes_to_remove:
                if cleaned_word.endswith(suffix):
                    base_word = cleaned_word[:-len(suffix)]
                    # Check normalized base against normalized keys
                    if base_word in telugu_map or base_word in multipliers or base_word in normalizer_map or not base_word:
                        cleaned_word = base_word
                        # Check for 'అ' artifact AFTER base_word assignment
                        if suffix == normalize_nfc("అండి") and cleaned_word == normalize_nfc("అ"):
                             cleaned_word = ""
                        break
            # Check normalizer map again after suffix removal
            if cleaned_word in normalizer_map: cleaned_word = normalizer_map[cleaned_word]

        if cleaned_word and cleaned_word != normalize_nfc("అ"):
            normalized_words.append(cleaned_word)


    print(f"Normalized words (NFC): {normalized_words}")
    if not normalized_words: return 0

    total_value = 0
    words_to_process = normalized_words[:] # Already normalized

    # --- 2. Helper Function (Uses NFC normalized map) ---
    def parse_simple_sequence(sequence):
        value = 0
        for word in sequence: # sequence words are already NFC normalized
             # print(f"  [Helper] Checking '{word}'")
             num_val = telugu_map.get(word) # Look up NFC normalized word
             if num_val is not None:
                 value += num_val
             else:
                 if not (len(sequence)==1 and word in multipliers):
                     print(f"    [Helper Warning] Word '{word}' (repr:{repr(word)}) not found in normalized telugu_map.")
        # print(f"  --> Simple sequence result: {value}")
        return value
    # --- End Helper ---

    # --- 3. Process Multipliers (Uses NFC normalized keys) ---
    for mult_word in multiplier_order: # Iterate through NFC multiplier keys
        mult_value = multipliers[mult_word]
        indices_to_remove = []
        i = 0; processed_in_this_pass = False
        while i < len(words_to_process):
            if words_to_process[i] == mult_word: # Compare normalized word with normalized key
                processed_in_this_pass = True; coefficient_words = []; coeff_indices = []; j = i - 1
                # Collect coefficients (already normalized)
                while j >= 0 and words_to_process[j] not in multipliers: # Check against normalized multiplier map
                    coefficient_words.insert(0, words_to_process[j]); coeff_indices.insert(0, j); j -= 1
                coefficient = 1
                if coefficient_words:
                    parsed_coeff = parse_simple_sequence(coefficient_words) # Use corrected helper
                    if parsed_coeff > 0: coefficient = parsed_coeff
                segment_value = coefficient * mult_value; total_value += segment_value
                print(f"  Processed segment: {coefficient_words} '{mult_word}' -> {coefficient}*{mult_value}={segment_value}. Total:{total_value}")
                indices_to_remove.extend(coeff_indices); indices_to_remove.append(i)
            i += 1
        if indices_to_remove: # Remove logic same as before
             unique_indices = sorted(list(set(indices_to_remove)), reverse=True)
             temp_words = words_to_process[:]
             for index in unique_indices:
                  if index < len(temp_words): del temp_words[index]
             words_to_process = temp_words
             if processed_in_this_pass: print(f"  Remaining words after '{mult_word}': {words_to_process}")

    # --- 4. Process Remainder ---
    if words_to_process:
        remaining_value = parse_simple_sequence(words_to_process) # Use corrected helper
        print(f"  Processed remaining {words_to_process} -> {remaining_value}")
        total_value += remaining_value

    # --- 5. Final ---
    print(f"--> Final parsed number: {total_value}")
    is_input_zero = (len(normalized_words) == 1 and normalize_nfc("సున్నా") in normalized_words[0])
    if total_value == 0 and not is_input_zero:
        try:
            digits = "".join(filter(str.isdigit, text.replace(" ",""))) # Use original text for fallback digits
            if digits: print("Fallback: Digit extraction."); return int(digits)
        except ValueError: pass
        print("Warning: Resulted in 0."); return 0
    else: return total_value
# --- END: REPLACE THE ENTIRE FUNCTION IN telugu_protype_sql.py ---


# --- Function to capture and recognize speech (Enhanced for Scheme Recognition) ---
def recognize_speech(detected_lang="te", input_method="voice"):
    r = sr.Recognizer()
    max_retries = 3
    retries = 0
    text = ""

    # Cache scheme names for recognition
    scheme_names = get_all_scheme_names()
    print(f"Loaded {len(scheme_names)} scheme names for recognition")

    while retries < max_retries:
        with sr.Microphone() as source:
            prompt = "దయచేసి మాట్లాడండి..." if detected_lang == "te" else "Please say something..."
            print(prompt)
            try:
                r.adjust_for_ambient_noise(source, duration=0.7)
                print(f"(Energy threshold: {r.energy_threshold:.2f}) Listening...")
                audio = r.listen(source, timeout=7, phrase_time_limit=20)
            except sr.WaitTimeoutError:
                print("Timeout: No speech detected.")
                retries += 1
                if retries < max_retries:
                    error_te = "క్షమించండి, నేను ఏమీ వినలేకపోయాను. దయచేసి మళ్లీ చెప్పండి."
                    error_en = "Sorry, I didn't hear anything. Please say again."
                    communicate(error_te, error_en, input_method, detected_lang)
                continue
            except Exception as e:
                 print(f"Error during listening: {e}")
                 retries += 1
                 continue

        temp_audio_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_audio.write(audio.get_wav_data())
                temp_audio_path = temp_audio.name

            print("Transcribing audio...")
            forced_decoder_ids = None
            # Force Telugu if current language is Telugu - helps bias Whisper
            if detected_lang == "te":
                 try:
                     forced_decoder_ids = whisper_pipeline.tokenizer.get_decoder_prompt_ids(language="te", task="transcribe")
                 except Exception as e:
                     print(f"Warning: Could not get forced decoder IDs for Telugu: {e}")

            # Pass the forced decoder IDs to the pipeline
            result = whisper_pipeline(temp_audio_path, generate_kwargs={"forced_decoder_ids": forced_decoder_ids})
            raw_text = result["text"].strip()
            print(f"Whisper raw output: '{raw_text}'")

            # --- Intelligent Content-Based Filtering ---
            text_to_check = raw_text.lower().strip()
            filtered = False  # Default value
            
            # 0. Empty check
            if not text_to_check:
                print("Filtered out empty transcription.")
                filtered = True
            # 1. Check for scheme names (highest priority)
            elif is_recognized_scheme(text_to_check, scheme_names):
                print(f"ACCEPTED: Contains recognized scheme name: '{raw_text}'")
                filtered = False
            # 2. Check for yes/no responses (high priority)
            elif is_yes_no_response(text_to_check, detected_lang):
                print(f"ACCEPTED: Contains yes/no response: '{raw_text}'")
                filtered = False
            # 3. Check for essential short answers (new logic)
            else:
                essential_short_te = ["ఓసి", "బిసి", "ఎస్సీ", "ఎస్టీ", "మగ", "ఆడ", "అవును", "కాదు", "లేదు", "ఉంది"]
                essential_short_en = ["oc", "bc", "sc", "st", "yes", "no", "male", "female"]
                essentials = essential_short_te if detected_lang == 'te' else essential_short_en
                if text_to_check in essentials:
                    print(f"ACCEPTED: Kept essential short answer: '{raw_text}'")
                    filtered = False
                # 4. Check for exact garbage phrase matches
                elif is_exact_garbage_match(text_to_check, detected_lang):
                    print(f"Filtered out exact garbage phrase: '{raw_text}'")
                    filtered = True
                # 5. Check for standalone short sounds (only if very short)
                elif len(text_to_check) < 5 and is_short_sound(text_to_check, detected_lang):
                    print(f"Filtered out short sound: '{raw_text}'")
                    filtered = True
                # 6. Length check after other checks
                elif len(text_to_check) < MIN_MEANINGFUL_LENGTH:
                    print(f"Filtered out too short transcription: '{raw_text}'")
                    filtered = True
                # 7. Accept by default if passed all checks
                else:
                    print(f"ACCEPTED by default: '{raw_text}'")
                    filtered = False

            if filtered:
                retries += 1
                if retries < max_retries:
                    retry_te = "క్షమించండి, నేను సరిగ్గా వినలేకపోయాను. దయచేసి స్పష్టంగా మళ్లీ చెప్పండి."
                    retry_en = "Sorry, I didn't quite catch that. Please say again clearly."
                    communicate(retry_te, retry_en, input_method, detected_lang)
                continue # Go back to listen again
            else:
                # Successfully got non-garbage text
                text = raw_text
                # Re-estimate language based on actual *recognized* text
                is_telugu = any(0x0c00 <= ord(char) <= 0x0c7f for char in text)
                # Bias towards current language if unicode check is ambiguous (e.g., only english loan words)
                detected_language = "te" if is_telugu else ("en" if any('a' <= char.lower() <= 'z' for char in text) else detected_lang)
                print(f"Whisper recognized (accepted): {text} (Detected as: {detected_language})")
                return text, detected_language # Return recognized text and detected language

        except sr.RequestError as e:
            print(f"Could not request results from Whisper/SR service; {e}")
            service_error_te = "క్షమించండి, స్పీచ్ రికగ్నిషన్ సేవతో సమస్య ఉంది."
            service_error_en = "Sorry, there was an issue with the speech recognition service."
            communicate(service_error_te, service_error_en, input_method, detected_lang)
            return "", detected_lang # Return empty on error
        except Exception as e:
            print(f"Error during transcription or file handling: {e}")
            retries += 1
            if retries < max_retries:
                process_error_te = "క్షమించండి, మీ ఆడియోను ప్రాసెస్ చేయడంలో లోపం ఉంది. దయచేసి మళ్లీ ప్రయత్నించండి."
                process_error_en = "Sorry, an error occurred processing your audio. Please try again."
                communicate(process_error_te, process_error_en, input_method, detected_lang)
            continue # Try again if possible
        finally:
            # Ensure temporary file is always deleted
            if temp_audio_path and os.path.exists(temp_audio_path):
                try: os.remove(temp_audio_path)
                except Exception as e: print(f"Warning: Could not delete temp file {temp_audio_path}: {e}")

    # Reached max retries
    print("Maximum retries reached for speech recognition.")
    # Avoid speaking again, just return empty
    return "", detected_lang


# --- Helper Functions for Text Filtering ---
def get_all_scheme_names():
    """Get all scheme names from the database for recognition purposes."""
    scheme_names = []
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM schemes WHERE active_status = 1")
        scheme_names = [row[0].lower() for row in cursor.fetchall() if row[0]]
        conn.close()
    except Exception as e:
        print(f"Error fetching scheme names: {e}")
    
    # Add common transliterations and variations
    variations = []
    for name in scheme_names:
        # Add variation without spaces
        variations.append(name.replace(" ", ""))
        
    scheme_names.extend(variations)
    
    # Add hardcoded common schemes for fallback
    common_schemes = [
        "ఆరోగ్యశ్రీ", "అమ్మ ఒడి", "ఎన్టీఆర్ రైతు భరోసా", "విద్యా దీవెన", "అసరా పెన్షన్",
        "aarogyasri", "amma odi", "ntr rythu bharosa", "vidya deevena", "asara pension",
        "arogyasri", "ammavodi", "ntr raithu bharosa", "vidya devena", "asara pension"
    ]
    
    # Combine all schemes, remove duplicates, and ensure all lowercase
    all_schemes = list(set([s.lower() for s in scheme_names + common_schemes]))
    return all_schemes


def is_recognized_scheme(text, scheme_names):
    """Check if text contains any recognized scheme name."""
    text_lower = text.lower()
    
    # Direct match first (more precise)
    if text_lower in scheme_names:
        return True
        
    # Check for substring match (fuzzier)
    for scheme in scheme_names:
        if len(scheme) > 3 and scheme in text_lower:
            return True
            
    return False


def is_yes_no_response(text, detected_lang):
    """Check if text contains yes/no response."""
    text_lower = text.lower()
    
    yes_keywords = ["yes", "అవును", "యెస్", "అవునండి", "ఒకే", "సరే", "కరెక్ట్", "correct"]
    no_keywords = ["no", "కాదు", "లేదు", "నో", "వద్దు", "కాదండి"]
    
    for keyword in yes_keywords + no_keywords:
        if keyword == text_lower or keyword in text_lower.split():
            return True
            
    return False


def is_exact_garbage_match(text, detected_lang):
    """Check if text exactly matches a garbage phrase."""
    garbage_list = GARBAGE_PHRASES_TE if detected_lang == 'te' else GARBAGE_PHRASES_EN
    return text.lower() in garbage_list


def is_short_sound(text, detected_lang):
    """Check if text is just a short sound or filler word (revised to avoid blocking numbers)."""
    short_sounds = SHORT_SOUNDS_TE if detected_lang == 'te' else SHORT_SOUNDS_EN
    text_lower = text.lower()

    # Don't filter if it's just digits (like age '19', '20')
    if text_lower.isdigit():
        return False

    # Check for exact match with short sounds
    if any(sound == text_lower for sound in short_sounds):
        return True

    # Check for repeated sounds (e.g., "hu hu") - ensure it's not just digits
    words = text_lower.split()
    # *** CORRECTED LOGIC ***
    # Only filter if there are exactly two identical short words
    if len(words) == 2 and len(set(words)) == 1 and len(words[0]) <= 2 and not words[0].isdigit():
        print(f"Detected repeated short sound: '{text_lower}'")
        return True

    return False


# --- Function to get user input based on the input method ---
def get_user_input(input_method, language="te"):
    """
    Gets user input based on the selected input method.
    
    Args:
        input_method (str): The input method ('keyboard' or 'voice')
        language (str): The language to use ('en' or 'te'), defaults to 'te'
    
    Returns:
        str: The user's input text
    """
    if input_method == "voice":
        # Use speech recognition for voice input
        text, detected_lang = recognize_speech(language, input_method)
        return text
    else:
        # Use keyboard input
        user_input = input("> ").strip()
        return user_input


# --- Function to build user profile (Checks suspicious data, handles confirmation) ---
def build_user_profile(language="te", input_method="voice", is_for_others=False):
    """
    Build a user profile by asking a series of questions and processing the responses.
    Includes special flow for students <= 21, handles "I don't know", adds caste.

    Args:
        language (str): The language to use ('te' for Telugu, 'en' for English)
        input_method (str): Input method to use ('voice' or 'keyboard')
        is_for_others (bool): Whether the profile is for someone else

    Returns:
        dict: A dictionary containing the user's profile information or None if failed/aborted
    """
    profile = {
        "gender": None,
        "age": None,
        "caste": None, # NEW FIELD
        "income": None,
        "occupation": None, # Will store 'unemployed' or 'student'
        "is_employed": None, # Might be None if student path taken
        "owns_house": None,
        "is_student": None # Explicitly set based on questions
    }

    # Keywords for "I don't know"
    dont_know_keywords_te = ["తెలియదు", "చెప్పలేను", "రహస్యం", "గుర్తు లేదు"]
    dont_know_keywords_en = ["don't know", "unknown", "private", "confidential", "not sure", "i don't know", "no idea"]

    def check_dont_know(text):
        """Checks if the text indicates 'I don't know'."""
        if not text: return False
        text_lower = text.lower()
        keywords = dont_know_keywords_te if language == 'te' else dont_know_keywords_en
        return any(kw in text_lower for kw in keywords)

    # Function to validate inputs and filter garbage
    def validate_input(input_text, min_length=3):
        if not input_text:
            return False

        # Check for very short inputs (excluding yes/no and "I don't know")
        # Allow single/double digits for age/caste examples
        allowed_short = is_yes_no_response(input_text, language) or check_dont_know(input_text) or input_text.isdigit()
        if len(input_text) <= min_length and not allowed_short:
            print(f"Received short input: '{input_text}', length: {len(input_text)}")
            return False

        # Check for garbage phrases (using the fixed is_short_sound)
        if is_exact_garbage_match(input_text, language) or is_short_sound(input_text, language):
            print(f"Received garbage input: '{input_text}'")
            return False

        return True

    # Questions adjusted based on whether it's for self or others
    pronoun_te = "మీ" if not is_for_others else "వారి"
    pronoun_en = "your" if not is_for_others else "their"

    verb_te = "మీరు" if not is_for_others else "వారు"
    verb_en = "you" if not is_for_others else "they"

    # Shared retry logic - NEW HELPER FUNCTION
    def ask_question_with_retry(question_te, question_en, profile_key, extractor_func, validator_args={'min_length': 1}, validation_func=None):
        """
        Helper to ask a question, handle retries, "I don't know", and validation.

        Returns:
            bool: True if a valid value (or None for 'don't know') was obtained, False otherwise.
        """
        max_retries = 2
        retries = 0
        while retries <= max_retries:
            communicate(question_te, question_en, input_method, language)
            answer = get_user_input(input_method, language)

            if check_dont_know(answer):
                okay_te = "సరే, ఫర్వాలేదు."
                okay_en = "Okay, that's alright."
                communicate(okay_te, okay_en, input_method, language)
                profile[profile_key] = None # Explicitly set to None for 'don't know'
                # If the key is occupation, and they don't know, maybe set is_employed to None too?
                # This specific handling might be better placed outside the helper after the call returns.
                # if profile_key == "occupation":
                #     profile["is_employed"] = None # Unsure about employment if occupation unknown
                return True # Successfully handled 'don't know'

            # Pass validator_args correctly
            if not validate_input(answer, **validator_args):
                if retries < max_retries:
                    retry_q_te = "క్షమించండి, మీ జవాబు అర్థం కాలేదు. మళ్ళీ చెప్పగలరా?"
                    retry_q_en = "Sorry, I didn't understand your answer. Could you please repeat?"
                    communicate(retry_q_te, retry_q_en, input_method, language)
                    retries += 1
                    continue
                else:
                    print(f"Max retries reached for {profile_key} due to invalid input.")
                    profile[profile_key] = None # Failed to get valid input
                    return False # Failed after retries

            # Use extractor if provided, otherwise use the raw answer (cleaned)
            extracted_value = extractor_func(answer, language) if extractor_func else answer.strip()

            # Perform validation if a validation function is provided
            if validation_func:
                is_valid, error_te, error_en = validation_func(extracted_value)
                if is_valid:
                    profile[profile_key] = extracted_value
                    return True # Success
                else:
                     if retries < max_retries:
                        communicate(error_te, error_en, input_method, language)
                        retries += 1
                     else:
                         print(f"Max retries reached for {profile_key} due to validation failure.")
                         profile[profile_key] = None
                         return False # Failed after retries
            # Handle cases where extractor returns None (e.g., yes/no, gender not recognized)
            elif extractor_func and extracted_value is None:
                 if retries < max_retries:
                    # Generic error if extractor failed
                    retry_q_te = f"క్షమించండి, {profile_key} సమాచారాన్ని సరిగ్గా గుర్తించలేకపోయాను. దయచేసి మళ్లీ చెప్పండి."
                    retry_q_en = f"Sorry, I couldn't recognize the {profile_key} information correctly. Please say again."
                    communicate(retry_q_te, retry_q_en, input_method, language)
                    retries += 1
                 else:
                    print(f"Max retries reached for {profile_key} due to extraction failure.")
                    profile[profile_key] = None
                    return False # Failed after retries
            # If no validation func and extractor didn't fail (or no extractor used)
            else:
                 profile[profile_key] = extracted_value # Store the extracted (or raw validated) value
                 return True # Success

        print(f"Exiting retry loop for {profile_key} without success.")
        profile[profile_key] = None # Ensure it's None if loop finishes without success
        return False # Failed after retries


    # --- Start Asking Questions ---

    # 1. Age
    age_q_te = f"{pronoun_te} వయస్సు ఎంత?"
    age_q_en = f"What is {pronoun_en} age?"
    def validate_age(age):
        # Check if age is None first (could happen if extractor fails)
        if age is None:
             err_te = "వయస్సు సంఖ్య రూపంలో చెప్పండి."
             err_en = "Please provide the age as a number."
             return False, err_te, err_en
        if 0 < age < 120:
            return True, None, None
        else:
            err_te = "క్షమించండి, చెప్పిన వయస్సు సరైనది కాదు. దయచేసి 1-120 మధ్య వయస్సు చెప్పండి."
            err_en = "Sorry, the age mentioned is not valid. Please provide an age between 1-120."
            return False, err_te, err_en
    if not ask_question_with_retry(age_q_te, age_q_en, "age",
                                  (lambda a, l: telugu_to_number(a) if l == 'te' else extract_number(a)),
                                  {'min_length': 1}, validate_age): # Use min_length 1 for age
        print("Failed to get valid age.")
        # return None # Abort if age is mandatory


    # --- Conditional Flow Based on Age ---
    student_path = False
    if profile["age"] and profile["age"] <= 21:
        # 2a. Ask if student (only if age <= 21)
        student_q_te = f"{verb_te} విద్యార్థి/విద్యార్థినా? (అవును/కాదు)"
        student_q_en = f"Are {verb_en} a student? (Yes/No)"
        communicate(student_q_te, student_q_en, input_method, language)
        student_answer = get_user_input(input_method, language)

        if check_dont_know(student_answer):
             profile["is_student"] = None
             communicate("సరే, ఫర్వాలేదు.", "Okay, that's alright.", input_method, language)
             # Proceed with normal flow if student status unknown
        elif student_answer and validate_input(student_answer, min_length=1): # Yes/No needs min_length 1
            is_student = extract_yes_no(student_answer, language)
            profile["is_student"] = is_student

            if is_student:
                student_path = True # Take the student-specific path
                profile["occupation"] = "student"
                profile["is_employed"] = False
                print("Identified as student <= 21. Taking simplified path.")
                # Optional: Inform about student schemes here if desired
                # student_msg_te = "యువ విద్యార్థుల కోసం ప్రత్యేక పథకాలు ఉన్నాయి..."
                # student_msg_en = "There are specific schemes for young students..."
                # communicate(student_msg_te, student_msg_en, input_method, language)

                # Ask Caste (needed for student path)
                caste_q_te = f"{pronoun_te} కులం ఏమిటి? (ఉదా: OC, BC, SC, ST)"
                caste_q_en = f"What is {pronoun_en} caste category? (e.g., OC, BC, SC, ST)"
                if not ask_question_with_retry(caste_q_te, caste_q_en, "caste", extract_caste, {'min_length': 2}):
                     print("Failed to get caste for student.") # Handle failure

            # Else (is_student is False): student_path remains False, continue with normal flow below
        else:
             # Invalid input for student question, treat as unknown student status
             profile["is_student"] = None
             print("Could not determine student status.")
             # Proceed with normal flow

    # --- Normal Flow (or if student_path is False) ---
    if not student_path:
        # 3. Gender (if not already asked in student path)
        if profile["gender"] is None: # Check if gender was already collected
             gender_q_te = f"{pronoun_te} లింగం ఏమిటి? (మగ/ఆడ)"
             gender_q_en = f"What is {pronoun_en} gender? (Male/Female)"
             if not ask_question_with_retry(gender_q_te, gender_q_en, "gender", extract_gender, {'min_length': 2}):
                 print("Failed to get valid gender.")

        # 4. Caste (NEW - Ask in normal flow too)
        caste_q_te = f"{pronoun_te} కులం ఏమిటి? (ఉదా: OC, BC, SC, ST)"
        caste_q_en = f"What is {pronoun_en} caste category? (e.g., OC, BC, SC, ST)"
        if not ask_question_with_retry(caste_q_te, caste_q_en, "caste", extract_caste, {'min_length': 2}):
             print("Failed to get caste.")

        # 5. Employment Status
        employed_q_te = f"{verb_te} ప్రస్తుతం ఉద్యోగం చేస్తున్నారా లేదా ఏదైనా పని చేస్తున్నారా? (అవును/కాదు)"
        employed_q_en = f"Are {verb_en} currently employed or working? (Yes/No)"
        ask_question_with_retry(employed_q_te, employed_q_en, "is_employed", extract_yes_no, {'min_length': 1})

        # 6. Occupation (Logic depends on employment status)
        if profile["is_employed"] is False:
            profile["occupation"] = "unemployed"
            print("Marked occupation as 'unemployed'.")
            # If unemployed, set is_student to False if it's currently None
            if profile["is_student"] is None: profile["is_student"] = False
        elif profile["is_employed"] is True:
            occ_q_te = f"{pronoun_te} వృత్తి/ఉద్యోగం ఏమిటి? (ఉదా: రైతు, గృహిణి, ఉపాధ్యాయుడు)"
            occ_q_en = f"What is {pronoun_en} occupation or profession? (e.g., farmer, homemaker, teacher)"
            if ask_question_with_retry(occ_q_te, occ_q_en, "occupation", extract_occupation, {'min_length': 2}):
                # Check if occupation indicates student status, if not already set
                if profile["is_student"] is None and profile["occupation"] and profile["occupation"] != 'unemployed':
                     if check_is_student(profile["occupation"], language):
                          profile["is_student"] = True
                          print("Marked as student based on occupation.")
                     else:
                          profile["is_student"] = False # If employed in non-student job, likely not a student
            else:
                 print("Failed to get valid occupation details.")
                 # If they didn't know occupation, set is_employed back to None?
                 # Or keep is_employed=True and occupation=None? Current: occupation=None
        else: # is_employed is None
            profile["occupation"] = None
            print("Employment status unknown, cannot determine occupation.")
            # If employment unknown, student status also remains unknown unless set earlier


        # 7. Monthly Income (With examples and range fallback)
        income_q_te = f"{pronoun_te} నెలవారీ ఆదాయం ఎంత? (ఉదా: 10,000, 1 లక్ష 50 వేలు. తెలియకపోతే సుమారుగా చెప్పండి లేదా 'తెలియదు' అని చెప్పండి)"
        income_q_en = f"What is {pronoun_en} monthly income? (e.g., 10,000, 1 lakh 50 thousand. If unsure, give an estimate or say 'I don't know')"
        # ... (keep the corrected income loop from previous step) ...
        # Custom loop for income to handle the range fallback on "I don't know"
        max_retries = 2
        retries = 0
        income_set = False
        while retries <= max_retries and not income_set:
            communicate(income_q_te, income_q_en, input_method, language)
            income_answer = get_user_input(input_method, language)

            if check_dont_know(income_answer):
                 okay_te = "సరే, ఫర్వాలేదు."
                 okay_en = "Okay, that's alright."
                 communicate(okay_te, okay_en, input_method, language)
                 profile["income"] = None
                 # Ask for range as a fallback
                 range_q_te = "సుమారుగా రేంజ్ చెప్పగలరా? ఉదా: 10,000 లోపు, 10-30 వేల మధ్య, 30 వేల పైన?"
                 range_q_en = "Can you provide an approximate range? e.g., Below 10,000, Between 10-30 thousand, Above 30 thousand?"
                 communicate(range_q_te, range_q_en, input_method, language)
                 range_answer = get_user_input(input_method, language)
                 if range_answer and validate_input(range_answer, min_length=1):
                     income_range = extract_income_range(range_answer, language)
                     if income_range:
                         profile["income"] = income_range # Store range instead
                         print(f"Storing income range: {income_range}")
                     else:
                         profile["income"] = None # Couldn't parse range either
                 income_set = True # Mark as set after handling 'don't know'
            elif not validate_input(income_answer, min_length=1): # Income could be single digit
                # Handle invalid input / retry
                if retries < max_retries:
                    retry_q_te = "క్షమించండి, మీ జవాబు అర్థం కాలేదు. మళ్ళీ చెప్పగలరా?"
                    retry_q_en = "Sorry, I didn't understand your answer. Could you please repeat?"
                    communicate(retry_q_te, retry_q_en, input_method, language)
                    retries += 1
                    # continue - implicit continue at end of loop iteration
                else:
                     print("Max retries reached for income due to invalid input.")
                     profile["income"] = None
                     break # Exit loop on max retries for invalid input
            else:
                # Input is valid and not "I don't know", try parsing as number
                income = telugu_to_number(income_answer) if language == "te" else extract_number(income_answer)
                if income is not None: # Includes 0, which might be valid
                     # Basic validation for very high values, adjust threshold as needed
                     if income < 5000000: # e.g., less than 50 lakhs monthly
                         profile["income"] = income
                         income_set = True # Mark as set
                         # break - not needed due to loop condition `and not income_set`
                     else:
                          # Handle high value validation / retry
                          if retries < max_retries:
                              income_error_te = "క్షమించండి, ఆ ఆదాయం చాలా ఎక్కువగా అనిపిస్తోంది. దయచేసి మళ్ళీ సరిచూసి చెప్పండి."
                              income_error_en = "Sorry, that income seems very high. Please double-check and provide it again."
                              communicate(income_error_te, income_error_en, input_method, language)
                              retries += 1
                          else:
                              print("Max retries reached for income due to high value validation.")
                              profile["income"] = None
                              break # Exit loop
                else: # Could not parse as number
                    # Handle parsing error / retry
                    if retries < max_retries:
                        income_error_te = "క్షమించండి, ఆదాయం అర్థం కాలేదు. దయచేసి సంఖ్య రూపంలో చెప్పండి లేదా 'తెలియదు' అని చెప్పండి."
                        income_error_en = "Sorry, I couldn't understand the income. Please provide it as a number or say 'I don't know'."
                        communicate(income_error_te, income_error_en, input_method, language)
                        retries += 1
                    else:
                         print("Max retries reached for income due to parsing failure.")
                         profile["income"] = None
                         break # Exit loop


        # 8. House Ownership
        house_q_te = f"{pronoun_te} స్వంత ఇల్లు ఉందా? (అవును/కాదు)"
        house_q_en = f"Do {verb_en} own a house? (Yes/No)"
        # Reuse ask_question_with_retry helper
        if not ask_question_with_retry(house_q_te, house_q_en, "owns_house", extract_yes_no, {'min_length': 1}):
             print("Failed to get house ownership status.")


    # --- End of Conditional Flow ---


    # 9. Preview the collected profile and confirm (Updated Summary)
    preview_te = "నేను సేకరించిన సమాచారం:\n"
    preview_te += f"వయస్సు: {profile.get('age', 'తెలియదు')}\n"
    # --- Conditionally add Gender --- 
    if not student_path:
        preview_te += f"లింగం: {profile.get('gender', 'తెలియదు')}\n"
    # -----------------------------
    preview_te += f"కులం: {profile.get('caste', 'తెలియదు')}\n" # Added Caste

    # Only show student status if it was determined
    if profile.get('is_student') is not None:
        preview_te += f"విద్యార్థి: {'అవును' if profile['is_student'] else 'కాదు'}\n"

    # Only show employment/occupation/income/house if NOT on the student path
    if not student_path:
         # Display employment status if known
         if profile.get('is_employed') is not None:
              preview_te += f"ఉద్యోగ స్థితి: {'ఉద్యోగం ఉంది' if profile['is_employed'] else 'ఉద్యోగం లేదు'}\n"
         # Display occupation if known (and not implicitly 'student')
         if profile.get('occupation') and profile['occupation'] != 'student':
              preview_te += f"వృత్తి: {profile['occupation']}\n"
         # Display income if known
         if profile.get('income') is not None:
              preview_te += f"నెలవారీ ఆదాయం: {profile['income']}\n"
         # Display house ownership if known
         if profile.get('owns_house') is not None:
              preview_te += f"స్వంత ఇల్లు: {'అవును' if profile['owns_house'] else 'కాదు'}\n"

    preview_te += "\nఈ సమాచారం సరైనదేనా? (అవును/కాదు)"


    preview_en = "Here's the information I collected:\n"
    preview_en += f"Age: {profile.get('age', 'Unknown')}\n"
    # --- Conditionally add Gender ---
    if not student_path:
        preview_en += f"Gender: {profile.get('gender', 'Unknown')}\n"
    # -----------------------------
    preview_en += f"Caste: {profile.get('caste', 'Unknown')}\n" # Added Caste

    # Only show student status if it was determined
    if profile.get('is_student') is not None:
        preview_en += f"Student: {'Yes' if profile['is_student'] else 'No'}\n"

    # Only show employment/occupation/income/house if NOT on the student path
    if not student_path:
         # Display employment status if known
         if profile.get('is_employed') is not None:
             preview_en += f"Employment Status: {'Employed' if profile['is_employed'] else 'Unemployed'}\n"
         # Display occupation if known (and not implicitly 'student')
         if profile.get('occupation') and profile['occupation'] != 'student':
              preview_en += f"Occupation: {profile['occupation']}\n"
         # Display income if known
         if profile.get('income') is not None:
              preview_en += f"Monthly Income: {profile['income']}\n"
         # Display house ownership if known
         if profile.get('owns_house') is not None:
              preview_en += f"Owns House: {'Yes' if profile['owns_house'] else 'No'}\n"

    preview_en += "\nIs this information correct? (Yes/No)"

    communicate(preview_te, preview_en, input_method, language)

    # Confirmation and Correction Logic (Needs update for Caste and conditional fields)
    confirmation = get_user_input(input_method, language)
    if confirmation:
        confirm_value = extract_yes_no(confirmation, language)
        if confirm_value == False:
            # Ask which field needs correction (include caste)
            correction_q_te = "దయచేసి సరిచేయాల్సిన అంశం చెప్పండి (వయస్సు/లింగం/కులం/ఉద్యోగం/ఆదాయం/ఇల్లు/విద్యార్థి)."
            correction_q_en = "Which information is incorrect? Please specify (age/gender/caste/employment/income/house/student)."
            communicate(correction_q_te, correction_q_en, input_method, language)

            correction_field_input = get_user_input(input_method, language)
            if correction_field_input:
                # --- Simplified Correction Handling: Restart Profile Build ---
                # Mapping and re-asking specific fields adds complexity, especially with conditional paths.
                # For now, restarting is the most robust way to handle corrections.
                retry_te = "అర్థమైంది. దయచేసి మళ్లీ మొత్తం సమాచారాన్ని నమోదు చేద్దాం."
                retry_en = "I understand. Let's input all information again."
                communicate(retry_te, retry_en, input_method, language)
                return build_user_profile(language, input_method, is_for_others) # Recursively retry

    print("\n--- Final Profile ---")
    print(profile)
    print("---------------------\n")
    return profile


# --- Function to match schemes for user profile (ENHANCED WITH EXPLANATIONS) ---
def match_schemes(user_profile):
    """Match welfare schemes based on user profile with detailed eligibility reasoning."""
    conn = None
    matched_schemes = []
    ineligible_schemes = {} # Track schemes and why they're ineligible
    
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        # Get all active schemes
        cursor.execute("SELECT scheme_id, name, description FROM schemes WHERE active_status = 1")
        active_schemes = cursor.fetchall()

        # Process all active schemes
        print(f"\n--- Matching {len(active_schemes)} active schemes ---")
        
        for scheme_id, scheme_name, scheme_desc in active_schemes:
            # Get eligibility rules for this scheme
            cursor.execute("SELECT parameter, operator, value, description FROM eligibility_rules WHERE scheme_id = ?", (scheme_id,))
            rules = cursor.fetchall()
            
            if not rules:
                print(f"Warning: No eligibility rules found for scheme {scheme_id}: {scheme_name}")
                continue
                
            # Track eligibility status and rule failures
            eligible = True
            rule_failures = {"en": [], "te": []}
            skip_reason = None
            
            # Print current scheme being checked
            print(f"\nChecking eligibility for scheme {scheme_id}: {scheme_name}")
            
            for rule_param, operator, rule_value_str, rule_desc in rules:
                key = rule_param.lower() # Match against normalized profile keys
                user_value = user_profile.get(key)
                
                # If user didn't provide a value for this rule parameter
                if user_value is None:
                    print(f"  - Unknown: User didn't provide {key}")
                    skip_reason = f"Missing information for {key}"
                    continue # Don't make a decision based on unknown data
                
                try:
                    # Convert to comparable types for specific parameters
                    if key == "income" or key == "age":
                        # For numeric comparisons
                        user_num = int(user_value)
                        rule_num = int(rule_value_str)
                        
                        if operator == "<":
                            if not (user_num < rule_num):
                                eligible = False
                                print(f"  - Not eligible: {key} {user_num} is not < {rule_num}")
                                
                                # Add in both languages
                                if key == "income":
                                    rule_failures["en"].append(f"Income ({user_num}) is too high (must be less than {rule_num})")
                                    rule_failures["te"].append(f"ఆదాయం ({user_num}) చాలా ఎక్కువగా ఉంది ({rule_num} కంటే తక్కువ ఉండాలి)")
                                elif key == "age":
                                    rule_failures["en"].append(f"Age ({user_num}) is too high (must be less than {rule_num})")
                                    rule_failures["te"].append(f"వయస్సు ({user_num}) చాలా ఎక్కువగా ఉంది ({rule_num} కంటే తక్కువ ఉండాలి)")
                                    
                        elif operator == "<=":
                            if not (user_num <= rule_num):
                                eligible = False
                                print(f"  - Not eligible: {key} {user_num} is not <= {rule_num}")
                                
                                if key == "income":
                                    rule_failures["en"].append(f"Income ({user_num}) is too high (must be {rule_num} or less)")
                                    rule_failures["te"].append(f"ఆదాయం ({user_num}) చాలా ఎక్కువగా ఉంది ({rule_num} లేదా తక్కువ ఉండాలి)")
                                elif key == "age":
                                    rule_failures["en"].append(f"Age ({user_num}) is too high (must be {rule_num} or less)")
                                    rule_failures["te"].append(f"వయస్సు ({user_num}) చాలా ఎక్కువగా ఉంది ({rule_num} లేదా తక్కువ ఉండాలి)")
                                    
                        elif operator == ">":
                            if not (user_num > rule_num):
                                eligible = False
                                print(f"  - Not eligible: {key} {user_num} is not > {rule_num}")
                                
                                if key == "income":
                                    rule_failures["en"].append(f"Income ({user_num}) is too low (must be greater than {rule_num})")
                                    rule_failures["te"].append(f"ఆదాయం ({user_num}) చాలా తక్కువగా ఉంది ({rule_num} కంటే ఎక్కువ ఉండాలి)")
                                elif key == "age":
                                    rule_failures["en"].append(f"Age ({user_num}) is too low (must be greater than {rule_num})")
                                    rule_failures["te"].append(f"వయస్సు ({user_num}) చాలా తక్కువగా ఉంది ({rule_num} కంటే ఎక్కువ ఉండాలి)")
                                    
                        elif operator == ">=":
                            if not (user_num >= rule_num):
                                eligible = False
                                print(f"  - Not eligible: {key} {user_num} is not >= {rule_num}")
                                
                                if key == "income":
                                    rule_failures["en"].append(f"Income ({user_num}) is too low (must be {rule_num} or more)")
                                    rule_failures["te"].append(f"ఆదాయం ({user_num}) చాలా తక్కువగా ఉంది ({rule_num} లేదా ఎక్కువ ఉండాలి)")
                                elif key == "age":
                                    rule_failures["en"].append(f"Age ({user_num}) is too low (must be {rule_num} or more)")
                                    rule_failures["te"].append(f"వయస్సు ({user_num}) చాలా తక్కువగా ఉంది ({rule_num} లేదా ఎక్కువ ఉండాలి)")
                                    
                        elif operator == "=":
                            if not (user_num == rule_num):
                                eligible = False
                                print(f"  - Not eligible: {key} {user_num} is not = {rule_num}")
                                
                                if key == "income":
                                    rule_failures["en"].append(f"Income must be exactly {rule_num} (you have {user_num})")
                                    rule_failures["te"].append(f"ఆదాయం ఖచ్చితంగా {rule_num} ఉండాలి (మీది {user_num})")
                                elif key == "age":
                                    rule_failures["en"].append(f"Age must be exactly {rule_num} (you are {user_num})")
                                    rule_failures["te"].append(f"వయస్సు ఖచ్చితంగా {rule_num} ఉండాలి (మీది {user_num})")
                                    
                        elif operator == "!=":
                            if not (user_num != rule_num):
                                eligible = False
                                print(f"  - Not eligible: {key} {user_num} is not != {rule_num}")
                                
                                if key == "income":
                                    rule_failures["en"].append(f"Income must not be {rule_num} (you have {user_num})")
                                    rule_failures["te"].append(f"ఆదాయం {rule_num} కాకూడదు (మీది {user_num})")
                                elif key == "age":
                                    rule_failures["en"].append(f"Age must not be {rule_num} (you are {user_num})")
                                    rule_failures["te"].append(f"వయస్సు {rule_num} కాకూడదు (మీది {user_num})")
                                    
                    else:
                        # For string comparisons (e.g. occupation, gender)
                        user_str = str(user_value).lower().strip()
                        rule_str = str(rule_value_str).lower().strip()
                        if operator == "=":
                            # For occupation, consider more complex matching
                            if key == "occupation":
                                # Exact match, OR rule string is contained within user input for occupation
                                if user_str != rule_str and rule_str not in user_str:
                                    print(f"  - Occupation mismatch: '{user_str}' does not match or contain '{rule_str}'")
                                    eligible = False
                                    
                                    rule_failures["en"].append(f"Occupation must be {rule_str} (you are {user_str})")
                                    rule_failures["te"].append(f"వృత్తి {rule_str} అయి ఉండాలి (మీది {user_str})")
                                    
                            elif key == "ownership":
                                # Handle the ownership status specifically
                                house_status_match = (rule_str == "own_house" and user_str == "own_house") or \
                                                    (rule_str == "no_house" and user_str == "no_house")
                                if not house_status_match:
                                    eligible = False
                                    print(f"  - House ownership mismatch: {user_str} != {rule_str}")
                                    
                                    if rule_str == "own_house":
                                        rule_failures["en"].append("Must own a house (you don't)")
                                        rule_failures["te"].append("స్వంత ఇల్లు ఉండాలి (మీకు లేదు)")
                                    else:
                                        rule_failures["en"].append("Must not own a house (you do)")
                                        rule_failures["te"].append("స్వంత ఇల్లు ఉండకూడదు (మీకు ఉంది)")
                                        
                            elif key == "gender":
                                # Handle gender specifically
                                if user_str != rule_str:
                                    eligible = False
                                    print(f"  - Gender mismatch: {user_str} != {rule_str}")
                                    
                                    if rule_str == "female":
                                        rule_failures["en"].append("Must be female (you are male)")
                                        rule_failures["te"].append("స్త్రీ అయి ఉండాలి (మీరు పురుషులు)")
                                    else:
                                        rule_failures["en"].append("Must be male (you are female)")
                                        rule_failures["te"].append("పురుషుడు అయి ఉండాలి (మీరు స్త్రీ)")
                            
                            elif key == "is_student":
                                # Handle student status
                                if (user_str == "true" and rule_str == "false") or (user_str == "false" and rule_str == "true"):
                                    eligible = False
                                    print(f"  - Student status mismatch: {user_str} != {rule_str}")
                                    
                                    if rule_str == "true":
                                        rule_failures["en"].append("Must be a student (you are not)")
                                        rule_failures["te"].append("విద్యార్థి అయి ఉండాలి (మీరు కాదు)")
                                    else:
                                        rule_failures["en"].append("Must not be a student (you are)")
                                        rule_failures["te"].append("విద్యార్థి కాకూడదు (మీరు విద్యార్థి)")
                            
                            else:
                                # For other string fields, require exact match (after normalization)
                                if user_str != rule_str:
                                    eligible = False
                                    print(f"  - String mismatch: {user_str} != {rule_str}")
                                    
                                    rule_failures["en"].append(f"{key} must be {rule_str} (yours is {user_str})")
                                    rule_failures["te"].append(f"{key} {rule_str} అయి ఉండాలి (మీది {user_str})")
                                    
                        elif operator == "!=":
                            # For inequality checks, we maintain strict matching
                            if user_str == rule_str:
                                eligible = False
                                print(f"  - String equality when inequality required: {user_str} == {rule_str}")
                                
                                rule_failures["en"].append(f"{key} must not be {rule_str} (yours is {user_str})")
                                rule_failures["te"].append(f"{key} {rule_str} కాకూడదు (మీది {user_str})")
                                
                except (ValueError, TypeError) as e:
                    print(f"  - Error comparing {key}: {e}")
                    skip_reason = f"Error processing {key}"
                    continue # Skip rule if error in comparison
            
            # After checking all rules, classify the scheme
            if eligible and not skip_reason:
                print(f"  => ELIGIBLE for {scheme_name}")
                matched_schemes.append((scheme_id, scheme_name, scheme_desc))
            elif eligible and skip_reason:
                print(f"  => POTENTIALLY ELIGIBLE for {scheme_name} (but {skip_reason})")
                # Could add to a "maybe" list if we wanted to show these
            else:
                print(f"  => NOT ELIGIBLE for {scheme_name}")
                # Store the reasons for being ineligible with translated reasons
                ineligible_schemes[scheme_id] = rule_failures
        
        print(f"\nResults: {len(matched_schemes)} matching schemes found")
        
        # Also return ineligible schemes for explanation
        return matched_schemes, ineligible_schemes
        
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        return [], {}
    except Exception as e:
        print(f"Unexpected error during scheme matching: {e}")
        return [], {}
    finally:
        if conn: conn.close()


# --- Function to get required documents using SQLite ---
def get_required_documents(scheme_id, detected_lang="te"):
    conn = None
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        # Ensure the table and column name 'is_mandatory' matches your actual DB schema if using it
        cursor.execute("SELECT document_name FROM documents_required WHERE scheme_id = ? ORDER BY document_name", (scheme_id,))
        # ORDER BY is_mandatory DESC, document_name  <-- Use this if 'is_mandatory' column exists and is reliable
        docs = cursor.fetchall()

        if not docs: return "ఈ స్కీమ్‌కు నిర్దిష్ట డాక్యుమెంట్లు జాబితా చేయబడలేదు." if detected_lang == "te" else "No specific documents listed for this scheme."

        doc_list = [doc[0] for doc in docs]
        return ("అవసరమైన డాక్యుమెంట్లు: " if detected_lang == "te" else "Required documents: ") + ", ".join(doc_list)

    except sqlite3.Error as e: print(f"DB error fetching documents: {e}"); return "డాక్యుమెంట్ పొందడంలో లోపం." if detected_lang == "te" else "Error getting documents."
    except Exception as e: print(f"Unexpected error fetching documents: {e}"); return "డాక్యుమెంట్ పొందడంలో లోపం." if detected_lang == "te" else "Error getting documents."
    finally:
        if conn: conn.close()


# --- Function to convert text to speech and play it (Robustness) ---
def speak_response(text, lang):
    if not text: print("Warning: speak_response called with empty text."); return
    print(f"Assistant speaking ({lang}): {text}")
    temp_audio_path = None
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
            tts.save(temp_audio.name)
            temp_audio_path = temp_audio.name
        # Using playsound might block; consider platform-specific or non-blocking options for production
        playsound.playsound(temp_audio_path)
    except ConnectionError as e: print(f"TTS Error: Network connection: {e}"); print("(Skipped speaking)")
    except ImportError: print("playsound library not found or backend missing?"); print("(Skipped speaking)") # Specific check
    except Exception as e: print(f"Error during TTS/playback ({type(e).__name__}): {e}"); print("(Skipped speaking)")
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try: os.remove(temp_audio_path)
            except Exception as e: print(f"Warning: Could not delete temp file {temp_audio_path}: {e}")


# --- Function to handle communication based on input method ---
def communicate(message_te=None, message_en=None, input_method="voice", language="te"):
    """
    Handles communication with the user based on the selected language and input method.
    Supports either providing both language messages or a single message with language flag.
    
    Args:
        message_te (str): The message in Telugu (or primary message if message_en is None)
        message_en (str): The message in English
        input_method (str): The input method ('keyboard' or 'voice')
        language (str): The user's preferred language ('en' or 'te')
    """
    # Handle single message case
    if message_en is None and message_te is not None:
        # Single message mode - message_te contains the primary message
        # and language determines how to interpret it
        message = message_te
    else:
        # Two messages provided, select based on language
        message = message_te if language == "te" else message_en
    
    # Always print the message for both methods
    print(message)
    
    # Only speak the response if using voice input method
    if input_method == "voice":
        speak_response(message, language)


# --- Main loop for conversational AI ---
def main():
    try:
        # Setup database and check if it's successful
        setup_success = setup_database()
        if not setup_success:
            print("Database setup failed. Please check the database configuration.")
            return

        # Default settings - start with voice and Telugu
        language = "te"
        input_method = "voice"
        beneficiary_relation = "self"  # Default to self
        
        # 1. Start with a greeting message immediately
        welcome_message_te = "నమస్కారం! నేను ఆంధ్రప్రదేశ్ ప్రభుత్వ సంక్షేమ పథకాల సహాయకుడిని. మీకు ఉపయోగపడేదాన్ని మేము చూద్దాం."
        welcome_message_en = "Hello! I am the Andhra Pradesh Government Welfare Schemes Assistant. Let's find what's useful for you."
        communicate(welcome_message_te, welcome_message_en, input_method, language)
        
        # Ask about language preference
        lang_choice_te = "మీకు తెలుగు లేదా ఇంగ్లీష్‌లో మాట్లాడాలనుకుంటున్నారా?"
        lang_choice_en = "Would you like to speak in Telugu or English?"
        communicate(lang_choice_te, lang_choice_en, input_method, language)
        
        lang_response = get_user_input(input_method, language)
        
        if lang_response and ("english" in lang_response.lower() or "ఇంగ్లీష్" in lang_response.lower()):
            language = "en"
            response_msg = "Great! We'll continue in English."
            communicate(response_msg, response_msg, input_method, language)
        else:
            response_te = "చాలా బాగుంది! మనం తెలుగులో కొనసాగిద్దాం."
            response_en = "Great! We'll continue in Telugu."
            communicate(response_te, response_en, input_method, language)
        
        # Ask about input method preference only after language is established
        input_method_q_te = "మీరు టైప్ చేయడానికి లేదా మాట్లాడటానికి ఇష్టపడతారా?"
        input_method_q_en = "Would you prefer to type or speak?"
        communicate(input_method_q_te, input_method_q_en, input_method, language)
        
        method_response = get_user_input(input_method, language)
        
        # Only switch to keyboard if explicitly requested
        if method_response and ("type" in method_response.lower() or "టైప్" in method_response.lower()):
            input_method = "keyboard"
            if language == "te":
                input_method_confirm_te = "సరే, మీరు టైప్ చేయడానికి ఎంచుకున్నారు."
                input_method_confirm_en = "Okay, you've chosen to type."
                communicate(input_method_confirm_te, input_method_confirm_en, input_method, language)
            else:
                input_method_confirm_en = "Okay, you've chosen to type."
                communicate(input_method_confirm_en, input_method_confirm_en, input_method, language)
        else:
            # Keep voice as default
            if language == "te":
                input_method_confirm_te = "సరే, మనం మాట్లాడటం ద్వారా కొనసాగుదాం."
                input_method_confirm_en = "Okay, we'll continue through speaking."
                communicate(input_method_confirm_te, input_method_confirm_en, input_method, language)
            else:
                input_method_confirm_en = "Okay, we'll continue through speaking."
                communicate(input_method_confirm_en, input_method_confirm_en, input_method, language)
        
        # 2. Ask if the query is for self or for others
        for_whom_te = "మీరు ఈ సమాచారాన్ని మీ కోసం వెతుకుతున్నారా లేక మరొకరి కోసమా?"
        for_whom_en = "Are you seeking this information for yourself or for someone else?"
        communicate(for_whom_te, for_whom_en, input_method, language)
        
        for_whom_response = get_user_input(input_method, language)
        
        # Check if the response is for others
        others_keywords_te = ["ఇతరులు", "మరొకరి", "వేరొకరి", "వారి", "ఆయన", "ఆవిడ"]
        others_keywords_en = ["other", "another", "someone else", "them", "him", "her", "relative", "friend"]
        
        is_for_others = False
        
        if for_whom_response:
            if (language == "te" and any(keyword in for_whom_response.lower() for keyword in others_keywords_te)) or \
               (language == "en" and any(keyword in for_whom_response.lower() for keyword in others_keywords_en)):
                is_for_others = True
        
        # 4. If it's for others, ask about the relationship
        if is_for_others:
            relation_te = "వారికి మీతో ఏమి సంబంధం? ఉదాహరణకు: తల్లిదండ్రులు, బంధువులు, తాత-నానమ్మలు, పొరుగువారు, స్నేహితులు..."
            relation_en = "What is your relationship with them? For example: parents, relatives, grandparents, neighbors, friends..."
            communicate(relation_te, relation_en, input_method, language)
            
            relation_response = get_user_input(input_method, language)
            if relation_response:
                beneficiary_relation = relation_response.strip().lower()
                print(f"Beneficiary relation: {beneficiary_relation}")
                
                # Confirm the relationship
                relation_confirm_te = f"సరే, మీరు మీ {beneficiary_relation} కోసం సహాయం వెతుకుతున్నారు."
                relation_confirm_en = f"Okay, you're seeking assistance for your {beneficiary_relation}."
                communicate(relation_confirm_te, relation_confirm_en, input_method, language)
        else:
            # For self, just confirm
            self_confirm_te = "సరే, మీ కోసం ప్రభుత్వ పథకాల గురించి తెలుసుకుందాం."
            self_confirm_en = "Okay, let's find out about government schemes for you."
            communicate(self_confirm_te, self_confirm_en, input_method, language)
        
        # Build user profile for self or other
        user_profile = build_user_profile(language, input_method, is_for_others)
        
        if not user_profile:
            # If profile building failed
            error_te = "క్షమించండి, ప్రొఫైల్‌ని సృష్టించడంలో నేను వైఫల్యం చెందాను. దయచేసి మళ్లీ ప్రయత్నించండి."
            error_en = "Sorry, I failed to create the profile. Please try again."
            communicate(error_te, error_en, input_method, language)
            return
        
        # Store the beneficiary relationship in the profile
        user_profile["beneficiary_relation"] = beneficiary_relation
        
        # Ask about current benefits
        benefits_te = "ప్రస్తుతం ఏదైనా ప్రభుత్వ పథకాలు పొందుతున్నారా? ఉదాహరణకు: వృద్ధాప్య పింఛను, ఎన్టీఆర్ రైతు భరోసా, ఆరోగ్యశ్రీ"
        benefits_en = "Are they currently receiving any government schemes? For example: Old Age Pension, NTR Rythu Bharosa, Aarogyasri"
        
        if not is_for_others:
            benefits_te = "మీరు ప్రస్తుతం ఏదైనా ప్రభుత్వ పథకాలు పొందుతున్నారా? ఉదాహరణకు: వృద్ధాప్య పింఛను, ఎన్టీఆర్ రైతు భరోసా, ఆరోగ్యశ్రీ"
            benefits_en = "Are you currently receiving any government schemes? For example: Old Age Pension, NTR Rythu Bharosa, Aarogyasri"
        
        communicate(benefits_te, benefits_en, input_method, language)
        
        current_schemes = []
        
        # First step - get initial response about current schemes
        schemes_answer = get_user_input(input_method, language)
        
        # Check if any scheme names were mentioned
        if schemes_answer:
            schemes_detected = extract_scheme_names(schemes_answer, language)
            if schemes_detected:
                current_schemes.extend(schemes_detected)
            
            # Check if they mentioned "no schemes" or similar
            no_schemes_keywords_te = ["లేవు", "ఏమీ లేవు", "లేదు", "తీసుకోవడం లేదు"]
            no_schemes_keywords_en = ["no", "none", "not receiving", "don't receive", "do not receive"]
            
            schemes_answer_lower = schemes_answer.lower()
            has_no_schemes = False
            
            if language == "te" and any(kw in schemes_answer_lower for kw in no_schemes_keywords_te):
                has_no_schemes = True
            elif language == "en" and any(kw in schemes_answer_lower for kw in no_schemes_keywords_en):
                has_no_schemes = True
                
            if not has_no_schemes and not current_schemes:
                # Second step - specifically ask if they have any schemes
                specific_te = "మీరు ఈ క్రింది ప్రభుత్వ పథకాలలో ఏవైనా పొందుతున్నారా? వృద్ధాప్య పింఛను, ఎన్టీఆర్ రైతు భరోసా, ఆరోగ్యశ్రీ, కల్యాణ లక్ష్మి, ఎన్టీఆర్ కిట్, ఆసరా పింఛను, షాదీ ముబారక్, చేయూత?"
                specific_en = "Are you receiving any of these government schemes? Old Age Pension, NTR Rythu Bharosa, Aarogyasri, Kalyana Lakshmi, NTR Kit, Aasara Pension, Shaadi Mubarak, Cheyutha?"
                communicate(specific_te, specific_en, input_method, language)
                
                specific_answer = get_user_input(input_method, language)
                if specific_answer:
                    specific_schemes = extract_scheme_names(specific_answer, language)
                    if specific_schemes:
                        for scheme in specific_schemes:
                            if scheme not in current_schemes:
                                current_schemes.append(scheme)
        
        # Print current schemes for reference
        if current_schemes:
            print(f"Current schemes: {current_schemes}")
            
            # Check maximum scheme limit (government policy)
            if len(current_schemes) >= 2:
                max_schemes_te = "మీరు ఇప్పటికే 2 లేదా అంతకంటే ఎక్కువ ప్రభుత్వ పథకాలను పొందుతున్నట్లు కనిపిస్తోంది. ప్రభుత్వ నిబంధనల ప్రకారం, ఒక వ్యక్తి గరిష్టంగా 2 పథకాలను మాత్రమే పొందగలరు."
                max_schemes_en = "It appears you are already receiving 2 or more government schemes. According to government regulations, an individual can receive a maximum of 2 schemes."
                communicate(max_schemes_te, max_schemes_en, input_method, language)
                
                # Ask if they want to explore other options
                explore_te = "మీరు ఏమైనా ప్రత్యామ్నాయ ప్రభుత్వ పథకాలను తెలుసుకోవాలనుకుంటున్నారా?"
                explore_en = "Would you like to learn about alternative government schemes you might be eligible for?"
                communicate(explore_te, explore_en, input_method, language)
                
                explore_answer = get_user_input(input_method, language)
                if explore_answer:
                    explore_keywords_te = ["అవును", "కావాలి", "చెప్పండి", "తెలుసుకోవాలి"]
                    explore_keywords_en = ["yes", "sure", "okay", "please", "want to know"]
                    
                    if (language == "te" and not any(kw in explore_answer.lower() for kw in explore_keywords_te)) or \
                       (language == "en" and not any(kw in explore_answer.lower() for kw in explore_keywords_en)):
                        # User doesn't want to explore alternatives
                        bye_te = "సరే, ధన్యవాదాలు. మీకు మరింత సహాయం అవసరమైతే, మళ్ళీ వచ్చి చూడండి."
                        bye_en = "Alright, thank you. Please visit again if you need more assistance."
                        communicate(bye_te, bye_en, input_method, language)
                        return
        
        # Match user with eligible schemes
        matching_schemes, ineligible_schemes = match_schemes(user_profile)
        
        # 8. Share results and handle scheme interest
        if matching_schemes:
            eligible_te = f"మీరు {len(matching_schemes)} ప్రభుత్వ పథకాలకు అర్హులు కావచ్చు:"
            eligible_en = f"You may be eligible for {len(matching_schemes)} government schemes:"
            
            if is_for_others:
                eligible_te = f"వారు {len(matching_schemes)} ప్రభుత్వ పథకాలకు అర్హులు కావచ్చు:"
                eligible_en = f"They may be eligible for {len(matching_schemes)} government schemes:"
                
            communicate(eligible_te, eligible_en, input_method, language)
            
            # Display each scheme with a serial number
            for idx, scheme_id in enumerate(matching_schemes, 1):
                scheme_info = get_scheme_info(scheme_id)
                
                if not scheme_info:
                    continue
                
                scheme_msg_te = f"{idx}. {scheme_info['name_te']}: {scheme_info['description_te']}"
                scheme_msg_en = f"{idx}. {scheme_info['name_en']}: {scheme_info['description_en']}"
                communicate(scheme_msg_te, scheme_msg_en, input_method, language)
            
            # Ask if they're interested in any particular scheme
            interest_q_te = "ఏదైనా నిర్దిష్ట పథకం గురించి మరింత తెలుసుకోవాలనుకుంటున్నారా? దయచేసి సంఖ్య లేదా పథకం పేరు చెప్పండి."
            interest_q_en = "Would you like to learn more about any specific scheme? Please mention the number or scheme name."
            communicate(interest_q_te, interest_q_en, input_method, language)
            
            scheme_interest = get_user_input(input_method, language)
            selected_scheme = None
            
            if scheme_interest:
                # First try to match by number
                if scheme_interest.isdigit():
                    try:
                        idx = int(scheme_interest) - 1
                        if 0 <= idx < len(matching_schemes):
                            selected_scheme = matching_schemes[idx]
                    except:
                        pass
                
                # If not matched by number, try by name
                if not selected_scheme:
                    for scheme_id in matching_schemes:
                        scheme_info = get_scheme_info(scheme_id)
                        if not scheme_info:
                            continue
                        
                        scheme_name_te = scheme_info["name_te"].lower()
                        scheme_name_en = scheme_info["name_en"].lower()
                        
                        if scheme_name_te in scheme_interest.lower() or scheme_name_en in scheme_interest.lower():
                            selected_scheme = scheme_id
                            break
            
            # Show detailed information about the selected scheme
            if selected_scheme:
                scheme_info = get_scheme_info(selected_scheme)
                
                if language == "te":
                    details = f"పథకం పేరు: {scheme_info['name_te']}\n"
                    details += f"వివరణ: {scheme_info['description_te']}\n"
                    details += f"అర్హత: {scheme_info['eligibility_te']}\n" 
                    details += f"లాభాలు: {scheme_info['benefits_te']}\n"
                    details += f"దరఖాస్తు ప్రక్రియ: {scheme_info['application_te']}\n"
                    details += f"ధృవీకరణ పత్రాలు: {scheme_info['documents_te']}\n"
                    details += f"సంప్రదించవలసిన వ్యక్తి: {scheme_info['contact_te']}"
                else:
                    details = f"Scheme Name: {scheme_info['name_en']}\n"
                    details += f"Description: {scheme_info['description_en']}\n"
                    details += f"Eligibility: {scheme_info['eligibility_en']}\n"
                    details += f"Benefits: {scheme_info['benefits_en']}\n"
                    details += f"Application Process: {scheme_info['application_en']}\n"
                    details += f"Required Documents: {scheme_info['documents_en']}\n"
                    details += f"Contact Information: {scheme_info['contact_en']}"
                
                communicate(details, None, input_method, language)
                
                # Ask if they want to apply for this scheme
                apply_q_te = "ఈ పథకానికి దరఖాస్తు చేసుకోవాలనుకుంటున్నారా?"
                apply_q_en = "Would you like to apply for this scheme?"
                communicate(apply_q_te, apply_q_en, input_method, language)
                
                apply_answer = get_user_input(input_method, language)
                if apply_answer:
                    apply_interest = extract_yes_no(apply_answer, language)
                    
                    if apply_interest:
                        # User is interested in applying
                        if language == "te":
                            apply_info = f"దరఖాస్తు ప్రక్రియ:\n{scheme_info['application_te']}\n\n"
                            apply_info += f"అవసరమైన పత్రాలు:\n{scheme_info['documents_te']}\n\n"
                            apply_info += f"సంప్రదించవలసిన వివరాలు:\n{scheme_info['contact_te']}"
                        else:
                            apply_info = f"Application Process:\n{scheme_info['application_en']}\n\n"
                            apply_info += f"Required Documents:\n{scheme_info['documents_en']}\n\n"
                            apply_info += f"Contact Information:\n{scheme_info['contact_en']}"
                        
                        communicate(apply_info, None, input_method, language)
                    else:
                        # User is not interested in applying, check if they want to explore other schemes
                        if len(matching_schemes) > 1:
                            other_schemes_q_te = "మరో పథకం గురించి తెలుసుకోవాలనుకుంటున్నారా?"
                            other_schemes_q_en = "Would you like to learn about another scheme?"
                            communicate(other_schemes_q_te, other_schemes_q_en, input_method, language)
                            
                            other_answer = get_user_input(input_method, language)
                            if other_answer and extract_yes_no(other_answer, language):
                                # Show the list again without the previously selected scheme
                                remaining_schemes = [s for s in matching_schemes if s != selected_scheme]
                                
                                for idx, scheme_id in enumerate(remaining_schemes, 1):
                                    scheme_info = get_scheme_info(scheme_id)
                                    
                                    if not scheme_info:
                                        continue
                                    
                                    scheme_msg_te = f"{idx}. {scheme_info['name_te']}: {scheme_info['description_te']}"
                                    scheme_msg_en = f"{idx}. {scheme_info['name_en']}: {scheme_info['description_en']}"
                                    communicate(scheme_msg_te, scheme_msg_en, input_method, language)
                                
                                # Recursively process this part (could be refactored to avoid repetition)
                                interest_q_te = "ఏదైనా నిర్దిష్ట పథకం గురించి మరింత తెలుసుకోవాలనుకుంటున్నారా?"
                                interest_q_en = "Would you like to learn more about any specific scheme?"
                                communicate(interest_q_te, interest_q_en, input_method, language)
                                
                                # Continue with the same logic as before (omitted for brevity)
        else:
            # 9. No eligible schemes found - Give clear feedback
            message_te = "క్షమించండి, ప్రస్తుతం ఏ ప్రభుత్వ పథకాలకు అర్హులు కాదని కనిపిస్తోంది."
            message_en = "Sorry, it appears you are not eligible for any government schemes at this time."
            
            if is_for_others:
                message_te = "క్షమించండి, ప్రస్తుతం వారు ఏ ప్రభుత్వ పథకాలకు అర్హులు కాదని కనిపిస్తోంది."
                message_en = "Sorry, it appears they are not eligible for any government schemes at this time."
                
            communicate(message_te, message_en, input_method, language)
            
            # Share reasons for ineligibility if available
            if ineligible_schemes:
                reason_msg_te = "ఇందుకు కారణాలు:"
                reason_msg_en = "Reasons include:"
                communicate(reason_msg_te, reason_msg_en, input_method, language)
                
                # Show top 3 schemes and why they're ineligible
                counter = 0
                for scheme_id, reasons in ineligible_schemes.items():
                    if counter >= 3:  # Limit to top 3 to avoid overwhelming
                        break
                        
                    scheme_info = get_scheme_info(scheme_id)
                    if not scheme_info:
                        continue
                        
                    scheme_name_te = scheme_info["name_te"]
                    scheme_name_en = scheme_info["name_en"]
                    
                    reasons_text_te = ", ".join(reasons["te"]) if "te" in reasons else "తెలియని కారణం"
                    reasons_text_en = ", ".join(reasons["en"]) if "en" in reasons else "Unknown reason"
                    
                    scheme_msg_te = f"{scheme_name_te}: {reasons_text_te}"
                    scheme_msg_en = f"{scheme_name_en}: {reasons_text_en}"
                    communicate(scheme_msg_te, scheme_msg_en, input_method, language)
                    counter += 1
                
                # Encourage them to check back later
                update_msg_te = "మేము నిరంతరం కొత్త పథకాలను జోడిస్తూ ఉంటాము. దయచేసి తర్వాత మళ్లీ సందర్శించండి."
                update_msg_en = "We are constantly adding new schemes. Please check back later."
                communicate(update_msg_te, update_msg_en, input_method, language)
        
        # Final message
        final_te = "ధన్యవాదాలు! మరేదైనా సహాయం అవసరమైతే, దయచేసి మళ్లీ సంప్రదించండి."
        final_en = "Thank you! Please reach out again if you need any further assistance."
        communicate(final_te, final_en, input_method, language)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print(traceback.format_exc())

# --- Function to get user input based on the input method ---
def get_user_input(input_method, language="te"):
    """
    Gets user input based on the selected input method.
    
    Args:
        input_method (str): The input method ('keyboard' or 'voice')
        language (str): The language to use ('en' or 'te'), defaults to 'te'
    
    Returns:
        str: The user's input text
    """
    if input_method == "voice":
        # Use speech recognition for voice input
        text, detected_lang = recognize_speech(language, input_method)
        return text
    else:
        # Use keyboard input
        user_input = input("> ").strip()
        return user_input

# --- Function to extract gender from text ---
def extract_gender(text, language="te"):
    """
    Extract gender information from text.
    
    Args:
        text (str): The text to extract gender from
        language (str): The language of the text ('te' for Telugu, 'en' for English)
        
    Returns:
        str: 'male' or 'female' or None if not determined
    """
    text_lower = text.lower()
    
    # Define keywords for male and female in Telugu and English
    male_keywords_te = ["మగ", "పురుషుడు", "అబ్బాయి", "మగా"]
    female_keywords_te = ["ఆడ", "స్త్రీ", "అమ్మాయి", "ఆడా"]
    male_keywords_en = ["male", "man", "boy", "gent"]
    female_keywords_en = ["female", "woman", "girl", "lady"]
    
    # Check for Telugu keywords if language is Telugu
    if language == "te":
        if any(keyword in text_lower for keyword in male_keywords_te):
            return "male"
        if any(keyword in text_lower for keyword in female_keywords_te):
            return "female"
    
    # Always check English keywords as fallback
    if any(keyword in text_lower for keyword in male_keywords_en):
        return "male"
    if any(keyword in text_lower for keyword in female_keywords_en):
        return "female"
    
    # If no keywords found, return None
    return None


# --- Function to extract number from text (for English) ---
def extract_number(text):
    """
    Extract a numerical value from English text.
    
    Args:
        text (str): The text to extract a number from
        
    Returns:
        int: The extracted number, or None if no number found
    """
    # Basic number extraction
    try:
        # Clean the text for better extraction
        cleaned_text = text.lower().replace(",", "").replace("rupees", "").replace("years", "").strip()
        
        # Check if the text consists of digits only
        if cleaned_text.isdigit():
            return int(cleaned_text)
        
        # Try to extract digits
        digits = ""
        for char in cleaned_text:
            if char.isdigit():
                digits += char
        
        if digits:
            return int(digits)
        
        # If no direct digits, process text for word numbers
        # (A more complex implementation would handle English word numbers like "twenty-five")
        return None
    except Exception:
        return None


# --- Function to extract occupation from text ---
def extract_occupation(text, language="te"):
    """
    Extract occupation information from text.
    
    Args:
        text (str): The text to extract occupation from
        language (str): The language of the text ('te' for Telugu, 'en' for English)
        
    Returns:
        str: The extracted occupation, or None if not determined
    """
    text_lower = text.lower().strip()
    
    # Define occupation mappings for common professions
    occupations_te = {
        "రైతు": "farmer",
        "వ్యవసాయం": "farmer",
        "సెటిల్": "farmer",
        "గృహిణి": "homemaker",
        "హౌస్‌వైఫ్": "homemaker",
        "ఉద్యోగం": "employee",
        "ఉద్యోగి": "employee",
        "వ్యాపారం": "business",
        "బిజినెస్": "business",
        "విద్యార్థి": "student",
        "చదువుకుంటున్నాను": "student",
        "డాక్టర్": "doctor",
        "వైద్యుడు": "doctor", 
        "ఉపాధ్యాయుడు": "teacher",
        "టీచర్": "teacher",
        "ఇంజనీర్": "engineer",
        "నిరుద్యోగి": "unemployed",
        "జాబ్ లేదు": "unemployed",
        "పని లేదు": "unemployed"
    }
    
    occupations_en = {
        "farmer": "farmer",
        "agriculture": "farmer",
        "farming": "farmer",
        "homemaker": "homemaker",
        "housewife": "homemaker",
        "employee": "employee",
        "working": "employee",
        "job": "employee",
        "business": "business",
        "businessman": "business",
        "entrepreneur": "business",
        "student": "student",
        "studying": "student",
        "doctor": "doctor",
        "physician": "doctor",
        "teacher": "teacher",
        "professor": "teacher",
        "engineer": "engineer",
        "unemployed": "unemployed",
        "no job": "unemployed",
        "not working": "unemployed"
    }
    
    # Check for occupation matches
    if language == "te":
        for key, value in occupations_te.items():
            if key in text_lower:
                return value
    
    # Check English terms regardless of language (many Telugu speakers use English terms)
    for key, value in occupations_en.items():
        if key in text_lower:
            return value
    
    # If no match found, return the original (cleaned) text
    return text_lower


# --- Function to check if person is a student ---
def check_is_student(text, language="te"):
    """
    Check if the text indicates the person is a student.
    
    Args:
        text (str): The text to check
        language (str): The language of the text ('te' for Telugu, 'en' for English)
        
    Returns:
        bool: True if the person is likely a student, False otherwise
    """
    text_lower = text.lower()
    
    # Define student keywords in Telugu and English
    student_keywords_te = ["విద్యార్థి", "చదువుకుంటున్నాను", "స్కూల్", "కాలేజీ", "యూనివర్సిటీ", "బడి", "చదువు"]
    student_keywords_en = ["student", "studying", "school", "college", "university", "education", "study"]
    
    # Check for Telugu keywords if language is Telugu
    if language == "te" and any(keyword in text_lower for keyword in student_keywords_te):
        return True
    
    # Check for English keywords
    if any(keyword in text_lower for keyword in student_keywords_en):
        return True
    
    # If occupation exactly equals student
    if text_lower == "student" or text_lower == "విద్యార్థి":
        return True
    
    return False


# --- Function to extract yes/no from text ---
def extract_yes_no(text, language="te"):
    """
    Extract yes/no information from text.
    
    Args:
        text (str): The text to extract yes/no from
        language (str): The language of the text ('te' for Telugu, 'en' for English)
        
    Returns:
        bool: True for yes, False for no, None if not determined
    """
    text_lower = text.lower()
    
    # Define yes/no keywords in Telugu and English
    yes_keywords_te = ["అవును", "ఔను", "కరెక్ట్", "సరే", "అలాగే", "ఒప్పు", "ఒకే", "యెస్"]
    no_keywords_te = ["కాదు", "లేదు", "నో", "వద్దు", "కాదండి", "వద్దండి"]
    yes_keywords_en = ["yes", "yeah", "correct", "sure", "okay", "ok", "true", "right"]
    no_keywords_en = ["no", "nope", "not", "don't", "doesn't", "false", "wrong"]
    
    # Check for Telugu keywords if language is Telugu
    if language == "te":
        if any(keyword in text_lower or keyword == text_lower for keyword in yes_keywords_te):
            return True
        if any(keyword in text_lower or keyword == text_lower for keyword in no_keywords_te):
            return False
    
    # Always check English keywords as fallback
    if any(keyword in text_lower or keyword == text_lower for keyword in yes_keywords_en):
        return True
    if any(keyword in text_lower or keyword == text_lower for keyword in no_keywords_en):
        return False
    
    # If no keywords found, return None
    return None


# --- Function to extract income range from text ---
def extract_income_range(text, language="te"):
    """
    Extract income range information from text.
    
    Args:
        text (str): The text to extract income range from
        language (str): The language of the text ('te' for Telugu, 'en' for English)
        
    Returns:
        str: The extracted income range or None if not determined
    """
    text_lower = text.lower()
    
    # Define range keywords in Telugu and English
    less_than_keywords_te = ["కంటే తక్కువ", "లోపు", "కంటే తక్కువ", "కంటే చిన్నది"]
    between_keywords_te = ["మధ్య", "నడుమ"]
    more_than_keywords_te = ["కంటే ఎక్కువ", "పైన", "కంటే పైన", "కంటే పెద్దది"]
    
    less_than_keywords_en = ["less than", "below", "under", "not more than"]
    between_keywords_en = ["between", "from", "in the range of", "ranging"]
    more_than_keywords_en = ["more than", "above", "over", "exceeding", "at least"]
    
    # Extract numbers from the text
    numbers = []
    words = text_lower.split()
    for word in words:
        # Clean the word and try to extract a number
        clean_word = word.replace(",", "").replace(".", "").replace("rs", "").strip()
        if clean_word.isdigit():
            numbers.append(int(clean_word))
        elif language == "te":
            # Try parsing Telugu numbers
            telugu_num = telugu_to_number(word)
            if telugu_num > 0:
                numbers.append(telugu_num)
    
    # Determine the range based on keywords and numbers
    if language == "te":
        if any(keyword in text_lower for keyword in less_than_keywords_te) and numbers:
            return f"< {max(numbers)}"
        if any(keyword in text_lower for keyword in more_than_keywords_te) and numbers:
            return f"> {min(numbers)}"
        if any(keyword in text_lower for keyword in between_keywords_te) and len(numbers) >= 2:
            numbers.sort()
            return f"{numbers[0]} - {numbers[-1]}"
    
    # Check English keywords
    if any(keyword in text_lower for keyword in less_than_keywords_en) and numbers:
        return f"< {max(numbers)}"
    if any(keyword in text_lower for keyword in more_than_keywords_en) and numbers:
        return f"> {min(numbers)}"
    if any(keyword in text_lower for keyword in between_keywords_en) and len(numbers) >= 2:
        numbers.sort()
        return f"{numbers[0]} - {numbers[-1]}"
    
    # If we have numbers but no clear range, use the first number
    if numbers:
        return str(numbers[0])
    
    # If no clear range could be determined
    return None


# --- Function to extract scheme names from text ---
def extract_scheme_names(text, language="te"):
    """
    Extract government scheme names from text.

    Args:
        text (str): Input text
        language (str): Language code (en/te)
        
    Returns:
        list: List of extracted scheme names
    """
    # Normalize input text and convert to lowercase
    text = text.lower().strip()
    
    # Dictionary of scheme keywords in both languages
    scheme_keywords = {
        "pension": ["pension", "పెన్షన్", "పింఛను"],
        "housing": ["housing", "house", "ఇల్లు", "ఇంటి", "గృహ"],
        "education": ["education", "school", "విద్య", "స్కాలర్షిప్", "చదువు"],
        "health": ["health", "medical", "ఆరోగ్య", "వైద్య", "మెడికల్", "బీమా"],
        "agriculture": ["agriculture", "farm", "వ్యవసాయ", "రైతు"],
        "employment": ["employment", "job", "ఉద్యోగ", "ఉపాధి", "జాబ్"],
        "financial": ["financial", "loan", "ఆర్థిక", "రుణ", "లోన్"],
        "women": ["women", "మహిళా", "స్త్రీల"]
    }
    
    # List to store extracted schemes
    extracted_schemes = []
    
    # Check for scheme keywords in input text
    for scheme, keywords in scheme_keywords.items():
        for keyword in keywords:
            if keyword in text:
                extracted_schemes.append(scheme)
                break
    
    return extracted_schemes

# --- Function to get a database connection ---
def get_db_connection():
    """Get a connection to the SQLite database.
    
    Returns:
        sqlite3.Connection: A connection to the database or None if an error occurs
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        return conn
    except sqlite3.Error as e:
        print(f"Database connection error: {e}")
        return None

# --- NEW: Function to extract caste from text ---
def extract_caste(text, language="te"):
    """
    Extract caste category (OC, BC, SC, ST) from text.

    Args:
        text (str): The text to extract caste from
        language (str): The language of the text ('te' for Telugu, 'en' for English)

    Returns:
        str: Normalized caste category ('oc', 'bc', 'sc', 'st') or None if not determined
    """
    text_lower = text.lower().strip()

    # Define keywords for caste categories
    caste_map = {
        'oc': ['oc', 'ఓసి', 'ఓసీ', 'general', 'forward caste', 'open category'],
        'bc': ['bc', 'బిసి', 'బీసీ', 'backward class', 'other backward class'],
        'sc': ['sc', 'ఎస్సీ', 'scheduled caste', 'దళిత', 'హరిజన'],
        'st': ['st', 'ఎస్టీ', 'scheduled tribe', 'గిరిజన', 'tribal', 'adivasi', 'ఆదివాసి']
    }

    # Check for keywords in the input text
    for category, keywords in caste_map.items():
        # Check language specific and general keywords
        target_keywords = keywords # Check all keywords regardless of language for robustness
        # if language == 'te':
        #     target_keywords = keywords # Or potentially filter keywords based on language
        # else:
        #     target_keywords = keywords

        if any(keyword in text_lower for keyword in target_keywords):
            return category # Return the normalized category ('oc', 'bc', 'sc', 'st')

    # Handle "I don't know" specifically for caste if needed, though ask_question_with_retry handles general case
    # if check_dont_know(text_lower): return None

    print(f"Could not extract caste category from: '{text}'")
    return None # Return None if no category identified

# Entry point for the script - following standard Python practice
if __name__ == "__main__":
    main()