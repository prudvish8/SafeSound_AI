# Minimal constants for filter_text and testing
GARBAGE_PHRASES_TE = ["బ్లా బ్లా", "అబద్ధం"]
GARBAGE_PHRASES_EN = ["asdfasdf", "blah blah"]
SHORT_SOUNDS_TE = ["హ్మ్"]
SHORT_SOUNDS_EN = ["hm"]
MIN_MEANINGFUL_LENGTH = 2

import unicodedata
import sqlite3
import logging
import pandas as pd
import os

DB_FILE = os.getenv("DB_FILE", "welfare_schemes.db")

# ... (rest of the code remains the same)

def telugu_to_number(text):
    # V8: Refined helper logic, map confirmation
    def normalize_nfc(s): return unicodedata.normalize('NFC', s)

    telugu_map_raw = {
        "సున్నా": 0, "ఒకటి": 1, "రెండు": 2, "మూడు": 3, "నాలుగు": 4, "ఐదు": 5,
        "ఆరు": 6, "ఏడు": 7, "ఎనిమిది": 8, "తొమ్మిది": 9, "పది": 10,
        "పదకొండు": 11, "పన్నెండు": 12, "పదమూడు": 13, "పద్నాలుగు": 14, "పదిహేను": 15,
        "పదహారు": 16, "పదిహేడు": 17, "పద్దెనిమిది": 18, "పందొమ్మిది": 19,
        "ఇరవై": 20, "ముప్పై": 30, "నలభై": 40, "యాభై": 50, "అరవై": 60,
        "డెబ్బై": 70, "ఎనభై": 80, "తొంభై": 90 # CRITICAL
    }
    telugu_map = {normalize_nfc(k): v for k, v in telugu_map_raw.items()}
    # Verify problematic keys explicitly
    if normalize_nfc("ముప్పై") not in telugu_map: logging.debug("FATAL DEBUG: ముప్పై not in map after normalization!")
    if normalize_nfc("తొంభై") not in telugu_map: logging.debug("FATAL DEBUG: తొంభై not in map after normalization!")


    normalizer_map_raw = {
        "కోట్ల": "కోటి", "కోట్లు": "కోటి", "లక్షల": "లక్ష", "లక్షా": "లక్ష", "లక్షలు": "లక్ష",
        "వేల": "వేలు", "వేలా": "వేలు", "వేయి": "వేలు", "వెయ్యి": "వేలు", "వేలాండ్": "వేలు",
        "వందల": "వంద", "వందలు": "వంద", "ఒక": "ఒకటి", "రెండ": "రెండు",
        "ఎనభయ్": "ఎనభై", # Keep common ASR error mapping
        "ఎనిమిదండి": "ఎనిమిది", "మూడండి": "మూడు", "నాలుగండి": "నాలుగు",
    }
    normalizer_map = {normalize_nfc(k): normalize_nfc(v) for k, v in normalizer_map_raw.items()}

    multipliers_raw = {"కోటి": 10000000, "లక్ష": 100000, "వేలు": 1000, "వంద": 100}
    multipliers = {normalize_nfc(k): v for k, v in multipliers_raw.items()}
    multiplier_order = [normalize_nfc(k) for k in ["కోటి", "లక్ష", "వేలు", "వంద"]]

    # 1. Input Cleaning & Normalization
    logging.debug(f"\n[telugu_to_number V8] Input: '{text}'")
    cleaned_text_nfc = normalize_nfc(text.replace(",", "").replace("రూపాయలు", "").replace("సంవత్సరాలు", "").strip().lower())
    if not cleaned_text_nfc: return None # Return None for empty
    if cleaned_text_nfc.isdigit():
        try: return int(cleaned_text_nfc)
        except ValueError: return None # Should not happen, but safe

    words = cleaned_text_nfc.split()
    normalized_words = []
    # (Normalization loop - simplified for brevity, assume it works as before to produce list of NFC words)
    for word in words:
        nfc_word=normalize_nfc(word)
        if nfc_word in normalizer_map: cleaned=normalizer_map[nfc_word]
        else: cleaned=nfc_word
        suffixes_to_remove=[normalize_nfc(s) for s in ["ండి","అండి"]]
        base = cleaned  # Ensure base is always defined
        for suffix in suffixes_to_remove:
            if cleaned.endswith(suffix):
                base=cleaned[:-len(suffix)]
                if base in telugu_map or base in multipliers or base in normalizer_map or not base:
                    cleaned=base
                    break
        if cleaned in normalizer_map:
            cleaned=normalizer_map[cleaned] # Recheck map
        if cleaned and cleaned!=normalize_nfc("అ"):
            normalized_words.append(cleaned)


    logging.debug(f"Normalized words (NFC): {normalized_words}")
    if not normalized_words: return None # Return None if no words left

    total_value = 0; words_to_process = normalized_words[:];

    # 2. *** CORRECTED Helper V8 ***
    def parse_simple_sequence(sequence):
        """Parses sequences like ['ముప్పై', 'ఆరు'] into 36."""
        val = 0; temp_val = 0;
        # Process from right to left for simpler addition (unit then ten) might be more robust? No, left-to-right basic addition
        for word in sequence:
             num = telugu_map.get(word) # Lookup normalized word
             if num is not None:
                 # Simple additive logic: If we see a number, add it.
                 # This works for "ముప్పై ఆరు" (30 + 6) and "ఆరు" (6)
                 val += num
             else: # Word not in number map (might be leftover multiplier?)
                 if not (len(sequence)==1 and word in multipliers): # Only allow lone multiplier
                     logging.debug(f"[ParseSimpleWarn] Ignoring '{word}' in {sequence}")
        # logging.debug(f"  ParseSimple result: {val}") # Debug
        return val

    # 3. Process Multipliers (using CORRECTED helper)
    for mult_word in multiplier_order:
        mult_value=multipliers[mult_word]; indices_to_remove=[]; i=0; processed=False
        while i < len(words_to_process):
            if words_to_process[i] == mult_word:
                processed=True; coeffs=[]; coeff_idx=[]; j=i-1
                while j >= 0 and words_to_process[j] not in multipliers: coeffs.insert(0, words_to_process[j]); coeff_idx.insert(0, j); j -= 1;
                # *** CALL CORRECTED HELPER ***
                parsed = parse_simple_sequence(coeffs); # Parse the coefficient words
                coeff = parsed if parsed > 0 else 1; # Default to 1 if empty/failed parse
                segment = coeff * mult_value; total_value += segment;
                logging.debug(f"  Segment: {coeffs} '{mult_word}' -> {coeff}*{mult_value}={segment}. Total:{total_value}")
                indices_to_remove.extend(coeff_idx); indices_to_remove.append(i)
            i += 1
        if indices_to_remove:
             unique=sorted(list(set(indices_to_remove)), reverse=True); temp=words_to_process[:];
             for idx in unique:
                  if idx<len(temp): del temp[idx]
             words_to_process = temp
             if processed: logging.debug(f"  Remain after '{mult_word}': {words_to_process}")

    # 4. Process Remainder (using CORRECTED helper)
    if words_to_process: remaining=parse_simple_sequence(words_to_process); total_value+=remaining; logging.debug(f"  Remaining {words_to_process} -> {remaining}");

    # 5. Final Return (Return None if result is 0 for non-zero input)
    logging.debug(f"--> Final parsed number: {total_value}")
    is_zero = (len(normalized_words) == 1 and normalize_nfc("సున్నా") in normalized_words[0])
    if total_value == 0 and not is_zero:
        # Try digit fallback *only* if word parse failed AND it wasn't supposed to be 0
        try:
            digits="".join(filter(str.isdigit, text.replace(" ","")));
            if digits: logging.debug("Fallback: Digit extraction."); return int(digits)
        except ValueError: pass
        logging.debug("Warning: Resulted in 0 or parsing failed."); return None # Return None on failure
    else:
        return total_value # Return the calculated value (can be 0 if input was 'సున్నా')

from typing import Optional

# --- Helper: Text Filtering (Refactored & Type-Annotated) ---
def filter_text(raw_text: str, lang: str = "te") -> Optional[str]:
    if not raw_text:
        logging.debug("[Filter] Empty input.")
        return None

    check = raw_text.lower().strip()
    if not check:
        logging.debug("[Filter] Empty after clean.")
        return None

    # ≪ NEW: immediately accept pure digits ≫
    if check.isdigit():
        logging.debug(f"[Filter] Kept digit input: '{raw_text}'")
        return raw_text

    essential_short_te = ["ఓసి", "బిసి", "ఎస్సీ", "ఎస్టీ", "మగ", "ఆడ", "అవును", "కాదు", "లేదు", "ఉంది"]
    essential_short_en = ["hi", "hello","oc", "bc", "sc", "st", "yes", "no", "male", "female"]
    essentials = essential_short_te if lang == 'te' else essential_short_en

    g_list = GARBAGE_PHRASES_TE if lang == 'te' else GARBAGE_PHRASES_EN
    s_list = SHORT_SOUNDS_TE if lang == 'te' else SHORT_SOUNDS_EN

    if check in essentials:
        logging.debug(f"[Filter] Kept essential short answer: '{raw_text}'")
        return raw_text

    if check in g_list:
        logging.debug(f"[Filter] Filtered exact garbage: '{raw_text}'")
        return None

    if check in s_list:
        logging.debug(f"[Filter] Filtered short sound: '{raw_text}'")
        return None

    if len(raw_text) < MIN_MEANINGFUL_LENGTH:
        logging.debug(f"[Filter] Filtered too short (and not essential): '{raw_text}'")
        return None

    logging.debug(f"[Filter] Input passed filtering: '{raw_text}'")
    return raw_text
# --- End Refactored Filter ---

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
    male_keywords_te = ["మగ", "పురుషుడు", "అబ్బాయి", "మగా"]
    female_keywords_te = ["ఆడ", "స్త్రీ", "అమ్మాయి", "ఆడా"]
    male_keywords_en = ["male", "man", "boy", "gent"]
    female_keywords_en = ["female", "woman", "girl", "lady"]
    if language == "te":
        if any(keyword in text_lower for keyword in male_keywords_te):
            return "male"
        if any(keyword in text_lower for keyword in female_keywords_te):
            return "female"
    if any(keyword in text_lower for keyword in female_keywords_en):
        return "female"
    if any(keyword in text_lower for keyword in male_keywords_en):
        return "male"
    return None


def extract_number(text):
    """
    Extract a numerical value from English text.
    Args:
        text (str): The text to extract a number from
    Returns:
        int: The extracted number, or None if no number found
    """
    try:
        cleaned_text = text.lower().replace(",", "").replace("rupees", "").replace("years", "").strip()
        if cleaned_text.isdigit():
            return int(cleaned_text)
        digits = ""
        for char in cleaned_text:
            if char.isdigit():
                digits += char
        if digits:
            return int(digits)
        return None
    except Exception:
        return None


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
    occupations_te = {
        "రైతు": "farmer", "వ్యవసాయం": "farmer", "సెటిల్": "farmer", "గృహిణి": "homemaker",
        "హౌస్‌వైఫ్": "homemaker", "ఉద్యోగం": "employee", "ఉద్యోగి": "employee", "వ్యాపారం": "business",
        "బిజినెస్": "business", "విద్యార్థి": "student", "చదువుకుంటున్నాను": "student", "డాక్టర్": "doctor",
        "వైద్యుడు": "doctor", "ఉపాధ్యాయుడు": "teacher", "టీచర్": "teacher", "ఇంజనీర్": "engineer",
        "నిరుద్యోగి": "unemployed", "జాబ్ లేదు": "unemployed", "పని లేదు": "unemployed"
    }
    occupations_en = {
        "farmer": "farmer", "agriculture": "farmer", "farming": "farmer", "homemaker": "homemaker",
        "housewife": "homemaker", "employee": "employee", "working": "employee", "job": "employee",
        "business": "business", "businessman": "business", "entrepreneur": "business", "student": "student",
        "studying": "student", "doctor": "doctor", "physician": "doctor", "teacher": "teacher",
        "professor": "teacher", "engineer": "engineer", "unemployed": "unemployed", "no job": "unemployed",
        "not working": "unemployed"
    }
    if language == "te":
        for key, value in occupations_te.items():
            if key in text_lower:
                return value
    for key, value in occupations_en.items():
        if key in text_lower:
            return value
    return text_lower


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
    student_keywords_te = ["విద్యార్థి", "చదువుకుంటున్నాను", "స్కూల్", "కాలేజీ", "యూనివర్సిటీ", "బడి", "చదువు"]
    student_keywords_en = ["student", "studying", "school", "college", "university", "education", "study"]
    if language == "te" and any(keyword in text_lower for keyword in student_keywords_te):
        return True
    if any(keyword in text_lower for keyword in student_keywords_en):
        return True
    if text_lower == "student" or text_lower == "విద్యార్థి":
        return True
    return False


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
    yes_keywords_te = ["అవును", "ఔను", "కరెక్ట్", "సరే", "అలాగే", "ఒప్పు", "ఒకే", "యెస్"]
    no_keywords_te = ["కాదు", "లేదు", "నో", "వద్దు", "కాదండి", "వద్దండి"]
    yes_keywords_en = ["yes", "yeah", "correct", "sure", "okay", "ok", "true", "right"]
    no_keywords_en = ["no", "nope", "not", "don't", "doesn't", "false", "wrong"]
    if language == "te":
        if any(keyword in text_lower for keyword in yes_keywords_te):
            return True
        if any(keyword in text_lower for keyword in no_keywords_te):
            return False
    if any(keyword in text_lower for keyword in yes_keywords_en):
        return True
    if any(keyword in text_lower for keyword in no_keywords_en):
        return False
    return None


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
    less_than_keywords_te = ["కంటే తక్కువ", "లోపు", "కంటే తక్కువ", "కంటే చిన్నది"]
    between_keywords_te = ["మధ్య", "నడుమ"]
    more_than_keywords_te = ["కంటే ఎక్కువ", "పైన", "కంటే పైన", "కంటే పెద్దది"]
    less_than_keywords_en = ["less than", "below", "under", "not more than"]
    between_keywords_en = ["between", "from", "in the range of", "ranging"]
    more_than_keywords_en = ["more than", "above", "over", "exceeding", "at least"]
    numbers = []
    words = text_lower.split()
    for word in words:
        clean_word = word.replace(",", "").replace(".", "").replace("rs", "").strip()
        if clean_word.isdigit():
            numbers.append(int(clean_word))
        elif language == "te":
            telugu_num = telugu_to_number(word)
            if telugu_num > 0:
                numbers.append(telugu_num)
    if language == "te":
        if any(keyword in text_lower for keyword in less_than_keywords_te) and numbers:
            return f"< {max(numbers)}"
        if any(keyword in text_lower for keyword in more_than_keywords_te) and numbers:
            return f"> {min(numbers)}"
        if any(keyword in text_lower for keyword in between_keywords_te) and len(numbers) >= 2:
            numbers.sort()
            return f"{numbers[0]} - {numbers[-1]}"
    if any(keyword in text_lower for keyword in less_than_keywords_en) and numbers:
        return f"< {max(numbers)}"
    if any(keyword in text_lower for keyword in more_than_keywords_en) and numbers:
        return f"> {min(numbers)}"
    if any(keyword in text_lower for keyword in between_keywords_en) and len(numbers) >= 2:
        numbers.sort()
        return f"{numbers[0]} - {numbers[-1]}"
    if numbers:
        return str(numbers[0])
    return None


def extract_scheme_names(text, language="te"):
    """
    Extract government scheme names from text.
    Args:
        text (str): Input text
        language (str): Language code (en/te)
    Returns:
        list: List of extracted scheme names
    """
    text = text.lower().strip()
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
    extracted_schemes = []
    for scheme, keywords in scheme_keywords.items():
        for keyword in keywords:
            if keyword in text:
                extracted_schemes.append(scheme)
                break
    return extracted_schemes


def get_db_connection():
    """Get a connection to the SQLite database.
    Returns:
        sqlite3.Connection: A connection to the database or None if an error occurs
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        return conn
    except sqlite3.Error as e:
        logging.debug(f"Database connection error: {e}")
        return None


def extract_caste(text):
    t = text.lower().strip()
    if t in ["sc", "st", "bc", "oc"]:
        return t.upper()
    return None

def extract_ownership(text: str, lang: str = "en") -> str | None:
    """Return 'own_house' or 'no_house' if we can tell; otherwise None."""
    t = text.lower().strip()
    yes_set = {"yes", "y", "own", "ownhouse", "have", "చ yes", "ఉంది", "ఆవున్", "ఆవును", "ఉన్నాం", "ఉన్నది", "ఉన్నాయి"}
    no_set  = {"no", "n", "rent", "rented", "nohouse", "లేదు", "వద్దు", "లేదని"}
    if t in yes_set:
        return "own_house"
    if t in no_set:
        return "no_house"
    return None

def profile_summary_en(profile):
    """Return a nicely formatted English summary."""
    parts = [
        f"Age: {profile.get('age', '-')}",
        f"Gender: {profile.get('gender', '-')}",
        f"Occupation: {profile.get('occupation', '-')}",
        f"Income: {profile.get('income', '-')}",
        f"Caste: {profile.get('caste', '-')}",
        f"Ownership: {profile.get('ownership', '-')}",
    ]
    if 'kids_count' in profile:
        parts.append(f"School-going kids: {profile['kids_count']}")
    return "\n".join(parts)


def profile_summary_te(profile):
    """Telugu version (add సెపరేటర్ only when kids_count exists)."""
    parts = [
        f"వయస్సు: {profile.get('age', '-')}",
        f"లింగం: {profile.get('gender', '-')}",
        f"వృత్తి: {profile.get('occupation', '-')}",
        f"ఆదాయం: {profile.get('income', '-')}",
        f"కులం: {profile.get('caste', '-')}",
        f"ఇల్లు/భూమి: {profile.get('ownership', '-')}",
    ]
    if 'kids_count' in profile:
        parts.append(f"పాఠశాల పిల్లలు: {profile['kids_count']}")
    return "\n".join(parts)

def extract_student_choice(text, language="te"):
    """Parses user input to determine choice between Upskilling (1) or Jobs (2)."""
    if not text:
        logging.debug("--- DEBUG extract_student_choice: Input is None or empty")
        return None
    text_lower = text.lower().strip()
    logging.debug(f"--- DEBUG extract_student_choice: Input='{text_lower}', Lang='{language}'") # Debug Input

    upskill_kw_te = ["ఒకటి", "నైపుణ్యం", "శిక్షణ", "స్కిల్"]
    upskill_kw_en = ["1", "one", "upskilling", "skill", "training"]
    jobs_kw_te = ["రెండు", "ఉద్యోగం", "ఉద్యోగాలు", "జాబ్", "జాబ్స్", "ఇంటర్న్షిప్"]
    jobs_kw_en = ["2", "two", "job", "jobs", "drive", "mela", "internship"]

    result = None # Default
    # Telugu keywords
    if language == 'te':
        if any(kw in text_lower for kw in upskill_kw_te):
            result = 1
        elif any(kw in text_lower for kw in jobs_kw_te):
            result = 2
    # English/digits (regardless of lang)
    if result is None:
        if any(kw in text_lower for kw in upskill_kw_en):
            result = 1
        elif any(kw in text_lower for kw in jobs_kw_en):
            result = 2
    logging.debug(f"--- DEBUG extract_student_choice: Returning: {result} (Type: {type(result)})") # Debug Return
    return result

# ... (rest of the code remains the same)

from typing import Tuple, List, Dict
import sqlite3, os

MONTHS_IN_YEAR = 12

def _dbg(label: str, result: bool):
    from flask import current_app
    tick = "✅" if result else "❌"
    current_app.logger.info(f"{label:40} {tick}")
    return result

def rule_thalliki(profile):
    """Return True if the household has at least one child in school and income ≤ 120 000."""
    has_child = profile.get("kids_count", 0) >= 1
    low_income = profile.get("income", 0) <= 120000
    return _dbg("kids_count >= 1", has_child) and _dbg("income <= 120000", low_income)

def rule_farmer(profile):
    return _dbg("occupation = farmer", profile.get("occupation") == "farmer")

def rule_farmer_bc(profile):
    return rule_farmer(profile) and _dbg("caste = bc", profile.get("caste") == "bc")

def match_schemes(profile: dict) -> list[dict]:
    """
    Return just the schemes the person CAN apply for.
    Each item must have *name* and *blurb* keys.
    """
    all_schemes = [
        {"name": "Free Solar Rooftop Scheme",
         "blurb": "Free rooftop solar panels for low-income households",
         "eligible": _dbg("income <= 10000", profile.get("income", 0) <= 10000)},
        {"name": "Annadata Sukhibhava",
         "blurb": "Rs.20,000 assistance to farmers in instalments",
         "eligible": rule_farmer(profile)},
        {"name": "Free Power Supply (9h)",
         "blurb": "Free daytime electricity for BC farmers",
         "eligible": rule_farmer_bc(profile)},
        {"name": "Thalliki Vandanam",
         "blurb": "₹15 000 per school-age child from poor households. *requires children info*",
         "eligible": rule_thalliki(profile)},
    ]
    return [s for s in all_schemes if s["eligible"]]

from typing import Tuple, List, Dict
import sqlite3, os

DB_FILE = os.getenv("DB_FILE", "welfare_schemes.db")

# --- Robust extract_number: Telugu/English/voice/digits ---
try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate, SCHEMES
except ImportError:
    transliterate = lambda x, _1, _2: x
    SCHEMES = {'telugu': None, 'itrans': None}
from word2number import w2n

def extract_number(text: str, lang: str = "en") -> int:
    """
    Extracts an integer from Telugu or English text, supporting digits, number words, and transliterated forms.
    Returns int or None.
    """
    if not text:
        return None
    txt = text.strip().lower()
    # 1. Try digit parse
    try:
        return int(txt)
    except Exception:
        pass
    # 2. Try Telugu transliteration → ITRANS
    try:
        itrans = transliterate(txt, SCHEMES['telugu'], SCHEMES['itrans'])
        # Remove spaces, normalize
        itrans = itrans.replace(' ', '')
        # Try parsing as English number word
        return w2n.word_to_num(itrans)
    except Exception:
        pass
    # 3. Try parsing as English number word directly
    try:
        return w2n.word_to_num(txt)
    except Exception:
        pass
    return None

def get_required_documents(scheme_name: str) -> list[str]:
    """
    Return a plain-English list of documents needed for the given scheme_name.
    Schema expected:
      documents_required (scheme_id TEXT, doc_name TEXT, is_mandatory INTEGER)
    """
    with get_db_connection() as conn:
        if conn is None:
            logging.error(f"Could not connect to database for scheme {scheme_name}")
            return ["Error: Could not connect to database."]
        try:
            df = pd.read_sql(
                '''SELECT doc_name, is_mandatory
                   FROM documents_required
                   WHERE scheme_id = (
                       SELECT scheme_id
                       FROM schemes
                       WHERE name = ?)''',
                conn,
                params=[scheme_name],
            )
            if df.empty:
                logging.warning(f"No documents found for scheme {scheme_name}")
                return ["No documents listed for this scheme."]
            result = []
            for _, row in df.iterrows():
                doc = row['doc_name']
                mandatory = row['is_mandatory']
                if mandatory in (1, '1', True, 'True', 'true'):
                    result.append(f"{doc} (Mandatory)")
                else:
                    result.append(f"{doc} (Optional)")
            return result
        except Exception as e:
            logging.error(f"Error fetching documents for {scheme_name}: {e}")
            return ["Error: Unable to fetch required documents."]
