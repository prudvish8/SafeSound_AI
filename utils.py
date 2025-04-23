# Minimal constants for filter_text and testing
GARBAGE_PHRASES_TE = ["బ్లా బ్లా", "అబద్ధం"]
GARBAGE_PHRASES_EN = ["asdfasdf", "blah blah"]
SHORT_SOUNDS_TE = ["హ్మ్"]
SHORT_SOUNDS_EN = ["hm"]
MIN_MEANINGFUL_LENGTH = 2

import unicodedata
import sqlite3

DB_FILE = "welfare_schemes.db"

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
    if normalize_nfc("ముప్పై") not in telugu_map: print("FATAL DEBUG: ముప్పై not in map after normalization!")
    if normalize_nfc("తొంభై") not in telugu_map: print("FATAL DEBUG: తొంభై not in map after normalization!")


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
    print(f"\n[telugu_to_number V8] Input: '{text}'")
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


    print(f"Normalized words (NFC): {normalized_words}")
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
                     print(f"[ParseSimpleWarn] Ignoring '{word}' in {sequence}")
        # print(f"  ParseSimple result: {val}") # Debug
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
                print(f"  Segment: {coeffs} '{mult_word}' -> {coeff}*{mult_value}={segment}. Total:{total_value}")
                indices_to_remove.extend(coeff_idx); indices_to_remove.append(i)
            i += 1
        if indices_to_remove:
             unique=sorted(list(set(indices_to_remove)), reverse=True); temp=words_to_process[:];
             for idx in unique:
                  if idx<len(temp): del temp[idx]
             words_to_process = temp
             if processed: print(f"  Remain after '{mult_word}': {words_to_process}")

    # 4. Process Remainder (using CORRECTED helper)
    if words_to_process: remaining=parse_simple_sequence(words_to_process); total_value+=remaining; print(f"  Remaining {words_to_process} -> {remaining}");

    # 5. Final Return (Return None if result is 0 for non-zero input)
    print(f"--> Final parsed number: {total_value}")
    is_zero = (len(normalized_words) == 1 and normalize_nfc("సున్నా") in normalized_words[0])
    if total_value == 0 and not is_zero:
        # Try digit fallback *only* if word parse failed AND it wasn't supposed to be 0
        try:
            digits="".join(filter(str.isdigit, text.replace(" ","")));
            if digits: print("Fallback: Digit extraction."); return int(digits)
        except ValueError: pass
        print("Warning: Resulted in 0 or parsing failed."); return None # Return None on failure
    else:
        return total_value # Return the calculated value (can be 0 if input was 'సున్నా')

from typing import Optional

# --- Helper: Text Filtering (Refactored & Type-Annotated) ---
def filter_text(raw_text: str, lang: str = "te") -> Optional[str]:
    """Filters garbage, short, or empty text, allowing essential short answers."""
    if not raw_text:
        print("[Filter] Empty input.")
        return None

    check = raw_text.lower().strip()
    if not check:
        print("[Filter] Empty after clean.")
        return None

    essential_short_te = ["ఓసి", "బిసి", "ఎస్సీ", "ఎస్టీ", "మగ", "ఆడ", "అవును", "కాదు", "లేదు", "ఉంది"]
    essential_short_en = ["hi", "hello","oc", "bc", "sc", "st", "yes", "no", "male", "female"]
    essentials = essential_short_te if lang == 'te' else essential_short_en

    g_list = GARBAGE_PHRASES_TE if lang == 'te' else GARBAGE_PHRASES_EN
    s_list = SHORT_SOUNDS_TE if lang == 'te' else SHORT_SOUNDS_EN

    if check in essentials:
        print(f"[Filter] Kept essential short answer: '{raw_text}'")
        return raw_text

    if check in g_list:
        print(f"[Filter] Filtered exact garbage: '{raw_text}'")
        return None

    if check in s_list:
        print(f"[Filter] Filtered short sound: '{raw_text}'")
        return None

    if len(raw_text) < MIN_MEANINGFUL_LENGTH:
        print(f"[Filter] Filtered too short (and not essential): '{raw_text}'")
        return None

    print(f"[Filter] Input passed filtering: '{raw_text}'")
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
    if any(keyword in text_lower for keyword in male_keywords_en):
        return "male"
    if any(keyword in text_lower for keyword in female_keywords_en):
        return "female"
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
        print(f"Database connection error: {e}")
        return None


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
    caste_map = {
        'oc': ['oc', 'ఓసి', 'ఓసీ', 'general', 'forward caste', 'open category'],
        'bc': ['bc', 'బిసి', 'బీసీ', 'backward class', 'other backward class'],
        'sc': ['sc', 'ఎస్సీ', 'scheduled caste', 'దళిత', 'హరిజన'],
        'st': ['st', 'ఎస్టీ', 'scheduled tribe', 'గిరిజన', 'tribal', 'adivasi', 'ఆదివాసి']
    }
    for category, keywords in caste_map.items():
        if any(keyword in text_lower for keyword in keywords):
            return category
    print(f"Could not extract caste category from: '{text}'")
    return None

# ... (rest of the code remains the same)
