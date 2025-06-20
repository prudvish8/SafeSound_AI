# --- START OF COMPLETE app.py ---

import os
import sqlite3
import tempfile # Needed if utils doesn't import it
import uuid
import warnings
import logging
import requests
from transformers import pipeline # Keep if whisper loading is here
import speech_recognition as sr # Keep if you plan to use mic directly elsewhere
import torch
import yaml # For loading prompts
from string import Template # For safe formatting
from flask import Flask, request, Response, abort, make_response
from twilio.twiml.messaging_response import MessagingResponse
from twilio.request_validator import RequestValidator
from werkzeug.middleware.proxy_fix import ProxyFix # For ngrok https fix
import pandas as pd
import json
import sys

# --- At the TOP of app.py ---
from dotenv import load_dotenv # Import load_dotenv
from nlu import extract_profile_from_text 

# --- Load .env *before* creating Flask app ---
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_success = load_dotenv(dotenv_path=dotenv_path, verbose=True)

# --- Create your Flask app ---
app = Flask(__name__)

# --- Only print debug info when FLASK_ENV=development or app.debug==True ---
if app.debug:
    app.logger.debug(f".env file load successful: {load_success}")
    app.logger.debug(f"TWILIO_ACCOUNT_SID: {os.getenv('TWILIO_ACCOUNT_SID')!r}")
    app.logger.debug(f"TWILIO_AUTH_TOKEN:   {os.getenv('TWILIO_AUTH_TOKEN')!r}")
    app.logger.debug(f"WHISPER_REMOTE_URL:  {os.getenv('WHISPER_REMOTE_URL')!r}")

# --- Now get the variables (as before) ---
TWILIO_AUTH_TOKEN = os.environ.get('TWILIO_AUTH_TOKEN')
TWILIO_ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID')

# --- Import functions from your utility file ---
try:
    from utils import (
        get_db_connection,
        profile_summary_te, 
        profile_summary_en,
        match_schemes,
        get_required_documents
    )
    app.logger.info("Successfully imported required functions from utils.py")
except ImportError as e:
    app.logger.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    app.logger.error(f"FATAL IMPORT ERROR from utils.py: Cannot find function - {e}")
    app.logger.error("Ensure utils.py contains all necessary functions.")
    app.logger.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    exit()
except Exception as e:
    app.logger.error(f"!!! FATAL ERROR during import from utils.py: {e} !!!")
    exit()

# -------------------------------------------------

# --- Configuration & Global Setup ---
#DB_FILE = "welfare_schemes.db"
DB_FILE = os.getenv("DB_FILE", "welfare_schemes.db")


# Define garbage lists and minimum length globally
GARBAGE_PHRASES_TE = ["సరే సరే", "ఓకే ఓకే", "థాంక్యూ", "ధన్యవాదాలు", "హమ్ హమ్", "హ్మ్ హ్మ్", "ఉహు ఉహు", "నమస్కారం", "హలో", "హాయ్", "ఓయ్", "ఏయ్", "ఏరా", "ఏంటి", "హు హు", "హూ హూ", "హు", "ఉమ్ ఉమ్"]
GARBAGE_PHRASES_EN = ["okay okay", "thank you", "thanks", "hello", "hi", "hey", "what", "alright", "right", "great", "fine", "well", "so", "hum hum"]
SHORT_SOUNDS_TE = ["ఆహ్", "ఉమ్", "సరే", "ఆహా", "ఉహు", "ఆం", "ఆమ్", "హుఁ", "హూఁ", "ఊఁ", "అయ్యో", "ఓ", "ఆఁ"]
SHORT_SOUNDS_EN = ["uh", "um", "hmm", "ok", "aha", "uh huh", "uh uh", "oh", "ah", "mm hmm", "mmm", "huh"]
MIN_MEANINGFUL_LENGTH = 2

SKIP_KEYWORDS_TE = ["తెలియదు", "నాకు తెలియదు", "వద్దు", "చెప్పను", "స్కిప్", "తరువాత", "తర్వాత"]
SKIP_KEYWORDS_EN = ["i don't know", "skip", "no", "pass", "none", "later"]

EXIT_KEYWORDS_EN = ["exit", "quit", "bye", "goodbye", "cancel"]
EXIT_KEYWORDS_TE = ["ఆపు", "వద్దు", "బయటకి", "ముగించు", "చాలు"]
EXIT_KEYWORDS_ALL = [w.lower() for w in EXIT_KEYWORDS_EN] + EXIT_KEYWORDS_TE

FOLLOW_UPS = {
    "9": {
        "next_state": "AWAITING_KIDS_COUNT",
        "prompt_te": "మీ ఇంట్లో చదువుతున్న పిల్లల సంఖ్య ఎంత?",
        "prompt_en": "How many school-going children do you have?"
    }
}

# ─── Load everything from prompts.yaml ─────────────────────────────────────
try:
    with open("prompts.yaml", "r", encoding="utf-8") as f:
        PROMPTS = yaml.safe_load(f)
    if not PROMPTS:
        raise ValueError("prompts.yaml loaded empty")
    INTENT_KEYWORDS = PROMPTS.get("INTENT_KEYWORDS", {})
    app.logger.info("Successfully loaded prompts and intent‐keywords from prompts.yaml")
except FileNotFoundError:
    app.logger.error("FATAL: prompts.yaml not found")
    sys.exit(1)
except Exception as e:
    app.logger.error(f"FATAL: error parsing prompts.yaml: {e}")
    sys.exit(1)
# ─────────────────────────────────────────────────────────────────────────

def get_prompt(step: str, lang: str, key: str = "initial", default: str = "", **fmt_vars):
    """Fetch and interpolate a prompt from PROMPTS."""
    # safe descent into dict
    block = PROMPTS \
      .get("CONVERSATION_PROMPTS", {}) \
      .get(step, {}) \
      .get(lang, {})
    tpl = block.get(key) or block.get("initial") or block.get("q") or block.get("retry")
    if not tpl:
        app.logger.warning(f"Missing prompt for {step}/{lang}/{key}")
        return default
    return Template(tpl).safe_substitute(**fmt_vars)

from typing import Optional
def match_intent(state: str, lang: str, text: str) -> Optional[str]:
    """Exact‐match user text against your YAML‐driven keywords."""
    text = text.lower().strip()
    kws_for_state = INTENT_KEYWORDS.get(state, {}).get(lang, {})
    for intent, words in kws_for_state.items():
        # normalize each word once
        normalized = [w.lower().strip() for w in words]
        if text in normalized:
            return intent
    return None


# --- Helper: Get Intent Keywords from YAML ---
def get_intent_keywords(state, lang, key=None):
    """Fetch intent keywords for a given state and language from INTENT_KEYWORDS (YAML). Optionally filter by key (e.g., 'start', 'skip')."""
    try:
        kw_dict = INTENT_KEYWORDS.get(state, {}).get(lang, {})
        if key:
            return kw_dict.get(key, [])
        return kw_dict
    except Exception as e:
        app.logger.error(f"Error getting intent keywords for {state}/{lang}/{key}: {e}")
        return []
# --- End intent keyword helper ---

# --- Helper: normalise choice for menu selections ---
def normalise_choice(raw: str, lang: str) -> str:
    """
    Turn 'ఫోర్', 'four', '4', '04', 'నాలుగు' → '4'
    Leave everything else unchanged (trim, lower).
    """
    if not raw:
        return ""
    txt = raw.strip().lower()
    num = telugu_to_number(txt)
    if num is not None:
        return str(num)
    if txt.isdigit():
        return str(int(txt))
    return txt

# --- Helper: normalise occupation ---
def normalise_occupation(text: str) -> str:
    mapping = {
        'housewife':'homemaker', 'house wife':'homemaker',
        'home maker':'homemaker', 'homemaker':'homemaker'
    }
    t = text.lower().strip()
    return mapping.get(t, t)

# --- Helper: WhatsApp Menu Formatting ---
def menu_lines(schemes: list[dict]) -> list[str]:
    lines = ["Type a number or scheme name (or 'No'):"]
    for i, s in enumerate(schemes, 1):
        lines.append(f"{i}) {s['name']} – {s['blurb']}")
    return lines

# --- Helper: Resolve Scheme ---
def resolve_scheme(reply, menu):
    """Turn '1' or 'annadata' into the chosen scheme dict (or None)."""
    reply = (reply or '').strip().lower()
    if reply.isdigit():
        idx = int(reply) - 1
        return menu[idx] if 0 <= idx < len(menu) else None
    for s in menu:
        if reply in s["name"].lower():
            return s
    return None

# --- Configure Flask's Own Logger ---
app.logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] - %(message)s')
handler.setFormatter(formatter)
if not app.logger.handlers:
    app.logger.addHandler(handler)

# Suppress specific warnings (Keep these)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.models.whisper")
warnings.filterwarnings("ignore", message=".*attention mask.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*deprecated.*", category=FutureWarning)

# --- Twilio Auth & Validation Setup ---
if not TWILIO_AUTH_TOKEN or not TWILIO_ACCOUNT_SID:
    app.logger.warning("!!! Env vars TWILIO_AUTH_TOKEN or TWILIO_ACCOUNT_SID not set. Validation might skip/fail. Media download WILL fail. !!!")
validator = RequestValidator(TWILIO_AUTH_TOKEN) if TWILIO_AUTH_TOKEN else None

# --- Whisper Model Loading ---
# Load globally once on startup
WHISPER_MODEL = None
app.logger.info("Loading Whisper model...")
try:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    app.logger.info(f"Using device: {device} for Whisper")
    WHISPER_MODEL = pipeline("automatic-speech-recognition", model="vasista22/whisper-telugu-base", chunk_length_s=30, device=device)
    app.logger.info("Whisper model loaded successfully.")
except Exception as e:
    app.logger.error(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    app.logger.error(f"FATAL: Failed to load Whisper model: {e}")
    app.logger.error("Check model name, internet connection, and dependencies (torch, transformers).")
    app.logger.error(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    WHISPER_MODEL = None # Ensure it's None if loading failed

# --- Persistent User State Management ---
def init_state_db():
    conn = sqlite3.connect('user_states.db')
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE IF NOT EXISTS user_states (user_number TEXT PRIMARY KEY, state_data TEXT)')
    conn.commit()
    conn.close()

def save_user_state(user_number: str, state_data: dict):
    conn = sqlite3.connect('user_states.db')
    cursor = conn.cursor()
    cursor.execute('INSERT OR REPLACE INTO user_states (user_number, state_data) VALUES (?, ?)',
                   (user_number, json.dumps(state_data)))
    conn.commit()
    conn.close()

def load_user_state(user_number: str) -> dict:
    conn = sqlite3.connect('user_states.db')
    cursor = conn.cursor()
    cursor.execute('SELECT state_data FROM user_states WHERE user_number = ?', (user_number,))
    result = cursor.fetchone()
    conn.close()
    return json.loads(result[0]) if result else {'step': 'START', 'profile': {}, 'language': 'en'}

# --- Flask App Initialization & Proxy Fix ---
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1) # Handle proxy headers (like ngrok's https)

from typing import Optional

# --- Helper: Text Filtering (Refactored & Type-Annotated) ---
def filter_text(raw_text: str, lang: str = "te") -> Optional[str]:
    """Filters garbage, short, or empty text, allowing essential short answers AND digits."""
    if not raw_text:
        logging.info("[Filter] Empty input.")
        return None

    check = raw_text.lower().strip()
    if not check:
        logging.info("[Filter] Empty after clean.")
        return None

    # --- *** NEW: Allow digits immediately *** ---
    if check.isdigit():
        logging.info(f"[Filter] Kept digit input: '{raw_text}'")
        return raw_text # Return original text if it's purely digits
    # --- End New Check ---

    essential_short_te = ["ఓసి", "బిసి", "ఎస్సీ", "ఎస్టీ", "మగ", "ఆడ", "అవును", "కాదు", "లేదు", "ఉంది"]
    essential_short_en = ["oc", "bc", "sc", "st", "yes", "no", "male", "female"]
    activation_all = ["hi", "hello", "babu", "బాబు"]
    essentials = (essential_short_te if lang == 'te' else essential_short_en) + activation_all

    g_list = GARBAGE_PHRASES_TE if lang == 'te' else GARBAGE_PHRASES_EN
    s_list = SHORT_SOUNDS_TE if lang == 'te' else SHORT_SOUNDS_EN

    # 1. Check if essential (already handled digits above)
    if check in essentials:
        logging.info(f"[Filter] Kept essential word: '{raw_text}'")
        return raw_text

    # 2. Check garbage (and NOT essential)
    if check in g_list:
        logging.info(f"[Filter] Exact garbage: '{raw_text}'")
        return None

    # 3. Check short sound (and NOT essential)
    if check in s_list:
        logging.info(f"[Filter] Short sound: '{raw_text}'")
        return None

    # 4. Check length (and NOT essential/digit) - Use '<' strictly
    if len(raw_text) < MIN_MEANINGFUL_LENGTH:
        logging.info(f"[Filter] Too short (and not essential/digit): '{raw_text}'")
        return None

    # Passed all filters
    logging.info(f"[Filter] Input passed: '{raw_text}'")
    return raw_text

# --- End Refactored Filter ---

def is_exit(text: str) -> bool:
    if not text:
        return False
    cleaned = "".join(ch for ch in text.lower() if ch.isalpha())
    return cleaned in EXIT_KEYWORDS_ALL

# --- Main Webhook Route ---
@app.route('/whatsapp', methods=['POST'])
def whatsapp_webhook():
    request_id = str(uuid.uuid4())[:8]
    app.logger.info(f"--- Request {request_id}: /whatsapp HIT ---")
    app.logger.info(f"--- STATE AT START of webhook: {load_user_state(request.values.get('From'))}") # Log full state dict

    # 1. --- Request Validation (skip in TESTING mode) ---
    if not app.config.get("TESTING", False) and validator:
        url = request.url
        post_vars = request.form.to_dict()
        sig = request.headers.get("X-Twilio-Signature")
        is_valid = validator.validate(url, post_vars, sig)
        if not is_valid:
            app.logger.error(f"Req {request_id}: !!! Validation FAILED - 403 !!! Check Token/URL.");
            return make_response("Validation failed", 403)
        else:
            app.logger.info(f"Req {request_id}: Validation OK.")
    else:
        app.logger.debug("Skipping Twilio validation (TESTING mode).")

    # 2. --- Initialize Reply & State ---
    sender_id = request.values.get('From', None)
    processed_text = None
    # Default reply - set this specifically in except block if needed
    reply_text = None # Initialize reply_text to None to clearly track if it was set
    user_state = load_user_state(sender_id) # Get or init state
    current_lang = user_state.get('language', 'te')
    profile = user_state.get('profile', {})
    app.logger.info(f"Req {request_id}: Initial state: {user_state}")

    # --- *** ADD THIS LINE HERE *** ---
    skip_kw = SKIP_KEYWORDS_TE + SKIP_KEYWORDS_EN # Make combined list accessible locally
    # ---------------------------------

    # 3. --- Process Input & State Machine (Wrapped in Try/Except) ---
    try:
        # --- Process Input (Voice or Text) ---
        num_media = int(request.values.get('NumMedia', 0)); is_voice = False;
        media_url = request.values.get('MediaUrl0') if num_media > 0 else None;
        media_ct = request.values.get('MediaContentType0') if num_media > 0 else None;
        is_voice = bool(media_url and media_ct and 'audio' in media_ct);

        if is_voice:
            # Download, convert, and transcribe voice message
            processed_text = process_voice_message(media_url, request_id)
            pass # Make sure this logic correctly sets processed_text or handles errors
        else: # Text
            processed_text = request.values.get('Body', '').strip()
            app.logger.info(f"Req {request_id}: TEXT message: '{processed_text}'")

        # Update language state after getting input
        if processed_text is not None: # Check if we have text before lang detect
            is_tel=any(0x0c00 <= ord(c) <= 0x0c7f for c in processed_text); is_eng=any('a'<=c.lower()<='z' for c in processed_text);
            if is_eng and not is_tel: lang_detected='en'
            elif is_tel: lang_detected='te'
            else: lang_detected=current_lang # Fallback to current if mixed
            user_state['language'] = lang_detected; current_lang = lang_detected; # Update state & current lang
            app.logger.info(f"Req {request_id}: Language state updated to: {current_lang}")
            # Filter text AFTER determining language
            processed_text = filter_text(processed_text, current_lang)
            app.logger.info(f"Req {request_id}: Text after filtering: '{processed_text}'")
        # else: Keep previous reply_text (often None or error from voice processing)

        app.logger.debug(f"EXIT check raw: {repr(processed_text)}  raw cleaned: "
                         f"{''.join(ch for ch in processed_text.lower() if ch.isalpha())}")

        if is_exit(processed_text):
            bye = "సరే, సెషన్ ముగుస్తోంది. నమస్కారం!" if current_lang == "te" \
                  else "Okay, ending the session. Goodbye!"
            save_user_state(sender_id, {'step': 'START', 'profile': {}, 'language': current_lang})
            resp = MessagingResponse(); resp.message(bye)
            app.logger.info(f"Req {request_id}: EXIT detected  session reset")
            return Response(str(resp), mimetype="text/xml")

        # --- GLOBAL EXIT CHECK ---
        if processed_text and processed_text.lower().strip() in EXIT_KEYWORDS_ALL:
            logging.info(f"Req {request_id}: Exit command '{processed_text}' detected. Ending session.")
            # Clear state for this user
            save_user_state(sender_id, {'step': 'START', 'profile': {}, 'language': current_lang})
            # Send goodbye message
            bye_t="సరే, సెషన్ ముగిస్తున్నాను. నమస్కారం!"; bye_e="Okay, ending the session. Goodbye!"
            reply_text = bye_t if current_lang == 'te' else bye_e
            # Build and return response immediately
            response_twiml = MessagingResponse(); response_twiml.message(reply_text);
            logging.info(f"Req {request_id}: Sending TwiML reply: '{reply_text}'")
            return Response(str(response_twiml), mimetype='text/xml')

        # --- Core State Machine ---
        current_step = user_state.get('step', 'START')
        app.logger.info(f"Req {request_id}: Executing state='{current_step}', Input='{processed_text}'")

        if current_step == 'START':
            activation_kws = get_intent_keywords('START', current_lang, key='start')
            skip_kws = get_intent_keywords('START', current_lang, key='skip')
            if processed_text:
                answer_lower = processed_text.lower()
                if any(w in answer_lower for w in activation_kws):
                    # Combine greeting and target question
                    greeting = get_prompt('START', current_lang, key='initial')
                    target_q = get_prompt('AWAITING_TARGET', current_lang, key='q')
                    reply_text = f"{greeting}\n\n{target_q}"
                    user_state['step'] = 'AWAITING_TARGET'
                    save_user_state(sender_id, user_state)
                    resp = MessagingResponse()
                    resp.message(reply_text)
                    RED = "\033[91m"
                    BOLD = "\033[1m"
                    RESET = "\033[0m"
                    FRAME = "=" * 30
                    app.logger.info(f"{RED}{BOLD}\n{FRAME}\nTwiML to user: {reply_text}\n{FRAME}{RESET}")
                    return Response(str(resp), mimetype='text/xml')
                elif any(w in answer_lower for w in skip_kws):
                    # Handle skip logic if needed
                    reply_text = get_prompt('START', current_lang, key='skip_ack')
                else:
                    reply_text = get_prompt('START', current_lang, key='retry')
            else:
                reply_text = get_prompt('START', current_lang, key='retry')

        elif current_step == 'AWAITING_TARGET':
            self_kws = get_intent_keywords('AWAITING_TARGET', current_lang, key='self')
            other_kws = get_intent_keywords('AWAITING_TARGET', current_lang, key='other')
            extra_others = ['someone', 'someone else', 'other', 'them']
            other_kws = list(set(other_kws + extra_others))
            # Add 'me' to self keywords for English input
            extra_self = ['me']
            self_kws = list(set(self_kws + extra_self))
            if processed_text:
                answer_lower = processed_text.lower()
                yes_keywords = ['yes', 'అవును']
                if any(w in answer_lower or answer_lower in w for w in other_kws):
                    profile['target'] = 'other'
                    user_state['step'] = 'AWAITING_RELATION'
                    reply_text = get_prompt('AWAITING_RELATION', current_lang, key='q')
                    save_user_state(sender_id, user_state)
                    resp = MessagingResponse()
                    resp.message(reply_text)
                    RED = "\033[91m"
                    BOLD = "\033[1m"
                    RESET = "\033[0m"
                    FRAME = "=" * 30
                    app.logger.info(f"{RED}{BOLD}\n{FRAME}\nTwiML to user: {reply_text}\n{FRAME}{RESET}")
                    return Response(str(resp), mimetype='text/xml')
                elif any(w in answer_lower for w in self_kws) or any(y in answer_lower for y in yes_keywords):
                    profile['target'] = 'self'
                    user_state['step'] = 'AWAITING_AGE'
                    reply_text = get_prompt('AWAITING_AGE', current_lang, key='initial', pron_e="Your", pron_t="మీ")
                else:
                    reply_text = get_prompt('AWAITING_TARGET', current_lang, key='retry')
            else:
                reply_text = get_prompt('AWAITING_TARGET', current_lang, key='retry')

        elif current_step == 'AWAITING_RELATION':
            if processed_text:
                profile['relation'] = processed_text
                user_state['step'] = 'AWAITING_AGE'
                reply_text = get_prompt('AWAITING_AGE', current_lang, key='initial', pron_e="Their", pron_t="వారి")
            else:
                reply_text = get_prompt('AWAITING_RELATION', current_lang, key='retry')

        elif current_step == 'AWAITING_AGE':
            app.logger.info(f"Processing Age input: '{processed_text}'")
            parsed_age = None
            skip_kw = SKIP_KEYWORDS_TE if current_lang == 'te' else SKIP_KEYWORDS_EN
            is_skip = bool(processed_text and any(s in processed_text.lower() for s in skip_kw))
            next_step = current_step
            step_passed = False
            # --- Get retry flag ---
            age_retry_flag = user_state.get('age_retry_flag', False)

            if is_skip:
                app.logger.info("User skipped age."); profile['age'] = None;
                next_step = 'AWAITING_GENDER'; step_passed = True;
                target = profile.get('target', 'self')
                pron_t = "వారి" if target == 'other' else "మీ"
                pron_e = "Their" if target == 'other' else "Your"
                qt = f"{pron_t} లింగం? (మగ/ఆడ)"
                qe = f"{pron_e} gender? (Male/Female)"
                reply_text = qt if current_lang == 'te' else qe
                save_user_state(sender_id, user_state)
                user_state.pop('age_retry_flag', None)
            elif processed_text:
                try:
                    parsed_age = telugu_to_number(processed_text)
                    profile['age'] = parsed_age
                    app.logger.info(f"Parsed age value: {parsed_age} (Type: {type(parsed_age)})")
                    if parsed_age is not None and 1 <= parsed_age < 120:
                        if parsed_age <= 21:
                            app.logger.info("Age <= 21. Moving to AWAITING_STUDENT_STATUS.")
                            next_step = 'AWAITING_STUDENT_STATUS'
                            step_passed = True
                            qt = "మీరు విద్యార్థినా? (అవును/కాదు)"
                            qe = "Are you a student? (Yes/No)"
                            reply_text = qt if current_lang == 'te' else qe
                        else:
                            app.logger.info("Age > 21. Moving to AWAITING_GENDER.")
                            next_step = 'AWAITING_GENDER'
                            step_passed = True
                            target = profile.get('target', 'self')
                            pron_t = "మీ" if target == 'self' else "వారి"
                            pron_e = "Your" if target == 'self' else "Their"
                            qt = f"{pron_t} లింగం? (మగ/ఆడ)"
                            qe = f"{pron_e} gender? (Male/Female)"
                            reply_text = qt if current_lang == 'te' else qe
                        save_user_state(sender_id, user_state)
                        user_state.pop('age_retry_flag', None) # Clear flag on success
                    else:
                        app.logger.warning(f"Invalid age/parse fail: {parsed_age}")
                        profile['age'] = None
                        # --- Check retry flag ---
                        if age_retry_flag:
                            app.logger.warning("Second failed attempt for age. Skipping.")
                            qt="క్షమించండి, వయస్సు ఇంకా అర్థం కాలేదు. దాటవేస్తున్నాను. మీ లింగం?"
                            qe="Sorry, still couldn't get the age. Skipping for now. Your gender?"
                            reply_text=qt if current_lang=='te' else qe;
                            next_step = 'AWAITING_GENDER'; step_passed = True;
                            save_user_state(sender_id, user_state)
                            user_state.pop('age_retry_flag', None)
                        else:
                            qt = "వయస్సు సంఖ్య రూపంలో (1-120) చెప్పండి?"
                            qe = "Please state age as a number (1-120)."
                            reply_text = qt if current_lang == 'te' else qe
                            save_user_state(sender_id, user_state)
                            user_state['age_retry_flag'] = True
                except Exception as e:
                    app.logger.error(f"Error processing age: {e}", exc_info=True)
                    profile['age'] = None
                    reply_text = "Error processing age."
                    next_step = 'AWAITING_AGE'; step_passed = False
            else:
                qt = "వయస్సు చెప్పండి?"
                qe = "Please state the age."
                reply_text = qt if current_lang == 'te' else qe
                next_step = 'AWAITING_AGE'; step_passed = False

            user_state['step'] = next_step

        elif current_step == 'AWAITING_GENDER':
            male_kws = get_intent_keywords('AWAITING_GENDER', current_lang, key='male')
            female_kws = get_intent_keywords('AWAITING_GENDER', current_lang, key='female')
            skip_kws = get_intent_keywords('AWAITING_GENDER', current_lang, key='skip')
            if processed_text:
                answer_lower = processed_text.lower()
                if answer_lower in male_kws:
                    profile['gender'] = 'male'
                    user_state['step'] = 'AWAITING_OCCUPATION'
                    reply_text = get_prompt('AWAITING_OCCUPATION', current_lang, key='initial', pron_e="Their" if profile.get('target')=="other" else "Your", pron_t="వారి" if profile.get('target')=="other" else "మీ")
                elif answer_lower in female_kws:
                    profile['gender'] = 'female'
                    user_state['step'] = 'AWAITING_OCCUPATION'
                    reply_text = get_prompt('AWAITING_OCCUPATION', current_lang, key='initial', pron_e="Their" if profile.get('target')=="other" else "Your", pron_t="వారి" if profile.get('target')=="other" else "మీ")
                elif any(w in answer_lower for w in skip_kws):
                    user_state['step'] = 'AWAITING_OCCUPATION'
                    reply_text = get_prompt('AWAITING_GENDER', current_lang, key='skip_ack') + "\n" + get_prompt('AWAITING_OCCUPATION', current_lang, key='initial', pron_e="Their" if profile.get('target')=="other" else "Your", pron_t="వారి" if profile.get('target')=="other" else "మీ")
                else:
                    reply_text = get_prompt('AWAITING_GENDER', current_lang, key='retry')
            else:
                reply_text = get_prompt('AWAITING_GENDER', current_lang, key='retry')

        elif current_step == 'AWAITING_STUDENT_STATUS':
            yes_kws = get_intent_keywords('AWAITING_STUDENT_STATUS', current_lang, key='yes')
            no_kws = get_intent_keywords('AWAITING_STUDENT_STATUS', current_lang, key='no')
            skip_kws = get_intent_keywords('AWAITING_STUDENT_STATUS', current_lang, key='skip')
            if processed_text:
                answer_lower = processed_text.lower()
                if any(w in answer_lower for w in yes_kws):
                    profile['is_student'] = True
                    user_state['step'] = 'AWAITING_STUDENT_CHOICE'
                    reply_text = get_prompt('AWAITING_STUDENT_CHOICE', current_lang, key='initial')
                elif any(w in answer_lower for w in no_kws):
                    profile['is_student'] = False
                    user_state['step'] = 'AWAITING_GENDER'
                    reply_text = get_prompt('AWAITING_GENDER', current_lang, key='initial')
                elif any(w in answer_lower for w in skip_kws):
                    user_state['step'] = 'AWAITING_GENDER'
                    reply_text = get_prompt('AWAITING_STUDENT_STATUS', current_lang, key='skip_ack') + "\n" + get_prompt('AWAITING_GENDER', current_lang, key='initial')
                else:
                    reply_text = get_prompt('AWAITING_STUDENT_STATUS', current_lang, key='retry')
            else:
                reply_text = get_prompt('AWAITING_STUDENT_STATUS', current_lang, key='retry')

        elif current_step == 'AWAITING_STUDENT_CHOICE':
            upskilling_kws = get_intent_keywords('AWAITING_STUDENT_CHOICE', current_lang, key='upskilling')
            job_kws = get_intent_keywords('AWAITING_STUDENT_CHOICE', current_lang, key='job')
            skip_kws = get_intent_keywords('AWAITING_STUDENT_CHOICE', current_lang, key='skip')
            if processed_text:
                answer_lower = processed_text.lower()
                if any(w in answer_lower for w in upskilling_kws):
                    profile['student_choice'] = 'upskilling'
                    user_state['step'] = 'AWAITING_GENDER'
                    reply_text = get_prompt('AWAITING_GENDER', current_lang, key='initial')
                elif any(w in answer_lower for w in job_kws):
                    profile['student_choice'] = 'job'
                    user_state['step'] = 'AWAITING_GENDER'
                    reply_text = get_prompt('AWAITING_GENDER', current_lang, key='initial')
                elif any(w in answer_lower for w in skip_kws):
                    user_state['step'] = 'AWAITING_GENDER'
                    reply_text = get_prompt('AWAITING_STUDENT_CHOICE', current_lang, key='skip_ack') + "\n" + get_prompt('AWAITING_GENDER', current_lang, key='initial')
                else:
                    reply_text = get_prompt('AWAITING_STUDENT_CHOICE', current_lang, key='retry')
            else:
                reply_text = get_prompt('AWAITING_STUDENT_CHOICE', current_lang, key='retry')

        elif current_step == 'AWAITING_CASTE':
            oc_kws = get_intent_keywords('AWAITING_CASTE', current_lang, key='oc')
            bc_kws = get_intent_keywords('AWAITING_CASTE', current_lang, key='bc')
            sc_kws = get_intent_keywords('AWAITING_CASTE', current_lang, key='sc')
            st_kws = get_intent_keywords('AWAITING_CASTE', current_lang, key='st')
            skip_kws = get_intent_keywords('AWAITING_CASTE', current_lang, key='skip')
            if processed_text:
                answer_lower = processed_text.lower()
                if any(w in answer_lower for w in oc_kws):
                    profile['caste'] = 'oc'
                    user_state['step'] = 'AWAITING_OWNERSHIP'
                    target = profile.get('target', 'self')
                    pron_e = "you" if target == "self" else "they"
                    pron_t = "మీకు" if target == "self" else "వారికి"
                    reply_text = get_prompt('AWAITING_OWNERSHIP', current_lang, key='initial', pron_e=pron_e, pron_t=pron_t)
                elif any(w in answer_lower for w in bc_kws):
                    profile['caste'] = 'bc'
                    user_state['step'] = 'AWAITING_OWNERSHIP'
                    target = profile.get('target', 'self')
                    pron_e = "you" if target == "self" else "they"
                    pron_t = "మీకు" if target == "self" else "వారికి"
                    reply_text = get_prompt('AWAITING_OWNERSHIP', current_lang, key='initial', pron_e=pron_e, pron_t=pron_t)
                elif any(w in answer_lower for w in sc_kws):
                    profile['caste'] = 'sc'
                    user_state['step'] = 'AWAITING_OWNERSHIP'
                    target = profile.get('target', 'self')
                    pron_e = "you" if target == "self" else "they"
                    pron_t = "మీకు" if target == "self" else "వారికి"
                    reply_text = get_prompt('AWAITING_OWNERSHIP', current_lang, key='initial', pron_e=pron_e, pron_t=pron_t)
                elif any(w in answer_lower for w in st_kws):
                    profile['caste'] = 'st'
                    user_state['step'] = 'AWAITING_OWNERSHIP'
                    target = profile.get('target', 'self')
                    pron_e = "you" if target == "self" else "they"
                    pron_t = "మీకు" if target == "self" else "వారికి"
                    reply_text = get_prompt('AWAITING_OWNERSHIP', current_lang, key='initial', pron_e=pron_e, pron_t=pron_t)
                elif any(w in answer_lower for w in skip_kws):
                    user_state['step'] = 'AWAITING_OWNERSHIP'
                    target = profile.get('target', 'self')
                    pron_e = "you" if target == "self" else "they"
                    pron_t = "మీకు" if target == "self" else "వారికి"
                    reply_text = get_prompt('AWAITING_CASTE', current_lang, key='skip_ack') + "\n" + get_prompt('AWAITING_OWNERSHIP', current_lang, key='initial', pron_e=pron_e, pron_t=pron_t)
                else:
                    reply_text = get_prompt('AWAITING_CASTE', current_lang, key='retry')
            else:
                reply_text = get_prompt('AWAITING_CASTE', current_lang, key='retry')

        elif current_step == 'AWAITING_OWNERSHIP':
            yn = extract_yes_no(processed_text, current_lang) if processed_text else None
            if yn is not None:
                profile['ownership'] = bool(yn)
                age = profile.get('age')
                if age is not None and age < 21:
                    user_state['step'] = 'AWAITING_CONFIRMATION'
                    summary = profile_summary_en(profile) if current_lang == 'en' else profile_summary_te(profile)
                    reply_text = get_prompt('AWAITING_CONFIRMATION', current_lang, 'initial', summary=summary)
                else:
                    user_state['step'] = 'AWAITING_KIDS_COUNT'
                    reply_text = get_prompt('AWAITING_KIDS_COUNT', current_lang, 'initial')
            else:
                reply_text = get_prompt('AWAITING_OWNERSHIP', current_lang, 'retry')
            save_user_state(sender_id, user_state)

        elif current_step == 'AWAITING_KIDS_COUNT':
            skip_kws = get_intent_keywords('AWAITING_KIDS_COUNT', current_lang, 'skip')
            if processed_text and processed_text.lower() in skip_kws:
                profile['kids_count'] = None
            else:
                num = telugu_to_number(processed_text) if processed_text else None
                if num is None or num < 0:
                    reply_text = get_prompt('AWAITING_KIDS_COUNT', current_lang, 'retry')
                    save_user_state(sender_id, user_state)
                    resp = MessagingResponse(); resp.message(reply_text)
                    return Response(str(resp), mimetype='text/xml')
                profile['kids_count'] = num
            user_state['step'] = 'AWAITING_CONFIRMATION'
            summary = profile_summary_en(profile) if current_lang == 'en' else profile_summary_te(profile)
            reply_text = get_prompt('AWAITING_CONFIRMATION', current_lang, 'initial', summary=summary)
            save_user_state(sender_id, user_state)

        elif current_step == 'AWAITING_CONFIRMATION':
            yes_kws = get_intent_keywords('AWAITING_CONFIRMATION', current_lang, 'yes') or ['yes', 'అవును']
            no_kws  = get_intent_keywords('AWAITING_CONFIRMATION', current_lang, 'no')  or ['no', 'కాదు']
            ans = (processed_text or "").strip().lower()
            if ans in yes_kws:
                profile['confirmed'] = True
                schemes = match_schemes(profile)
                if schemes:
                    menu = ["Type a number or scheme name (or 'No'):"]
                    for i, s in enumerate(schemes, 1):
                        menu.append(f"{i}) {s['name']} – {s['blurb']}")
                    reply_text = "\n".join(menu)
                    user_state['matched_schemes'] = schemes
                    user_state['step'] = 'AWAITING_SCHEME_INTEREST'
                else:
                    reply_text = "Sorry, no schemes matched right now."
                    user_state['step'] = 'COMPLETED'
            elif ans in no_kws:
                profile.clear()
                save_user_state(sender_id, {'step':'START','profile':{},'language':current_lang})
                reply_text = get_prompt('START', current_lang, key='retry')
            else:
                reply_text = get_prompt('AWAITING_CONFIRMATION', current_lang, key='retry')
            save_user_state(sender_id, user_state)

        elif current_step == 'AWAITING_CONTACT_PREFERENCE':
            whatsapp_kws = get_intent_keywords('AWAITING_CONTACT_PREFERENCE', current_lang, key='whatsapp')
            phone_kws = get_intent_keywords('AWAITING_CONTACT_PREFERENCE', current_lang, key='phone')
            sms_kws = get_intent_keywords('AWAITING_CONTACT_PREFERENCE', current_lang, key='sms')
            skip_kws = get_intent_keywords('AWAITING_CONTACT_PREFERENCE', current_lang, key='skip')
            if processed_text:
                answer_lower = processed_text.lower()
                if any(w in answer_lower for w in whatsapp_kws):
                    profile['contact_preference'] = 'whatsapp'
                    user_state['step'] = 'AWAITING_SCHEME_INTEREST'
                    reply_text = get_prompt('AWAITING_SCHEME_INTEREST', current_lang, key='initial')
                elif any(w in answer_lower for w in phone_kws):
                    profile['contact_preference'] = 'phone'
                    user_state['step'] = 'AWAITING_SCHEME_INTEREST'
                    reply_text = get_prompt('AWAITING_SCHEME_INTEREST', current_lang, key='initial')
                elif any(w in answer_lower for w in sms_kws):
                    profile['contact_preference'] = 'sms'
                    user_state['step'] = 'AWAITING_SCHEME_INTEREST'
                    reply_text = get_prompt('AWAITING_SCHEME_INTEREST', current_lang, key='initial')
                elif any(w in answer_lower for w in skip_kws):
                    user_state['step'] = 'AWAITING_SCHEME_INTEREST'
                    reply_text = get_prompt('AWAITING_CONTACT_PREFERENCE', current_lang, key='skip_ack') + "\n" + get_prompt('AWAITING_SCHEME_INTEREST', current_lang, key='initial')
                else:
                    reply_text = get_prompt('AWAITING_CONTACT_PREFERENCE', current_lang, key='retry')
            else:
                reply_text = get_prompt('AWAITING_CONTACT_PREFERENCE', current_lang, key='retry')

        elif current_step == 'AWAITING_SCHEME_INTEREST':
            if not processed_text:
                reply_text = get_prompt('AWAITING_SCHEME_INTEREST', current_lang, 'retry')
            else:
                answer = processed_text.strip()
                picked = None
                # 1) Did they send a pure number?
                if answer.isdigit():
                    idx = int(answer) - 1
                    schemes = user_state.get('matched_schemes', [])
                    if 0 <= idx < len(schemes):
                        picked = schemes[idx]
                # 2) Otherwise, try case-insensitive name match
                if picked is None:
                    schemes = user_state.get('matched_schemes', [])
                    answer_lower = answer.lower()
                    for s in schemes:
                        if answer_lower in s['name'].lower():
                            picked = s
                            break
                # 3) If still nothing → retry
                if picked is None:
                    reply_text = get_prompt('AWAITING_SCHEME_INTEREST', current_lang, 'retry')
                else:
                    docs = get_required_documents(picked['name'])
                    doc_lines = "\n".join(f"• {d}" for d in docs)
                    reply_text = (f"The documents required for *{picked['name']}* are:\n"
                                  f"{doc_lines}\n\n"
                                  f"We’ll verify them and get back to you. Thank you!")
                    user_state['step'] = 'COMPLETED'
            save_user_state(sender_id, user_state)

        elif current_step == 'AWAITING_OCCUPATION':
            if processed_text:
                profile['occupation'] = normalise_occupation(processed_text)
                user_state['step'] = 'AWAITING_INCOME'
                reply_text = get_prompt('AWAITING_INCOME', current_lang, key='initial', pron_e="Their" if profile.get('target')=="other" else "Your", pron_t="వారి" if profile.get('target')=="other" else "మీ")
            else:
                reply_text = get_prompt('AWAITING_OCCUPATION', current_lang, key='retry')

        elif current_step == 'AWAITING_INCOME':
            skip_kws = get_intent_keywords('AWAITING_INCOME', current_lang, key='skip')
            if processed_text:
                # did user say “don’t know”?
                if any(w == processed_text.lower() for w in skip_kws):
                    profile['income'] = None
                else:
                    try:
                        profile['income'] = int(processed_text)
                    except ValueError:
                        reply_text = get_prompt('AWAITING_INCOME', current_lang, 'retry')
                        save_user_state(sender_id, user_state)
                        resp = MessagingResponse(); resp.message(reply_text)
                        return Response(str(resp), mimetype='text/xml')
                user_state['step'] = 'AWAITING_CASTE'
                # prompt next
                pron_e = "Their" if profile.get('target')=='other' else "Your"
                pron_t = "వారి"  if profile.get('target')=='other' else "మీ"
                reply_text = get_prompt('AWAITING_CASTE', current_lang, 'initial', pron_e=pron_e, pron_t=pron_t)
            else:
                reply_text = get_prompt('AWAITING_INCOME', current_lang, 'retry')

        elif current_step == 'END':
            app.logger.info("Ending conversation.")
            reply_text = "ధన్యవాదాలు!" if current_lang == 'te' else "Thank you!"
            save_user_state(sender_id, {'step': 'START', 'profile': {}, 'language': current_lang}); # Reset keeping lang
            app.logger.info(f"Req {request_id}: Updated state: {load_user_state(sender_id)}") # Log full state dict
            app.logger.info(f"Req {request_id}: End state: {load_user_state(sender_id)}")

        else: # Fallback Reset
             app.logger.warning(f"Unhandled step '{current_step}'. Resetting.")
             q_target_te="క్షమించండి, మొదలు నుండి ప్రారంభిద్దాం. మీకోసమా లేక ఇతరుల కోసమా?"; q_target_en="Sorry, let's start over. Yourself or someone else?";
             reply_text = q_target_te if current_lang == 'te' else q_target_en;
             save_user_state(sender_id, {'step': 'AWAITING_TARGET', 'profile': {}, 'language': current_lang}); # Reset keeping lang

        # Update state ONLY if we didn't return early (like multi-message greeting)
        if reply_text is not None:
             save_user_state(sender_id, user_state)
             app.logger.info(f"Req {request_id}: Updated state: {load_user_state(sender_id)}") # Log full state dict
             app.logger.info(f"Req {request_id}: End state: {load_user_state(sender_id)}")

    except Exception as e: # Catch errors during State Machine Logic / Parsing
        app.logger.error(f"Req {request_id}: ERROR processing State/Message: {e}", exc_info=True)
        reply_text = "క్షమించండి, ప్రాసెస్ చేయడంలో లోపం." if current_lang == 'te' else "Sorry, a processing error occurred."
        # Reset state on error? Maybe safer.
        save_user_state(sender_id, {'step': 'START', 'profile': {}, 'language': current_lang})

    # Ensure reply_text is set if state machine didn't return early
    if reply_text is None and 'response_twiml_multi' not in locals():
        reply_text = "Error: No reply generated."
        app.logger.error(f"Reached end of webhook without setting reply_text or returning response for state {current_step}")

    # --- 5. Send Final Reply (Only if reply_text is set) ---
    response_twiml = MessagingResponse()
    if reply_text: # Check if a reply needs to be sent
        response_twiml.message(reply_text)
        RED = "\033[91m"
        BOLD = "\033[1m"
        RESET = "\033[0m"
        FRAME = "=" * 30
        app.logger.info(f"{RED}{BOLD}\n{FRAME}\nTwiML to user: {reply_text}\n{FRAME}{RESET}")
    else:
        # This handles cases where TwiML was built and returned directly (multi-message)
        # or where an error happened very early. Sending empty TwiML is okay.
        app.logger.info(f"Req {request_id}: No explicit reply_text set, sending empty TwiML response.")

    # --- THIS RETURN IS CORRECTLY INDENTED ---
    RED = "\033[91m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
    FRAME = "=" * 30
    app.logger.info(f"{RED}{BOLD}\n{FRAME}\nTwiML response: {str(response_twiml)}\n{FRAME}{RESET}")
    return Response(str(response_twiml), mimetype='text/xml')

# --- Helper: Process Voice Message ---
def process_voice_message(media_url, request_id):
    """Downloads, converts, and transcribes voice message. Returns text or None."""
    download_path = None
    converted_path = None
    transcribed_text = None # Default return value on failure

    try:
        # 1. Check Credentials
        if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
            raise ValueError("Missing Twilio credentials (ACCOUNT_SID or AUTH_TOKEN) for media download")

        # 2. Download Audio
        app.logger.info(f"Req {request_id}: Downloading audio from {media_url}")
        audio_response = requests.get(media_url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
        audio_response.raise_for_status()
        app.logger.info(f"Req {request_id}: Download status: {audio_response.status_code}")

        # 3. Save Temp File
        temp_id = uuid.uuid4()
        download_path = os.path.join(tempfile.gettempdir(), f"{temp_id}_dl.audio")
        with open(download_path, 'wb') as f: f.write(audio_response.content)
        app.logger.info(f"Req {request_id}: Audio saved to {download_path}")

        # 4. Convert Audio -> WAV
        converted_path = os.path.join(tempfile.gettempdir(), f"{temp_id}_cnv.wav")
        app.logger.info(f"Req {request_id}: Converting {download_path} -> {converted_path}...")
        from pydub import AudioSegment
        audio_segment = AudioSegment.from_file(download_path)
        audio_segment.export(converted_path, format="wav")
        app.logger.info(f"Req {request_id}: Conversion OK.")

        # 5. Transcribe
        app.logger.info(f"Req {request_id}: Transcribing {converted_path}...")
        if WHISPER_MODEL:
            transcription_result = WHISPER_MODEL(converted_path)
            if transcription_result and isinstance(transcription_result, dict) and "text" in transcription_result:
                transcribed_text = transcription_result["text"].strip()
            elif transcription_result and isinstance(transcription_result, str):
                transcribed_text = transcription_result.strip()
            else: transcribed_text = None
            app.logger.info(f"Req {request_id}: Transcription result: '{transcribed_text}'")
        else:
            app.logger.error(f"Req {request_id}: Whisper model not loaded!"); transcribed_text = None;

    except requests.exceptions.RequestException as e: app.logger.error(f"Req {request_id}: Download error: {e}", exc_info=False) # Less verbose log
    except ValueError as e: app.logger.error(f"Req {request_id}: Value Error (Creds?): {e}", exc_info=False)
    except FileNotFoundError as e: app.logger.error(f"Req {request_id}: Convert Error (ffmpeg missing?): {e}", exc_info=False)
    except Exception as e: app.logger.error(f"Req {request_id}: Voice Processing Error: {e}", exc_info=True) # Keep traceback for others
    finally:
        # Clean up temporary files regardless of success/failure
        app.logger.debug(f"Req {request_id}: Entering finally block for cleanup.") # Add debug log
        if download_path and os.path.exists(download_path):
            app.logger.debug(f"Attempting to remove download path: {download_path}")
            try:
                os.remove(download_path)
                app.logger.info(f"Req {request_id}: Cleaned up {download_path}")
            except Exception as e:
                app.logger.warning(f"Could not delete temp dl file {download_path}: {e}")
        elif download_path:
            app.logger.debug(f"Download path {download_path} exists check returned False.")

        if converted_path and os.path.exists(converted_path):
            app.logger.debug(f"Attempting to remove converted path: {converted_path}")
            try:
                os.remove(converted_path)
                app.logger.info(f"Req {request_id}: Cleaned up {converted_path}")
            except Exception as e:
                app.logger.warning(f"Could not delete temp cnv file {converted_path}: {e}")
        elif converted_path:
            app.logger.debug(f"Converted path {converted_path} exists check returned False.")

    return transcribed_text

def setup_database():
    DB_FILE = os.environ.get('DB_FILE', 'welfare_schemes.db')
    conn = sqlite3.connect(DB_FILE)
    try:
        print("Populating schemes table from schemes_updated.csv")
        pd.read_csv('schemes_updated.csv').to_sql('schemes', conn, if_exists='replace', index=False)
        print("Populating eligibility_rules table from eligibility_rules_updated.csv")
        pd.read_csv('eligibility_rules_updated.csv').to_sql('eligibility_rules', conn, if_exists='replace', index=False)
        if os.path.exists('scheme_categories_updated.csv'):
            print("Populating scheme_categories table from scheme_categories_updated.csv")
            pd.read_csv('scheme_categories_updated.csv').to_sql('scheme_categories', conn, if_exists='replace', index=False)
        if os.path.exists('documents_required.csv'):
            print("Populating documents_required table from documents_required.csv")
            pd.read_csv('documents_required.csv').to_sql('documents_required', conn, if_exists='replace', index=False)
        print("✔︎ Database setup finished")
    finally:
        conn.close()

# --- Entry Point ---
if __name__ == "__main__":
    init_state_db()
    setup_database()
    if WHISPER_MODEL is None: app.logger.error("FATAL: Whisper model not loaded."); exit();
    app.logger.info("Starting Flask App...")
    app.run(debug=True, host='0.0.0.0', port=5001) # Use the working port

# --- END OF COMPLETE telugu_protype_sql.py ---