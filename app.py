# --- START OF COMPLETE app.py ---

import os
import sqlite3
import tempfile
import traceback
import unicodedata # Needed if utils doesn't import it
import uuid
import time
import warnings
import logging
import requests
from transformers import pipeline # Keep if whisper loading is here
import speech_recognition as sr # Keep if you plan to use mic directly elsewhere
import torch
from flask import Flask, request, Response, abort, make_response
from twilio.twiml.messaging_response import MessagingResponse
from twilio.request_validator import RequestValidator
from werkzeug.middleware.proxy_fix import ProxyFix # For ngrok https fix
# --- At the TOP of app.py ---
import os
from dotenv import load_dotenv # Import load_dotenv

# --- Add Dotenv Debugging ---
dotenv_path = os.path.join(os.path.dirname(__file__), '.env') # Explicitly find .env path
load_success = load_dotenv(dotenv_path=dotenv_path, verbose=True) # Try loading and be verbose
# Print relevant vars *after* load_dotenv
app = Flask(__name__)
app.logger.info(f".env file load successful: {load_success}")
app.logger.info(f"Value of TWILIO_ACCOUNT_SID after load_dotenv: {os.environ.get('TWILIO_ACCOUNT_SID')}")
app.logger.info(f"Value of TWILIO_AUTH_TOKEN after load_dotenv: {os.environ.get('TWILIO_AUTH_TOKEN')}")
# --- End Dotenv Debugging ---

# --- Now get the variables (as before) ---
TWILIO_AUTH_TOKEN = os.environ.get('TWILIO_AUTH_TOKEN')
TWILIO_ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID')

# --- Import functions from your utility file ---
try:
    from utils import (
        telugu_to_number,
        extract_gender,
        extract_occupation,
        check_is_student,
        extract_yes_no,
        extract_caste,
        extract_income_range,
        extract_scheme_names,
        extract_number,
        get_db_connection
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
DB_FILE = "welfare_schemes.db"

# Define garbage lists and minimum length globally
GARBAGE_PHRASES_TE = ["సరే సరే", "ఓకే ఓకే", "థాంక్యూ", "ధన్యవాదాలు", "హమ్ హమ్", "హ్మ్ హ్మ్", "ఉహు ఉహు", "నమస్కారం", "హలో", "హాయ్", "ఓయ్", "ఏయ్", "ఏరా", "ఏంటి", "హు హు", "హూ హూ", "హు", "ఉమ్ ఉమ్"]
GARBAGE_PHRASES_EN = ["okay okay", "thank you", "thanks", "hello", "hi", "hey", "what", "alright", "right", "great", "fine", "well", "so", "hum hum"]
SHORT_SOUNDS_TE = ["ఆహ్", "ఉమ్", "సరే", "ఆహా", "ఉహు", "ఆం", "ఆమ్", "హుఁ", "హూఁ", "ఊఁ", "అయ్యో", "ఓ", "ఆఁ"]
SHORT_SOUNDS_EN = ["uh", "um", "hmm", "ok", "aha", "uh huh", "uh uh", "oh", "ah", "mm hmm", "mmm", "huh"]
MIN_MEANINGFUL_LENGTH = 2

SKIP_KEYWORDS_TE = ["తెలియదు", "నాకు తెలియదు", "వద్దు", "చెప్పను", "స్కిప్", "తరువాత", "తర్వాత"]
SKIP_KEYWORDS_EN = ["i don't know", "skip", "no", "pass", "none", "later"]

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

# --- Global State Dictionary (Simple In-Memory) ---
conversation_state = {}

# --- Flask App Initialization & Proxy Fix ---
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1) # Handle proxy headers (like ngrok's https)

from typing import Optional

# --- Helper: Text Filtering (Refactored & Type-Annotated) ---
def filter_text(raw_text: str, lang: str = "te") -> Optional[str]:
    """Filters garbage, short, or empty text, allowing essential short answers."""
    if not raw_text:
        logging.info("[Filter] Empty input.")
        return None

    check = raw_text.lower().strip()
    if not check:
        logging.info("[Filter] Empty after clean.")
        return None

    essential_short_te = ["ఓసి", "బిసి", "ఎస్సీ", "ఎస్టీ", "మగ", "ఆడ", "అవును", "కాదు", "లేదు", "ఉంది"]
    essential_short_en = ["hi", "hello", "oc", "bc", "sc", "st", "yes", "no", "male", "female"]
    essentials = essential_short_te if lang == 'te' else essential_short_en

    g_list = GARBAGE_PHRASES_TE if lang == 'te' else GARBAGE_PHRASES_EN
    s_list = SHORT_SOUNDS_TE if lang == 'te' else SHORT_SOUNDS_EN

    if check in essentials:
        logging.info(f"[Filter] Kept essential short answer: '{raw_text}'")
        return raw_text

    if check in g_list:
        logging.info(f"[Filter] Filtered exact garbage: '{raw_text}'")
        return None

    if check in s_list:
        logging.info(f"[Filter] Filtered short sound: '{raw_text}'")
        return None

    if len(raw_text) < MIN_MEANINGFUL_LENGTH:
        logging.info(f"[Filter] Filtered too short (and not essential): '{raw_text}'")
        return None

    logging.info(f"[Filter] Input passed filtering: '{raw_text}'")
    return raw_text
# --- End Refactored Filter ---

# --- Main Webhook Route ---
@app.route('/whatsapp', methods=['POST'])
def whatsapp_webhook():
    request_id = str(uuid.uuid4())[:8]; app.logger.info(f"--- Request {request_id}: /whatsapp HIT ---")

    # 1. --- Request Validation ---
    if validator:
        url=request.url; post_vars=request.form.to_dict(); sig=request.headers.get('X-Twilio-Signature')
        is_valid = validator.validate(url, post_vars, sig)
        if not is_valid:
            app.logger.error(f"Req {request_id}: !!! Validation FAILED - 403 !!! Check Token/URL.");
            return make_response("Validation failed", 403) # Early return on failure
        else: app.logger.info(f"Req {request_id}: Validation OK.")
    else: app.logger.warning(f"Req {request_id}: Skipping validation.")

    # 2. --- Initialize Reply & State ---
    sender_id = request.values.get('From', None)
    processed_text = None
    # Default reply - set this specifically in except block if needed
    reply_text = None # Initialize reply_text to None to clearly track if it was set
    user_state = None # Initialize user_state

    if not sender_id:
        app.logger.error(f"Req {request_id}: Missing 'From' field. Cannot process.")
        # Send an empty response or generic error? Empty is usually safe for Twilio.
        return Response(str(MessagingResponse()), mimetype='text/xml')

    app.logger.info(f"Req {request_id}: From: {sender_id}")
    user_state = conversation_state.get(sender_id, {'step': 'START', 'profile': {}, 'lang': 'te'}) # Get or init state
    current_lang = user_state.get('lang', 'te')
    app.logger.info(f"Req {request_id}: Initial state: {user_state}")


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
            user_state['lang'] = lang_detected; current_lang = lang_detected; # Update state & current lang
            app.logger.info(f"Req {request_id}: Language state updated to: {current_lang}")
            # Filter text AFTER determining language
            processed_text = filter_text(processed_text, current_lang)
            app.logger.info(f"Req {request_id}: Text after filtering: '{processed_text}'")
        # else: Keep previous reply_text (often None or error from voice processing)

        # --- Core State Machine ---
        current_step = user_state.get('step', 'START')
        user_profile = user_state.get('profile', {})
        app.logger.info(f"Req {request_id}: Executing state='{current_step}', Input='{processed_text}'")

        if current_step == 'START':
            activation_check_text = processed_text.lower().strip() if processed_text else ""

            # *** Use consistent case and check against correct keywords ***
            is_voice_activation = (activation_check_text == "babu")
            is_voice_activation_te = (activation_check_text == "బాబు")
            is_text_activation = (activation_check_text in ["hi", "hello"])

            if is_voice_activation or is_text_activation or is_voice_activation_te:
                output_lang = 'te' # Default to Telugu output
                input_lang = current_lang # Use detected current_lang for input
                if is_text_activation and input_lang == 'en':
                    output_lang = 'en'
                user_state['lang'] = output_lang
                current_lang = output_lang

                gt = "నమస్కారం! AP సంక్షేమ పథకాల సహాయకుడికి స్వాగతం."; ge = "Hello! Welcome."
                q_st = "మీకోసమా లేక ఇతరుల కోసమా?"; q_se = "For yourself or someone else?"
                resp = MessagingResponse(); resp.message(gt if current_lang == 'te' else ge); resp.message(q_st if current_lang == 'te' else q_se)
                user_state['step'] = 'AWAITING_TARGET'; reply_text = None

                conversation_state[sender_id] = user_state; app.logger.info(f"Req {request_id}: Updated state: {user_state}")
                app.logger.info(f"Req {request_id}: Sending multi-message reply...")
                return Response(str(resp), mimetype='text/xml')
            else:
                app.logger.info(f"No activation detected in '{processed_text}'. Sending prompt.")
                p_t="ప్రారంభించడానికి 'Babu' అని వాయిస్ నోట్ పంపండి లేదా 'Hi' అని టైప్ చేయండి."; p_e="Say 'Babu' (voice) or type 'Hi'."
                reply_text = p_t if current_lang == 'te' else p_e
                # Step remains START

        elif current_step == 'AWAITING_TARGET':
             # ...(Paste your validated logic for AWAITING_TARGET here)...
             # Ensure this block sets reply_text to the *next* question
             # and updates user_state['step']
             # Example structure:
             if processed_text:
                 answer_lower = processed_text.lower()
                 if any(w in answer_lower for w in ["other", "others", "ఇతరులు"]):
                     # Ask Relation
                     user_profile['target']='other'; user_state['step']='AWAITING_RELATION';
                     q_rt="వారికి మీరేమవుతారు?"; q_re="Their relation?"; reply_text=q_rt if current_lang=='te' else q_re;
                 else: # Assume self
                     # Ask Age
                     user_profile['target']='self'; user_state['step']='AWAITING_AGE';
                     q_at="మీ వయస్సు?"; q_ae="Your age?"; reply_text=q_at if current_lang=='te' else q_ae;
             else: # Re-ask if no text
                q_t="మీకోసమా లేక ఇతరుల కోసమా?"; q_e="Yourself or someone else?"; reply_text=q_t if current_lang=='te' else q_e;

        elif current_step == 'AWAITING_RELATION':
             # ...(Paste your validated logic for AWAITING_RELATION here)...
             if processed_text:
                 user_profile['relation']=processed_text; user_state['step']='AWAITING_AGE';
                 q_at="వారి వయస్సు?"; q_ae="Their age?"; reply_text=q_at if current_lang=='te' else q_ae;
             else: # Retry
                 q_rt="వారికి మీరేమవుతారు?"; q_re="What is their relation?"; reply_text=q_rt if current_lang=='te' else q_re;

        elif current_step == 'AWAITING_AGE':
            app.logger.info(f"Processing Age input: '{processed_text}'")
            parsed_age = None
            skip_kw = SKIP_KEYWORDS_TE if current_lang == 'te' else SKIP_KEYWORDS_EN
            text_lower_for_skip = processed_text.lower() if processed_text else ""  # Handle None safely
            is_skip = bool(processed_text and any(s in text_lower_for_skip for s in skip_kw))

            if is_skip:
                app.logger.info("User skipped age.")
                user_profile['age'] = None
            elif processed_text:
                try:
                    parsed_age = telugu_to_number(processed_text) # Use corrected V8 parser
                    app.logger.info(f"Parsed age value: {parsed_age} (Type: {type(parsed_age)})")
                    # Validation is key HERE, after parsing
                    if parsed_age is not None and isinstance(parsed_age, int) and 1 <= parsed_age < 120:
                        # --- VALID AGE ---
                        user_profile['age'] = parsed_age
                        # Determine next step based on age
                        if parsed_age <= 21:
                            app.logger.info("Age <= 21. Moving to AWAITING_STUDENT_STATUS.")
                            user_state['step'] = 'AWAITING_STUDENT_STATUS'
                            qt="మీరు విద్యార్థినా? (అవును/కాదు)"; qe="Are you a student? (Yes/No)";
                            reply_text=qt if current_lang=='te' else qe;
                        else:
                            app.logger.info("Age > 21. Moving to AWAITING_GENDER.")
                            user_state['step'] = 'AWAITING_GENDER'
                            pron_t="మీ" if user_profile.get('target')=='self' else "వారి"; pron_e="Your" if user_profile.get('target')=='self' else "Their";
                            qt = f"{pron_t} లింగం? (మగ/ఆడ)"; qe = f"{pron_e} gender? (Male/Female)";
                            reply_text=qt if current_lang=='te' else qe;
                    else:
                        # --- INVALID/FAILED PARSE ---
                        app.logger.warning(f"Invalid age range or parse failure: {parsed_age}")
                        user_profile['age'] = None # Ensure age is None
                        qt = "వయస్సు సరైనది కాదు/అర్థం కాలేదు. 1-120 మధ్య సంఖ్య చెప్పండి?";
                        qe = "Invalid age / parse error. Please state age (1-120).";
                        reply_text = qt if current_lang == 'te' else qe;
                        user_state['step'] = 'AWAITING_AGE'; # Stay to re-ask
                except Exception as e: # Catch unexpected errors during/after parsing
                    app.logger.error(f"Error processing age: {e}", exc_info=True)
                    user_profile['age'] = None; reply_text = "Error processing age."
                    user_state['step'] = 'AWAITING_AGE'; # Re-ask on error too
            else:
                # No valid text received (empty or filtered out)
                qt = "వయస్సు చెప్పండి?"; qe = "Please state the age."; reply_text=qt if current_lang=='te' else qe;
                user_state['step'] = 'AWAITING_AGE'; # Stay to re-ask

            # Move to Gender if age was skipped
            if is_skip:
                user_state['step'] = 'AWAITING_GENDER'
                pron_t="మీ" if user_profile.get('target')=='self' else "వారి"; pron_e="Your" if user_profile.get('target')=='self' else "Their";
                qt=f"{pron_t} లింగం?"; qe=f"{pron_e} gender?"; reply_text=qt if current_lang=='te' else qe;

        elif current_step == 'AWAITING_STUDENT_STATUS':
            app.logger.info(f"Processing Student Status input: '{processed_text}'")
            target=user_profile.get('target','self'); pron_t="వారి" if target=='other' else "మీ"; pron_e="Their" if target=='other' else "Your";
            if processed_text:
                is_student = extract_yes_no(processed_text, current_lang) # Use your helper
                user_profile['is_student'] = is_student
                app.logger.info(f"Stored is_student: {is_student}")
            else:
                user_profile['is_student'] = None # Mark unknown if no clear answer/skip
            # Always proceed to Gender next
            user_state['step'] = 'AWAITING_GENDER'
            qt = f"{pron_t} లింగం? (మగ/ఆడ)"; qe = f"{pron_e} gender? (Male/Female)";
            reply_text = qt if current_lang=='te' else qe;

        elif current_step == 'AWAITING_GENDER':
            app.logger.info(f"Processing Gender input: '{processed_text}'")
            # 1. Parse Gender
            if processed_text:
                user_profile['gender'] = extract_gender(processed_text, current_lang) # Use helper from utils
                app.logger.info(f"Parsed gender: {user_profile['gender']}")
            else:
                user_profile['gender'] = None

            # 2. Determine next question: Occupation
            target=user_profile.get('target','self'); pron_t="వారి" if target=='other' else "మీ"; pron_e="Their" if target=='other' else "Your";
            q_ot = f"{pron_t} వృత్తి? (ఉదా: రైతు)"; q_oe = f"{pron_e} occupation? (Ex: farmer)"
            reply_text = q_ot if current_lang=='te' else q_oe

            # 3. Update state
            user_state['step'] = 'AWAITING_OCCUPATION'

        # --- ADD LOGIC FOR OTHER STATES ---
        # elif current_step == 'AWAITING_GENDER': etc...

        else: # Fallback Reset
             app.logger.warning(f"Unhandled step '{current_step}'. Resetting.")
             q_target_te="క్షమించండి, మొదలు నుండి ప్రారంభిద్దాం. మీకోసమా లేక ఇతరుల కోసమా?"; q_target_en="Sorry, let's start over. Yourself or someone else?";
             reply_text = q_target_te if current_lang == 'te' else q_target_en;
             user_state = {'step': 'AWAITING_TARGET', 'profile': {}, 'lang': current_lang}; # Reset keeping lang

        # Update state ONLY if we didn't return early (like multi-message greeting)
        if reply_text is not None:
             conversation_state[sender_id] = user_state
             app.logger.info(f"Req {request_id}: End state: {user_state}")

    except Exception as e: # Catch errors during State Machine Logic / Parsing
        app.logger.error(f"Req {request_id}: ERROR processing State/Message: {e}", exc_info=True)
        reply_text = "క్షమించండి, ప్రాసెస్ చేయడంలో లోపం." if current_lang == 'te' else "Sorry, a processing error occurred."
        # Reset state on error? Maybe safer.
        conversation_state[sender_id] = {'step': 'START', 'profile': {}, 'lang': current_lang}

    # Ensure reply_text is set if state machine didn't return early
    if reply_text is None and 'response_twiml_multi' not in locals():
        reply_text = "Error: No reply generated."
        app.logger.error(f"Reached end of webhook without setting reply_text or returning response for state {current_step}")

    # --- 5. Send Final Reply (Only if reply_text is set) ---
    response_twiml = MessagingResponse()
    if reply_text: # Check if a reply needs to be sent
        response_twiml.message(reply_text)
        app.logger.info(f"Req {request_id}: Sending TwiML reply: '{reply_text}'")
    else:
        # This handles cases where TwiML was built and returned directly (multi-message)
        # or where an error happened very early. Sending empty TwiML is okay.
        app.logger.info(f"Req {request_id}: No explicit reply_text set, sending empty TwiML response.")

    # --- THIS RETURN IS CORRECTLY INDENTED ---
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

# --- Entry Point ---
if __name__ == '__main__':
    if WHISPER_MODEL is None: app.logger.error("FATAL: Whisper model not loaded."); exit();
    app.logger.info("Starting Flask App...")
    app.run(debug=True, host='0.0.0.0', port=5001) # Use the working port

# --- END OF COMPLETE telugu_protype_sql.py ---