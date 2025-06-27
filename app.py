# --- START OF COMPLETE, FINAL app.py ---

import os
import tempfile
import logging
import requests
import json
import threading
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
from twilio.request_validator import RequestValidator
from dotenv import load_dotenv
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# --- Load environment variables first ---
load_dotenv()

# --- Import centralized logging configuration ---
from logging_config import get_logger

# --- Import your own modules ---
from nlu import extract_profile_from_text
from utils import profile_summary_te, profile_summary_en, match_schemes, get_scheme_details_by_name, get_required_documents
# --- Import new session management ---
from session_manager import get_user_state, save_user_state, clear_user_state

# --- Create Flask app ---
app = Flask(__name__)

# --- Get logger for this module ---
logger = get_logger(__name__)

# --- RATE LIMITING CONFIGURATION ---
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

# --- RATE LIMIT ERROR HANDLER ---
@app.errorhandler(429)  # Too Many Requests
def ratelimit_handler(e):
    """Handle rate limit exceeded errors with a proper TwiML response."""
    logger.warning(f"Rate limit exceeded for IP: {get_remote_address()}")
    response = MessagingResponse()
    response.message(
        "Rate limit exceeded. Please wait a moment before sending another message. "
        "You can send up to 20 messages per minute."
    )
    return str(response), 429

# --- ROBUST LOGGING AND CLIENT SETUP ---
logger.info("Flask app logger configured.")
logger.info("Rate limiting configured with 20 requests per minute per IP.")

TWILIO_ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.environ.get('TWILIO_AUTH_TOKEN')

# --- Twilio Template SIDs ---
TWILIO_TEMPLATE_SID_EN_CONFIRM = os.environ.get('TWILIO_TEMPLATE_SID_EN_CONFIRM')
TWILIO_TEMPLATE_SID_TE_CONFIRM = os.environ.get('TWILIO_TEMPLATE_SID_TE_CONFIRM')
TWILIO_TEMPLATE_SID_TE_SCHEME_DETAILS = os.environ.get('TWILIO_TEMPLATE_SID_TE_SCHEME_DETAILS')
TWILIO_TEMPLATE_SID_EN_SCHEME_DETAILS = os.environ.get('TWILIO_TEMPLATE_SID_EN_SCHEME_DETAILS')

try:
    twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    # Initialize Twilio request validator for signature verification
    validator = RequestValidator(TWILIO_AUTH_TOKEN)
    logger.info("Twilio client and request validator initialized successfully.")
except Exception as e:
    logger.exception("Could not initialize Twilio client")
    twilio_client = None
    validator = None

# --- Voice Transcription ---
def transcribe_voice_message(media_url):
    WHISPER_REMOTE_URL = os.environ.get('WHISPER_REMOTE_URL')
    try:
        auth = (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        media_content = requests.get(media_url, auth=auth).content
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as temp_audio_file:
            temp_audio_file.write(media_content)
            temp_filename = temp_audio_file.name
        with open(temp_filename, 'rb') as f:
            files = {'file': (os.path.basename(temp_filename), f, 'audio/ogg')}
            response = requests.post(WHISPER_REMOTE_URL, files=files)
            response.raise_for_status()
            transcription = response.json().get("text", "").strip()
            logger.info(f"Whisper transcription successful: '{transcription}'")
            return transcription
    except Exception as e:
        logger.exception("An error occurred during transcription")
        return None
    finally:
        if 'temp_filename' in locals() and os.path.exists(temp_filename):
            os.remove(temp_filename)

# --- Background Worker Functions ---
def process_voice_message_in_background(sender_id, media_url, twilio_number):
    logger.info(f"BACKGROUND (VOICE): Starting processing for {sender_id}")
    transcribed_text = transcribe_voice_message(media_url)
    if not transcribed_text:
        logger.error("BACKGROUND (VOICE): Transcription failed.")
        return
    process_text_in_background(sender_id, transcribed_text, twilio_number)

def process_text_in_background(sender_id, text, twilio_number):
    logger.info(f"BACKGROUND (TEXT): Starting NLU extraction for {sender_id}")
    extracted_profile = extract_profile_from_text(text)
    if not extracted_profile:
        logger.error("BACKGROUND (TEXT): NLU extraction failed.")
        if twilio_client:
            twilio_client.messages.create(from_=twilio_number, to=sender_id, body="Sorry, I could not understand the details.")
        return
    
    user_state = get_user_state(sender_id)
    user_state['profile'] = extracted_profile
    lang = extracted_profile.get('language', 'telugu')
    user_state['lang'] = lang
    save_user_state(sender_id, user_state)
    
    # --- THIS IS THE CRITICAL FIX #1: BILINGUAL CONFIRMATION ---
    if lang == 'english':
        summary = profile_summary_en(extracted_profile)
        template_sid = TWILIO_TEMPLATE_SID_EN_CONFIRM
    else:
        summary = profile_summary_te(extracted_profile)
        template_sid = TWILIO_TEMPLATE_SID_TE_CONFIRM

    if twilio_client:
        try:
            twilio_client.messages.create(
                from_=twilio_number, to=sender_id,
                content_sid=template_sid,
                content_variables=json.dumps({'1': summary})
            )
            logger.info(f"Successfully sent confirmation template {template_sid} to user.")
        except Exception as e:
            logger.exception("Failed to send template message")
    logger.info(f"BACKGROUND: Processing complete for {sender_id}")

# ==============================================================================
# =================== THE COMPLETE WHATSAPP WEBHOOK LOGIC ======================
# ==============================================================================

@app.route('/whatsapp', methods=['POST'])
@limiter.limit("20 per minute")
def whatsapp_webhook():
    """
    WhatsApp webhook endpoint with Twilio signature validation for security.
    """
    # --- CRITICAL SECURITY: TWILIO SIGNATURE VALIDATION ---
    if validator is None:
        logger.error("Twilio validator not initialized. Rejecting request for security.")
        return "Forbidden", 403
    
    signature = request.headers.get('X-Twilio-Signature')
    if not signature:
        logger.warning("Missing X-Twilio-Signature header. Rejecting request.")
        return "Forbidden", 403
    try:
        # Flask's request.form is a MultiDict, which is what Twilio expects
        url = request.url
        post_vars = request.form  # Use request.form directly, not converted to dict
        is_valid = validator.validate(url, post_vars, signature)
        if not is_valid:
            logger.warning(f"Invalid Twilio signature from IP: {request.remote_addr}")
            return "Forbidden", 403
        logger.info(f"Valid Twilio signature verified from IP: {request.remote_addr}")
    except Exception as e:
        logger.exception("Error validating Twilio signature")
        return "Forbidden", 403

    # --- PROCESS THE VALIDATED REQUEST ---
    form_data = request.form
    sender_id = form_data.get('From')
    user_state = get_user_state(sender_id)
    is_voice_message = 'MediaUrl0' in form_data
    processed_text = form_data.get('Body', '').strip()

    # --- GLOBAL EXIT COMMAND ---
    if processed_text.lower() == 'exit':
        user_state.clear()
        user_state['step'] = 'START'
        save_user_state(sender_id, user_state)
        reply_text = "Your session has been reset. To start over, please provide your details."
        response = MessagingResponse()
        response.message(reply_text)
        return str(response)

    # --- MAIN STATE MACHINE ---
    current_step = user_state.get('step', 'START')
    lang = user_state.get('lang', 'en')
    reply_text = ""

    if current_step == 'START':
        user_input_lower = processed_text.lower()
        if user_input_lower == 'te':
            user_state['lang'] = 'te'
            reply_text = "నమస్కారం! దయచేసి మీ పూర్తి వివరాలు వాయిస్ నోట్ లేదా టెక్స్ట్ రూపంలో చెప్పండి."
            user_state['step'] = 'AWAITING_PROFILE_DATA'
        else:
            user_state['lang'] = 'en'
            reply_text = (
                "Welcome to the AP Welfare Schemes Assistant.\n\n"
                "Please tell me about yourself or the person you're asking for in a single message (voice or text).\n\n"
                "తెలుగు భాష కోసం TE అని టైప్ చేయండి."
            )
            if is_voice_message or (processed_text and user_input_lower != 'hi'):
                if is_voice_message:
                    thread = threading.Thread(target=process_voice_message_in_background, args=(sender_id, form_data['MediaUrl0'], form_data.get('To')))
                    thread.start()
                    reply_text = "Thank you. Processing your voice message..."
                else:
                    thread = threading.Thread(target=process_text_in_background, args=(sender_id, processed_text, form_data.get('To')))
                    thread.start()
                    reply_text = "Thank you. Processing your text message..."
            else:
                 user_state['step'] = 'AWAITING_PROFILE_DATA'
        save_user_state(sender_id, user_state)
        
    elif current_step == 'AWAITING_PROFILE_DATA':
        if is_voice_message:
            thread = threading.Thread(target=process_voice_message_in_background, args=(sender_id, form_data['MediaUrl0'], form_data.get('To')))
            thread.start()
            reply_text = "Thank you. Processing your voice message..."
        elif processed_text and processed_text.lower() not in ['te', 'en', 'hi']:
            thread = threading.Thread(target=process_text_in_background, args=(sender_id, processed_text, form_data.get('To')))
            thread.start()
            reply_text = "Thank you. Processing your text message..."
        else:
            reply_text = "Please provide your details in a voice note or as a text message."
            
    elif current_step == 'AWAITING_CONFIRMATION':
        if processed_text == 'అవును' or processed_text.lower() == 'yes':
            profile_data = user_state['profile']
            matched_schemes_df = match_schemes(profile_data)
            if not matched_schemes_df.empty:
                scheme_names = matched_schemes_df['SchemeName'].tolist()
                user_state['matched_schemes'] = scheme_names
                schemes_list = "\n".join([f"{i+1}. {name}" for i, name in enumerate(scheme_names)])
                reply_text = (f"Thank you! Based on your profile, you may be eligible for the following schemes:\n\n{schemes_list}\n\n"
                              "To get more details about a scheme, please type its number.")
                user_state['step'] = 'AWAITING_SCHEME_CHOICE'
            else:
                reply_text = "Thank you. Based on your profile, no matching schemes were found at this time."
                user_state['step'] = 'DONE'
            save_user_state(sender_id, user_state)
        elif processed_text == 'కాదు' or processed_text.lower() == 'no':
            reply_text = "I'm sorry about that. Please provide your complete details again to start over."
            user_state.clear()
            user_state['step'] = 'START'
            save_user_state(sender_id, user_state)
        else:
            reply_text = "Please use the provided buttons to select 'Yes' or 'No'."

    elif current_step == 'AWAITING_SCHEME_CHOICE':
        try:
            choice_index = int(processed_text) - 1
            matched_schemes = user_state.get('matched_schemes', [])
            if 0 <= choice_index < len(matched_schemes):
                chosen_scheme_name = matched_schemes[choice_index]
                user_state['selected_scheme_name'] = chosen_scheme_name
                scheme_details = get_scheme_details_by_name(chosen_scheme_name)
                
                if scheme_details:
                    documents = get_required_documents(scheme_details['scheme_id'])
                    doc_list_str = "\n".join([f"• {doc}" for doc in documents]) if documents else "Not specified."
                    
                    # --- THIS IS THE CRITICAL FIX #2: BILINGUAL SCHEME DETAILS ---
                    lang = user_state.get('lang', 'en')
                    
                    if lang == 'te':
                        template_sid = TWILIO_TEMPLATE_SID_TE_SCHEME_DETAILS
                    else:
                        template_sid = TWILIO_TEMPLATE_SID_EN_SCHEME_DETAILS
                    
                    if twilio_client:
                        twilio_client.messages.create(
                            from_=form_data.get('To'), to=sender_id,
                            content_sid=template_sid,
                            content_variables=json.dumps({
                                '1': scheme_details['name'],
                                '2': scheme_details.get('description', 'No description available.'),
                                '3': doc_list_str
                            })
                        )
                    reply_text = "" 
                    user_state['step'] = 'AWAITING_PROCEED_OR_REVIEW'
                    save_user_state(sender_id, user_state)
                else:
                    reply_text = f"Sorry, an error occurred fetching details for '{chosen_scheme_name}'."
            else:
                reply_text = "Please select a valid number from the list."
        except ValueError:
            reply_text = "Please type only the number of the scheme."

    elif current_step == 'AWAITING_PROCEED_OR_REVIEW':
        if processed_text.lower() == 'proceed' or processed_text == 'కొనసాగించు':
            chosen_scheme_name = user_state.get('selected_scheme_name', 'the selected scheme')
            scheme_details = get_scheme_details_by_name(chosen_scheme_name)
            reply_parts = [f"Great! Let's proceed with the *{chosen_scheme_name}* application.\n"]
            if scheme_details and scheme_details.get('application_procedure'):
                reply_parts.append("*Application Procedure:*")
                reply_parts.append(scheme_details['application_procedure'])
            else:
                reply_parts.append("No specific application procedure is available in the database.")
            reply_text = "\n".join(reply_parts)
            user_state['step'] = 'DONE'
            save_user_state(sender_id, user_state)
        elif processed_text.lower() == 'back to list' or processed_text == 'జాబితాకు తిరిగి వెళ్ళు':
            scheme_names = user_state.get('matched_schemes', [])
            schemes_list = "\n".join([f"{i+1}. {name}" for i, name in enumerate(scheme_names)])
            reply_text = (f"Here is the list of matched schemes again:\n\n{schemes_list}\n\n"
                          "Please type the number of the scheme you want to review.")
            user_state['step'] = 'AWAITING_SCHEME_CHOICE'
            save_user_state(sender_id, user_state)
        else:
            reply_text = "Please use the 'Proceed' or 'Back to list' buttons to continue."

    response = MessagingResponse()
    response.message(reply_text)
    return str(response)

if __name__ == '__main__':
    app.run(debug=True, port=5001)

# --- END OF COMPLETE, FINAL app.py ---