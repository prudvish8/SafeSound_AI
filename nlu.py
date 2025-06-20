# --- START OF nlu.py ---
import os
import google.generativeai as genai
import json
import logging

# --- What this part does ---
# It loads your secret API key from the .env file and configures the library.
# The try/except block is good practice. If the key is missing, it logs an
# error instead of crashing the whole app.
try:
    # This line assumes you have python-dotenv installed and loaded in your main app
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")
    genai.configure(api_key=GOOGLE_API_KEY)
    logging.info("Google Generative AI configured successfully.")
except Exception as e:
    logging.error(f"Failed to configure Google Generative AI: {e}")

# --- The Core NLU Function ---
def extract_profile_from_text(text: str) -> dict | None:
    """
    Sends transcribed text to a Google LLM (Gemini) to extract profile details.
    Returns a dictionary on success, None on failure.
    """
    # We choose the model. 'gemini-1.5-flash' is perfect: fast, cheap, and powerful.
    model = genai.GenerativeModel('gemini-1.5-flash')

    # --- This is the most important part: The Prompt ---
    # A prompt is a set of instructions for the AI. We are telling it exactly
    # what to do, what format to use, and what rules to follow.
    prompt = f"""
    You are an expert data extraction assistant for an Andhra Pradesh government scheme chatbot.
    Your task is to analyze the following Telugu or English text and extract user profile information into a valid JSON object.

    The possible JSON keys are:
    - "target": (string) Should be "self" or "other".
    - "relation": (string) If target is "other", what is the relation (e.g., "father", "mother", "friend").
    - "age": (integer) The age in years.
    - "gender": (string) Should be "male" or "female".
    - "occupation": (string) The user's occupation. Normalize common terms (e.g., "రైతు", "వ్యవసాయం" -> "farmer"; "నిరుద్యోగి" -> "unemployed").
    - "income": (integer) The user's MONTHLY income as a number.
    - "ownership": (boolean) Should be true if they own a house, false if they do not.
    - "caste": (string) The user's caste or community (e.g.,"OC", "EWS", "BC", "SC", "ST").
    - "religion": (string) The user's religion (e.g., "Hindu", "Muslim", "Christian", "Sikh", "Jain", "Other").
    - "language": (string) The primary language of the user's text ("english" or "telugu").


    Rules:
    1.  If a piece of information is not mentioned in the text, DO NOT include its key in the JSON.
    2.  The output MUST be only the JSON object and nothing else. No introductory text.
    3.  Analyze the entire text to infer the details. For example, "నా కోసం" means target is "self". "మా నాన్న కోసం" means relation is "father" and target is "other".

    Here is the text to analyze:
    ---
    {text}
    ---

    JSON Output:
    """

    try:
        logging.info(f"Sending text to Gemini for NLU extraction: '{text[:100]}...'")
        # This is the line that actually calls the Google API
        response = model.generate_content(prompt)

        # LLMs sometimes add ```json ``` around their output. This cleaning step removes it.
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()
        logging.info(f"Gemini raw response: {cleaned_response}")

        # This converts the JSON text string into a Python dictionary we can use
        extracted_data = json.loads(cleaned_response)
        logging.info(f"Successfully extracted profile data: {extracted_data}")
        return extracted_data

    except Exception as e:
        # If the API fails or returns bad data, we log the error and return None.
        # This tells our main app that the extraction failed.
        logging.error(f"Error during Gemini API call or JSON parsing: {e}")
        return None
# --- In nlu.py, add this new function ---

def get_confirmation_intent(text: str) -> str:
    """
    Uses Gemini to classify the user's confirmation intent.
    Returns 'CONFIRMED', 'REJECTED', or 'UNCLEAR'.
    """
    logging.info(f"get_confirmation_intent: Received text: '{text}'")
    model = genai.GenerativeModel('gemini-1.5-flash')

    prompt = f"""
    You are a classification assistant for a chatbot. The user was just shown a summary of their profile and was asked "Is this correct?".
    Analyze the user's response and classify it into one of three categories:
    - "CONFIRMED": If the user agrees, says yes, or indicates the information is correct.
    - "REJECTED": If the user disagrees, says no, or indicates the information is wrong.
    - "UNCLEAR": If the user asks a question, changes the topic, or gives an ambiguous answer.

    The user's response is:
    ---
    {text}
    ---

    Category:
    """
    try:
        response = model.generate_content(prompt)
        logging.info(f"Gemini raw intent response: '{response.text.strip()}'")
        # Clean the response to get only the category word
        intent = response.text.strip().upper()
        if intent in ['CONFIRMED', 'REJECTED', 'UNCLEAR']:
            logging.info(f"NLU classified confirmation intent as: {intent}")
            return intent
        else:
            logging.warning(f"NLU returned unexpected intent: {intent}")
            return 'UNCLEAR'
    except Exception as e:
        logging.error(f"Error during intent classification: {e}")
        return 'UNCLEAR'


# --- END OF nlu.py ---