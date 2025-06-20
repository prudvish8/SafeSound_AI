# settings.py  ─ single source of config & constants
import os
from dotenv import load_dotenv, find_dotenv

# Load local .env (if present)
load_dotenv(find_dotenv())

# ───────────── Sensitive values (env-vars) ─────────────
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN  = os.getenv("TWILIO_AUTH_TOKEN")

# ───────────── Non-sensitive defaults ─────────────
DB_FILE              = os.getenv("DB_FILE", "welfare_schemes.db")
DEFAULT_LANGUAGE     = "te"
SUPPORTED_LANGUAGES  = ["te", "en"]

WELCOME_MESSAGE      = "Welcome to the AP Welfare Schemes Assistant!"
HELP_MESSAGE         = "You can ask about available welfare schemes or your eligibility."
FALLBACK_MESSAGE     = "I didn't quite catch that. Could you please repeat."

SESSION_TIMEOUT_MINUTES = 10
MAX_AUDIO_DURATION_SEC  = 15
