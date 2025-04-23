# WhatsApp Telugu Chatbot Backend

A prototype backend for a WhatsApp chatbot that supports Telugu and English, parses user responses (including numbers, gender, caste, etc.), and interacts with users via Twilio and Flask. Designed for modularity, robustness, and easy extension.

---

## Features
- **Voice and Text Input:** Supports both text and voice messages using Whisper.
- **Accurate Telugu Number Parsing:** Handles complex Telugu numerals and digit fallback.
- **Robust Input Filtering:** Accepts valid short answers (e.g., "మగ", "ఆడ", "yes", "no", caste codes) and filters out garbage/irrelevant input.
- **State Machine Dialogue:** Guides users through age, gender, occupation, income, and more.
- **Centralized Utilities:** All parsing and extraction helpers live in `utils.py` for easy maintenance.
- **Comprehensive Logging:** Key events and errors are logged for debugging.

---

## Setup
1. **Clone the Repo:**
   ```bash
   git clone <your-repo-url>
   cd whatsapp_chatbot_backend
   ```
2. **Create & Activate a Virtual Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install Requirements:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Configure Environment:**
   - Copy `.env.example` to `.env` and fill in your Twilio and DB credentials:
     - `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `DB_FILE`, etc.

---

## Running the App
```bash
python app.py
```
- The Flask server will start. Use ngrok or similar for public webhook access.

---

## Project Structure
```
whatsapp_chatbot_backend/
├── app.py                  # Main Flask app and webhook
├── utils.py                # Parsing, extraction, and helper functions
├── telugu1_protype_sql.py  # Prototype/experimental logic
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (not committed)
└── README.md               # This file
```

---

## Key Design Principles
- **Modular:** Helpers are in `utils.py` and reusable.
- **Type-Safe:** Type hints and docstrings throughout.
- **Robust:** Handles edge cases and logs errors clearly.
- **Extensible:** Easy to add new states, languages, or parsing logic.

---

## Contributing
- PRs and issues welcome! Please add tests for new features.

---

## License
MIT License

---

## Credits
- Built with [Flask](https://flask.palletsprojects.com/), [Twilio](https://www.twilio.com/), [transformers](https://huggingface.co/docs/transformers/index), and [torch](https://pytorch.org/).
- Telugu NLP logic inspired by open-source projects and community contributions.
