# 🛡️ Safesound LLC – Welfare Schemes AI Assistant

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-green)]()

A **voice-first WhatsApp assistant**, initially serving Telugu and English speakers to discover government welfare schemes. Powered by AI/NLU, it converts a single voice or text message into a personalized profile, matches it to relevant programs, and delivers guidance. Our vision: to one day support **any language and context globally** and operate seamlessly on a **modular smartphone platform**.

---

## 🧭 Vision & Roadmap

1. **Phase 1:** Launch on WhatsApp for Telugu + English.
2. **Phase 2:** Improve AI‑powered understanding (NLU).
3. **Phase 3:** Future hardware support: build a **modular phone** with the assistant pre-installed.
4. **Phase 4:** Expand to **universal, language-agnostic voice assistant** for civic services worldwide.

---

## 🚀 Features

- **Voice + Text Input** – via WhatsApp using Whisper for Telugu and English.
- **AI‑Powered NLU** – extracts structured user data: age, income, intent, gender, caste, etc.
- **Intelligent Matching** – selects relevant welfare schemes from a verified database.
- **Seamless Workflow** – sends document requirements and next steps via WhatsApp.

---

## 🛠️ Tech Stack

| Component            | Technology                    |
|----------------------|-------------------------------|
| Backend              | Python 3.10+, Flask           |
| NLU & Embeddings     | OpenAI/Google Gemini/Fine‑tuned Whisper |
| Vector Search        | LangChain + pgvector          |
| Messaging            | Twilio WhatsApp API           |
| Database             | SQLite / JSON session store   |
| Logging & Testing    | `logging_config.py`, `pytest` |

---

## ⚡ Quick Start

```bash
git clone https://github.com/your-org/safesound-assistant.git
cd safesound-assistant
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add your TWILIO_* and GEMINI_API_KEY
python app.py
````

Use ngrok to expose your bot and update your Twilio webhook accordingly:

```bash
ngrok http 5000
```

---

## 📁 Project Layout

```
safesound/
├── app.py             # Webhook + dialog flow
├── nlu.py             # NLU & schema extraction
├── utils.py           # Telugu-parsers, number handling
├── logging_config.py  # Structured logging
├── session_manager.py # Tracks user conversations
├── welfare_schemes.db # Sample scheme database
├── user_states.json   # Persistent session storage
├── requirements.txt   # Installed dependencies
├── .env.example       # Env vars template
└── README.md          # This file
```

---

## 🎯 Principles & Best Practices

* **Minimal & Modular**: Logic split into clear, single-responsibility modules.
* **Type-safe & Tested**: Type hints across codebase + unit tests.
* **Localisation-first**: Built for Telugu-English; designed to scale across languages.
* **Security-minded**: Validates Twilio webhook signatures, secures keys with `.env`.
* **Extensible**: Easy to plug in new states, languages, or hardware platform.

---

## 🧬 Future Expansion

* Improve **NLU accuracy** using Gemini or fine-tuned models.
* Add **language detection** for seamless user experience.
* Build on **modular phone prototype** once NLU scales.
* Scale pilot with nonprofits and AP state government; then expand globally.

---

## 🤝 Contributing & Contact

We welcome contributions—whether through code, documentation, or issue reports. Please open an issue or submit a PR, and include tests for new features.

**Founder & Engineer:**
Prudvish Korrapati — Prudvish@safesound.ai

---

## 📄 License

This project is released under the **MIT License**. Check [LICENSE] for details.


