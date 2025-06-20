import os
from gtts import gTTS
import time

# --- Simulated conversation steps ---
CONVO_STEPS = [
    ("greeting", "బాబు! స్వాగతం. మీకోసమా లేక ఇతరుల కోసమా?"),
    ("target_confirm", "మీ వయస్సు?"),
    ("age_confirm", "మీ లింగం? (మగ/ఆడ)"),
    ("gender_confirm", "మీ వృత్తి? (ఉదా: రైతు)"),
    ("occupation_confirm", "ధన్యవాదాలు! మీ వివరాలు నమోదు అయ్యాయి."),
]


def speak_and_display(text, lang="te"):
    print(f"[DEBUG] Current working directory: {os.getcwd()}")
    print(f"Bot: {text}")
    try:
        filename = f"tts_debug_{int(time.time())}.mp3"
        print(f"[DEBUG] About to create gTTS object with text: {text} and lang: {lang}")
        tts = gTTS(text=text, lang=lang)
        print(f"[DEBUG] About to save audio file as: {filename}")
        tts.save(filename)
        print(f"[TTS] Saved audio as: {filename}")
        print(f"[DEBUG] About to play audio file: {filename}")
        if os.name == 'posix':
            os.system(f"afplay '{filename}'")
        else:
            os.system(f"start {filename}")
        time.sleep(0.5)
    except Exception as e:
        print(f"[TTS ERROR]: {e}")


def main():
    print("=== Telugu Voice+Display Bot Prototype ===")
    user_input = input("User (say something): ").strip()
    # Accept both Telugu and English 'babu' for easier testing
    if user_input.startswith("బాబు") or user_input.lower().startswith("babu"):
        print("[Activation: Full Voice+Display Mode]")
        for step_name, bot_text in CONVO_STEPS:
            speak_and_display(bot_text, lang="te")
            # Simulate user response for prototype
            user_input = input(f"User ({step_name}): ")
    else:
        print("[Normal mode: Only display, no speech]")
        for step_name, bot_text in CONVO_STEPS:
            print(f"Bot: {bot_text}")
            user_input = input(f"User ({step_name}): ")

if __name__ == "__main__":
    main()
