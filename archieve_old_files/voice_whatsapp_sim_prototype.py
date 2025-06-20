import os
from gtts import gTTS
import whisper

# --- State Machine Logic ---
def get_next_bot_reply(state, user_text):
    """Simple example state machine: returns next bot reply and updated state."""
    # This is a placeholder; expand as needed for your real flow
    if state == 'START':
        return 'మీ వయస్సు?', 'AWAITING_AGE'
    elif state == 'AWAITING_AGE':
        return 'మీ లింగం? (మగ/ఆడ)', 'AWAITING_GENDER'
    elif state == 'AWAITING_GENDER':
        return 'మీ వృత్తి? (ఉదా: రైతు)', 'AWAITING_OCCUPATION'
    elif state == 'AWAITING_OCCUPATION':
        return 'ధన్యవాదాలు! మీ వివరాలు నమోదు అయ్యాయి.', 'END'
    else:
        return 'సంప్రదించండి.', 'END'

# --- STT: Transcribe audio file using Whisper ---
def transcribe_audio(audio_path, model=None):
    if model is None:
        model = whisper.load_model('base')
    result = model.transcribe(audio_path, language='te')
    return result['text'].strip()

# --- TTS: Generate audio file from text ---
def text_to_speech(text, output_path, lang='te'):
    tts = gTTS(text=text, lang=lang)
    tts.save(output_path)
    print(f'[TTS] Saved: {output_path}')

# --- Main Simulation Loop ---
def main():
    print('=== WhatsApp Voice Bot File-Based Prototype ===')
    model = whisper.load_model('base')
    state = 'START'
    step = 1
    while state != 'END':
        input_audio = f'input_{step}.wav'
        if not os.path.exists(input_audio):
            print(f'[WAITING] Please provide: {input_audio}')
            break
        print(f'[STT] Transcribing {input_audio}...')
        user_text = transcribe_audio(input_audio, model)
        print(f'[USER SAID]: {user_text}')
        bot_reply, state = get_next_bot_reply(state, user_text)
        print(f'[BOT REPLY]: {bot_reply}')
        # Save text reply
        with open(f'reply_{step}.txt', 'w', encoding='utf-8') as f:
            f.write(bot_reply)
        # Generate TTS audio reply
        text_to_speech(bot_reply, f'reply_{step}.mp3', lang='te')
        step += 1
    print('=== Conversation Ended ===')

if __name__ == '__main__':
    main()
