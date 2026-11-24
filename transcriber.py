import gc
import whisper
import librosa
import numpy as np
import nltk
from transformers import pipeline
import streamlit as st

# Fallback-safe sentence tokenizer (no network requests required)
def split_into_sentences(text):
    try:
        return nltk.sent_tokenize(text)
    except Exception:
        import re
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        return sentences if sentences else [text]

@st.cache_resource(show_spinner=False)
def load_whisper_model():
    print("Loading Whisper Multilingual Speech Engine (cached)...")
    # 'tiny' or 'base' model - 'base' fits within cached tier and supports 99 languages natively
    return whisper.load_model("base")

@st.cache_resource(show_spinner=False)
def load_text_emotion_classifier():
    print("Loading DistilRoBERTa Emotion Classifier (cached)...")
    try:
        classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,
            local_files_only=True
        )
    except Exception:
        classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None
        )
    return classifier

TEXT_EMOTION_ICONS = {
    "anger":    ("⚡", "Anger / Frustration"),
    "disgust":  ("🥀", "Disgust"),
    "fear":     ("🌪️", "Anxiety / Fear"),
    "joy":      ("😄", "Joy / Elation"),
    "neutral":  ("⚖️", "Neutral / Composed"),
    "sadness":  ("🌧️", "Sadness / Grief"),
    "surprise": ("✨", "Surprise"),
}

def transcribe_and_analyze(audio_path, language="dual"):
    """
    Transcribes audio with full support for:
      - 'dual': Auto Code-Switching (seamlessly handles Tamil + English in the same clip)
      - 'en': English forced
      - 'ta': Tamil forced
    """
    try:
        y, sr = librosa.load(audio_path, sr=16000)
    except Exception as e:
        print(f"Error loading audio in transcriber: {e}")
        y, sr = np.array([], dtype=np.float32), 16000

    if len(y.shape) > 1:
        y = np.mean(y, axis=1)
    y = y.astype(np.float32)

    if len(y) < 2048:
        print("Warning: Audio buffer is empty or too short for transcription.")
        return {
            "transcript": "[No audio recorded / Empty clip]",
            "sentences": [],
            "dominant_text_emotion": "neutral",
            "word_count": 0,
            "language_detected": "en"
        }

    whisper_model = load_whisper_model()
    text_emotion_classifier = load_text_emotion_classifier()

    transcribe_kwargs = {"fp16": False, "task": "transcribe"}

    # In 'dual' mode, we leave language unspecified so Whisper automatically detects and switches
    if language == "en":
        transcribe_kwargs["language"] = "en"
    elif language == "ta":
        transcribe_kwargs["language"] = "ta"
    # if 'dual' or 'auto', language is omitted, triggering Whisper's internal language detector

    print(f"Transcribing audio with Whisper [Mode: {language}]...")
    try:
        result = whisper_model.transcribe(y, **transcribe_kwargs)
    except Exception as e:
        print(f"Whisper automatic detection fallback: {e}")
        transcribe_kwargs["language"] = "en"
        result = whisper_model.transcribe(y, **transcribe_kwargs)

    full_transcript = result.get("text", "").strip()
    whisper_segments = result.get("segments", [])
    detected_lang = result.get("language", "en")

    if not full_transcript:
        return {
            "transcript": "[No distinct speech detected]",
            "sentences": [],
            "dominant_text_emotion": "neutral",
            "word_count": 0,
            "language_detected": detected_lang
        }

    sentence_results = []
    segments_to_use = whisper_segments if whisper_segments else [{"text": full_transcript, "start": 0.0, "end": 0.0}]

    for seg in segments_to_use:
        seg_text = seg["text"].strip()
        if not seg_text:
            continue

        try:
            raw_output = text_emotion_classifier(seg_text)
            if isinstance(raw_output, list) and len(raw_output) > 0:
                scores_list = raw_output[0] if isinstance(raw_output[0], list) else raw_output
            else:
                scores_list = []
            
            sorted_emotions = sorted(
                [x for x in scores_list if isinstance(x, dict)],
                key=lambda x: x.get("score", 0),
                reverse=True
            )
        except Exception:
            sorted_emotions = []

        if not sorted_emotions:
            top_label = "neutral"
            top_score = 0.5
            emotions_dict = {"Neutral / Composed": 50.0}
        else:
            top_label = sorted_emotions[0].get("label", "neutral")
            top_score = sorted_emotions[0].get("score", 0.0)
            emotions_dict = {
                TEXT_EMOTION_ICONS.get(e.get("label", ""), ("❓", e.get("label", "").title()))[1]: round(e.get("score", 0.0) * 100, 1)
                for e in sorted_emotions if isinstance(e, dict)
            }

        icon, readable = TEXT_EMOTION_ICONS.get(top_label, ("❓", top_label.title()))

        sentence_results.append({
            "text": seg_text,
            "start_time": round(float(seg.get("start", 0.0)), 2),
            "end_time": round(float(seg.get("end", 0.0)), 2),
            "top_emotion": readable,
            "top_emotion_key": top_label,
            "icon": icon,
            "confidence": round(float(top_score) * 100, 1),
            "all_emotions": emotions_dict
        })

    del y, whisper_segments
    gc.collect()

    return {
        "transcript": full_transcript,
        "sentences": sentence_results,
        "word_count": len(full_transcript.split()),
        "language_detected": detected_lang
    }
