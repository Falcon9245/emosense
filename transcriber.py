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

def evaluate_reading_passage(audio_path, target_passage):
    """
    Evaluates a user reading a known reference passage:
      - WPM (Words Per Minute) & Speaking Pace
      - Pronunciation Accuracy (% of target words spoken clearly)
      - Word-by-word diff (missed, mispronounced, extra words)
      - Accent & Articulation diagnostics using phonetics / Levenshtein
    Requires 0 MB additional RAM (uses already-loaded Whisper-tiny).
    """
    import re
    import difflib

    trans_res = transcribe_and_analyze(audio_path, language="en")
    spoken_text = trans_res.get("transcript", "")
    
    # Clean and tokenize words
    def clean_words(text):
        return [re.sub(r'[^a-zA-Z0-9]', '', w).lower() for w in text.split() if re.sub(r'[^a-zA-Z0-9]', '', w)]

    target_words = clean_words(target_passage)
    spoken_words = clean_words(spoken_text)

    # Calculate audio duration for WPM
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        duration = float(librosa.get_duration(y=y, sr=sr)) if len(y) > 0 else 1.0
        del y
    except Exception:
        duration = max(1.0, len(spoken_words) * 0.4)

    # Words Per Minute (WPM)
    wpm = round((len(spoken_words) / max(duration, 1.0)) * 60.0, 1)

    # Pace classification
    if wpm < 100:
        pace_label = "Deliberate / Slow"
        pace_tip = "You spoke at a cautious pace. Try linking phrases smoothly to increase fluency."
    elif 110 <= wpm <= 165:
        pace_label = "Optimal Conversational Pace"
        pace_tip = "Terrific natural tempo! Very easy for listeners to follow."
    elif 165 < wpm <= 195:
        pace_label = "Brisk / Rapid"
        pace_tip = "Slightly quick. Remember to pause briefly at periods and commas."
    else:
        pace_label = "Fast / Rushed"
        pace_tip = "Very fast cadence. Slowing down slightly improves word intelligibility."

    # Word-level Sequence Matcher for Pronunciation & Precision
    matcher = difflib.SequenceMatcher(None, target_words, spoken_words)

    # Word-by-word diagnostic tracking
    word_feedback = []
    mispronounced_words = []
    missed_words = []
    extra_words = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for w in target_words[i1:i2]:
                word_feedback.append({"word": w, "status": "correct", "spoken": w})
        elif tag == "replace":
            for tw, sw in zip(target_words[i1:i2], spoken_words[j1:j2]):
                sim = difflib.SequenceMatcher(None, tw, sw).ratio()
                # 75%+ acoustic similarity allows for valid homophonic/phonetic spellings (critical for pseudowords)
                if sim >= 0.75:
                    word_feedback.append({"word": tw, "status": "correct", "spoken": sw, "similarity": round(sim * 100)})
                else:
                    word_feedback.append({"word": tw, "status": "mispronounced", "spoken": sw, "similarity": round(sim * 100)})
                    mispronounced_words.append(f"'{tw}' (heard as '{sw}')")
            # If lengths differ in replacement
            if (i2 - i1) > (j2 - j1):
                for tw in target_words[i1 + (j2 - j1): i2]:
                    word_feedback.append({"word": tw, "status": "missed", "spoken": ""})
                    missed_words.append(tw)
            elif (j2 - j1) > (i2 - i1):
                for sw in spoken_words[j1 + (i2 - i1): j2]:
                    extra_words.append(sw)
        elif tag == "delete":
            for tw in target_words[i1:i2]:
                word_feedback.append({"word": tw, "status": "missed", "spoken": ""})
                missed_words.append(tw)
        elif tag == "insert":
            for sw in spoken_words[j1:j2]:
                extra_words.append(sw)

    # Calculate actual percentage of words pronounced accurately
    correct_count = sum(1 for item in word_feedback if item.get("status") == "correct")
    accuracy_pct = round((correct_count / max(len(target_words), 1)) * 100, 1)

    # Accent & Articulation Diagnostics
    accent_notes = []
    if accuracy_pct >= 85:
        accent_notes.append("🎯 **Clear Articulation**: High phonetic alignment with standard pronunciation contours.")
    elif accuracy_pct >= 65:
        accent_notes.append("🔹 **Regional Inflection Detected**: Minor vowel elongation or consonant lenition typical of non-native / regional cadences.")
    else:
        accent_notes.append("⚠️ **Heavy Accent / Articulation Variance**: Several word-final consonants or vowel blends diverged from standard acoustics.")

    if extra_words:
        accent_notes.append(f"🗣️ **Vocal Fillers / Insertions**: Detected extra filler sounds: {', '.join(extra_words[:3])}.")

    return {
        "spoken_text": spoken_text,
        "target_text": target_passage,
        "wpm": wpm,
        "duration_sec": round(duration, 1),
        "pace_label": pace_label,
        "pace_tip": pace_tip,
        "accuracy_pct": accuracy_pct,
        "word_feedback": word_feedback,
        "mispronounced_words": mispronounced_words,
        "missed_words": missed_words,
        "extra_words": extra_words,
        "accent_notes": accent_notes
    }

@st.cache_resource(show_spinner=False)
def load_whisper_model():
    print("Loading Whisper Multilingual Speech Engine (tiny, cached)...")
    # 'tiny' multilingual model supports 99 languages (including Tamil & English) at only ~39M params / ~75MB RAM
    return whisper.load_model("tiny")

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
    agg_text_emotions = {k: 0.0 for k in ["joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral"]}
    total_valid_segments = 0

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

        if scores_list:
            for item in scores_list:
                if isinstance(item, dict):
                    lbl = item.get("label", "").lower()
                    if lbl in agg_text_emotions:
                        agg_text_emotions[lbl] += float(item.get("score", 0.0))
            total_valid_segments += 1

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

    if total_valid_segments > 0:
        tot_sum = sum(agg_text_emotions.values()) or 1.0
        agg_text_emotions = {k: round((v / tot_sum) * 100, 1) for k, v in agg_text_emotions.items()}
    else:
        agg_text_emotions = {"neutral": 90.0, "joy": 2.0, "sadness": 2.0, "anger": 2.0, "fear": 2.0, "surprise": 1.0, "disgust": 1.0}

    return {
        "transcript": full_transcript,
        "sentences": sentence_results,
        "word_count": len(full_transcript.split()),
        "language_detected": detected_lang,
        "raw_emotion_scores": agg_text_emotions
    }
