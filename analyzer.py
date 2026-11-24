import os
import torch
import librosa
import numpy as np
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import imageio_ffmpeg
import tempfile

os.environ["PATH"] = tempfile.gettempdir() + os.pathsep + os.environ.get("PATH", "")

MODEL_NAME = "superb/hubert-large-superb-er"

print("Loading HuBERT acoustic emotion model...")
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME, local_files_only=True)
model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME, local_files_only=True)
model.eval()

AUDIO_LABEL_MAP = {
    "neu": "neutral",
    "hap": "joy",
    "sad": "sadness",
    "ang": "anger"
}

def analyze_audio(file_path):
    # 1. Load Audio at 16kHz
    y, sr = librosa.load(file_path, sr=16000)
    duration = float(librosa.get_duration(y=y, sr=sr))

    if len(y.shape) > 1:
        y = np.mean(y, axis=1)
    y = y.astype(np.float32)

    # 2. High-Accuracy Fundamental Pitch Tracking using Probabilistic YIN (pYIN)
    # Pitch bounds: 65 Hz to 350 Hz (standard human vocal range)
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=65, fmax=350, sr=sr, frame_length=2048, hop_length=512
        )
        valid_pitches = f0[voiced_flag & ~np.isnan(f0)]
    except Exception:
        valid_pitches = np.array([])
        f0 = np.array([])

    if len(valid_pitches) > 0:
        median_pitch = float(np.median(valid_pitches))
        avg_pitch = float(np.mean(valid_pitches))
        pitch_std = float(np.std(valid_pitches))
        pitch_iqr = float(np.percentile(valid_pitches, 75) - np.percentile(valid_pitches, 25))
    else:
        median_pitch = 130.0
        avg_pitch = 130.0
        pitch_std = 0.0
        pitch_iqr = 0.0

    # Clean pitch timeline for graphs (replace unvoiced with NaN)
    pitch_timeline = [float(p) if (not np.isnan(p) and p > 40) else np.nan for p in f0]

    # 3. RMS Energy & Vocal Projection
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    avg_energy = float(np.mean(rms))
    energy_max = float(np.max(rms)) if len(rms) else 0.0

    # 4. Spectral Formants & Timbral Harmonics
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    avg_brightness = float(np.mean(spectral_centroids))

    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    avg_rolloff = float(np.mean(spectral_rolloff))

    # Vocal Jitter proxy (micro-frequency perturbation)
    if len(valid_pitches) > 3:
        diffs = np.abs(np.diff(valid_pitches))
        jitter_pct = float(np.mean(diffs) / (avg_pitch + 1e-6)) * 100.0
    else:
        jitter_pct = 1.0

    # 5. Scientific Multi-Parameter Acoustic Gender Profiler
    # Uses median fundamental pitch (F0), spectral centroid resonance, and roll-off
    gender_confidence = 0
    if median_pitch >= 165:
        # Strongly female pitch
        gender = "Female"
        gender_confidence = min(98, int(80 + (median_pitch - 165) * 0.4))
    elif median_pitch <= 145:
        # Strongly male pitch
        gender = "Male"
        gender_confidence = min(98, int(80 + (145 - median_pitch) * 0.4))
    else:
        # Ambiguous boundary (145-165 Hz): vote using Vocal Tract Resonance (Centroid)
        if avg_brightness > 2300 or avg_rolloff > 4200:
            gender = "Female"
            gender_confidence = 78
        else:
            gender = "Male"
            gender_confidence = 78

    # 6. Scientific Age Bracket Classification
    # Considers fundamental pitch, vocal tract length resonance, and micro-jitter
    if gender == "Female":
        if median_pitch > 225 and avg_brightness > 2600:
            age_group = "Youth / Young Adult (<21)"
            age_confidence = "High (High F0 & Bright Resonance)"
        elif jitter_pct > 6.5 or avg_brightness < 1650:
            age_group = "Senior / Mature Adult (45+)"
            age_confidence = "Moderate (Vocal Jitter & Lower Formants)"
        else:
            age_group = "Adult (21–45)"
            age_confidence = "High (Stable F0 Contour)"
    else:
        # Male
        if median_pitch > 150 and avg_brightness > 2400:
            age_group = "Youth / Young Adult (<21)"
            age_confidence = "Moderate (Elevated Youth Pitch)"
        elif median_pitch < 105 or jitter_pct > 7.0:
            age_group = "Senior / Mature Adult (45+)"
            age_confidence = "High (Deep Sub-Harmonics & Jitter)"
        else:
            age_group = "Adult (21–45)"
            age_confidence = "High (Standard Chest Resonance)"

    # 7. HuBERT Deep Acoustic Emotion Model
    inputs = feature_extractor(y, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)[0].numpy()

    id2label = model.config.id2label
    audio_emotions = {}
    for idx, prob in enumerate(probs):
        raw_label = id2label.get(idx, f"label_{idx}").lower()
        mapped = AUDIO_LABEL_MAP.get(raw_label, raw_label)
        audio_emotions[mapped] = round(float(prob) * 100, 1)

    # 8. Explainable Voice Confidence Metric (0 - 100)
    energy_score = min(100.0, (avg_energy / 0.045) * 55.0)
    stability_score = max(0.0, 100.0 - (pitch_std * 0.35) - (jitter_pct * 3.0))
    confidence_score = int(np.clip((energy_score * 0.55) + (stability_score * 0.45), 15, 99))

    # Evidence & Diagnostics
    reasons = []
    if avg_energy > 0.04:
        reasons.append("⚡ **Solid Vocal Projection**: Assertive chest volume with commanding acoustic delivery.")
    elif avg_energy > 0.018:
        reasons.append("🔹 **Balanced Volume Range**: Audible conversational articulation with stable dynamic control.")
    else:
        reasons.append("⚠️ **Whisper Dynamics Detected**: Subdued vocal volume indicating reservation or fatigue.")

    if pitch_std < 35.0 and jitter_pct < 4.0:
        reasons.append("🛡️ **Rock-Solid Pitch Control**: Razor-sharp fundamental frequency stability with minimal jitter.")
    elif pitch_std <= 75.0:
        reasons.append("✨ **Expressive Cadence**: Natural emotional inflection without nervous tremor.")
    else:
        reasons.append("⚠️ **Micro-Pitch Instability**: Frequency flutter suggests stress, anxiety, or high emotional charge.")

    # Speech Coach Feedback
    coaching_tips = []
    if avg_energy < 0.022:
        coaching_tips.append("📢 **Diaphragm Amplification**: Inhale fully from your abdomen to project sound without vocal strain.")
    if pitch_std > 75.0:
        coaching_tips.append("🧘 **Tempo Regulation**: Slow your speaking cadence slightly to stabilize pitch modulation.")
    if not coaching_tips:
        coaching_tips.append("🏆 **Exceptional Delivery**: Masterful equilibrium of acoustic projection, pitch control, and cadence.")

    # Distress Detection
    is_distressed = (pitch_std > 80.0 and avg_energy < 0.03) or (jitter_pct > 8.0 and avg_energy < 0.025)

    return {
        "audio_emotions": audio_emotions,
        "confidence_score": confidence_score,
        "reasons": reasons,
        "coaching_tips": coaching_tips,
        "demographics": {
            "gender": gender,
            "gender_confidence": f"{gender_confidence}%",
            "age_group": age_group,
            "age_confidence": age_confidence,
            "f0_median": f"{round(median_pitch, 1)} Hz"
        },
        "acoustics": {
            "median_pitch": f"{round(median_pitch, 1)} Hz",
            "avg_pitch": f"{round(avg_pitch, 1)} Hz",
            "pitch_stability": "High (Rock-steady)" if pitch_std < 35 else ("Moderate" if pitch_std < 75 else "Unsteady / Trembling"),
            "loudness": "High (Projected)" if avg_energy > 0.04 else ("Moderate" if avg_energy > 0.018 else "Low (Soft)"),
            "tone": "Warm Resonance" if avg_brightness < 2000 else "Bright & Crisp",
            "jitter": f"{round(jitter_pct, 2)}%",
            "duration": f"{round(duration, 2)}s"
        },
        "is_distressed": is_distressed,
        "valence": int(audio_emotions.get("joy", 0) - audio_emotions.get("sadness", 0) - audio_emotions.get("anger", 0) * 0.5),
        "waveform": y,
        "sample_rate": sr,
        "pitch_timeline": pitch_timeline
    }

def fuse_emotions(audio_result, text_result):
    EMOTION_KEYS = ["joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral"]

    text_probs = {}
    for s in text_result.get("sentences", []):
        key = s.get("top_emotion_key", "neutral")
        text_probs[key] = text_probs.get(key, 0) + s.get("confidence", 0)

    total_text = sum(text_probs.values()) or 1
    text_probs = {k: (v / total_text) * 100 for k, v in text_probs.items()}

    audio_probs = audio_result["audio_emotions"]

    # Weighted Multimodal Fusion (60% semantic sentiment + 40% acoustic biometric)
    fused = {}
    for key in EMOTION_KEYS:
        t = text_probs.get(key, 0)
        a = audio_probs.get(key, 0)
        fused[key] = round(t * 0.60 + a * 0.40, 1)

    total_fused = sum(fused.values()) or 1
    fused = {k: round((v / total_fused) * 100, 1) for k, v in fused.items()}
    top_fused_key = max(fused, key=fused.get)

    EMOTION_THEME = {
        "joy":      ("😄", "Joyful", "#ffffff", "rgba(255, 255, 255, 0.25)"),
        "sadness":  ("🌧️", "Sad / Melancholic", "#ffffff", "rgba(255, 255, 255, 0.25)"),
        "anger":    ("⚡", "Angry / Tense", "#ffffff", "rgba(255, 255, 255, 0.25)"),
        "fear":     ("🌪️", "Anxious", "#ffffff", "rgba(255, 255, 255, 0.25)"),
        "disgust":  ("🥀", "Disdainful", "#ffffff", "rgba(255, 255, 255, 0.25)"),
        "surprise": ("✨", "Surprised", "#ffffff", "rgba(255, 255, 255, 0.25)"),
        "neutral":  ("⚖️", "Calm & Composed", "#ffffff", "rgba(255, 255, 255, 0.25)"),
    }

    icon, label, color, glow = EMOTION_THEME.get(top_fused_key, ("🎙️", top_fused_key.title(), "#ffffff", "rgba(255, 255, 255, 0.25)"))

    return {
        "top_emotion": f"{icon} {label}",
        "top_key": top_fused_key,
        "top_color": color,
        "top_glow": glow,
        "top_emotion_score": fused[top_fused_key],
        "fused_emotions": fused
    }
