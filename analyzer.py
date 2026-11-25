import os
import gc
import torch
import librosa
import numpy as np
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import tempfile
import streamlit as st

# Ensure ffmpeg from system path or imageio is recognized if present
try:
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    os.environ["PATH"] = os.path.dirname(ffmpeg_exe) + os.pathsep + os.environ.get("PATH", "")
except (ImportError, ModuleNotFoundError, Exception):
    pass

MODEL_NAME = "superb/hubert-base-superb-er"

@st.cache_resource(show_spinner=False)
def load_hubert_model():
    print("Loading HuBERT acoustic emotion model (cached)...")
    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME, local_files_only=True)
        model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME, local_files_only=True)
    except Exception:
        feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
        model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return feature_extractor, model

AUDIO_LABEL_MAP = {
    "neu": "neutral",
    "hap": "joy",
    "sad": "sadness",
    "ang": "anger"
}

def analyze_audio(file_path):
    # 1. Load Audio at 16kHz
    try:
        y, sr = librosa.load(file_path, sr=16000)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        y, sr = np.array([], dtype=np.float32), 16000

    if len(y.shape) > 1:
        y = np.mean(y, axis=1)
    y = y.astype(np.float32)

    duration = float(librosa.get_duration(y=y, sr=sr)) if len(y) > 0 else 0.0

    # 1b. Audio Quality Gate: Reject corrupt, silent, or sub-second recordings
    if len(y) < 2048 or duration < 0.8:
        print("Warning: Audio signal is empty or too short (< 0.8s).")
        return {
            "status": "insufficient_audio",
            "top_audio_emotion": "Insufficient Speech",
            "top_audio_emotion_key": "neutral",
            "top_audio_score": 0.0,
            "audio_emotions": {"neutral": 20.0, "joy": 20.0, "sadness": 20.0, "anger": 20.0, "fear": 20.0},
            "confidence_score": 0,
            "reasons": ["⚠️ **Insufficient Audio**: Recording is under 1 second. Please record at least 2–4 seconds of clear speech."],
            "coaching_tips": ["Speak directly into your microphone for at least 2–4 seconds."],
            "acoustics": {
                "avg_pitch": "N/A",
                "median_pitch": "N/A",
                "pitch_std": "0.0 Hz",
                "pitch_range": "0.0 Hz",
                "avg_energy": 0.0,
                "spectral_brightness": "0 Hz",
                "spectral_rolloff": "0 Hz",
                "jitter": "0.0%",
                "duration": f"{round(duration, 2)}s"
            },
            "is_distressed": False,
            "valence": 0,
            "waveform": y if len(y) > 0 else np.zeros(1600, dtype=np.float32),
            "sample_rate": sr,
            "pitch_timeline": []
        }

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

    # Quality Gate Check 2: Require at least 6 voiced speech frames (reject ambient noise or silence)
    if len(valid_pitches) < 6:
        return {
            "status": "insufficient_audio",
            "top_audio_emotion": "Unvoiced / Silent",
            "top_audio_emotion_key": "neutral",
            "top_audio_score": 0.0,
            "audio_emotions": {"neutral": 20.0, "joy": 20.0, "sadness": 20.0, "anger": 20.0, "fear": 20.0},
            "confidence_score": 0,
            "reasons": ["⚠️ **No Voiced Speech Detected**: Recording contains silence or ambient background noise with no vocal harmonic frequencies."],
            "coaching_tips": ["Speak closer to the microphone in a quiet room."],
            "acoustics": {
                "avg_pitch": "N/A",
                "median_pitch": "N/A",
                "pitch_std": "0.0 Hz",
                "pitch_range": "0.0 Hz",
                "avg_energy": 0.0,
                "spectral_brightness": "0 Hz",
                "spectral_rolloff": "0 Hz",
                "jitter": "0.0%",
                "duration": f"{round(duration, 2)}s"
            },
            "is_distressed": False,
            "valence": 0,
            "waveform": y if len(y) > 0 else np.zeros(1600, dtype=np.float32),
            "sample_rate": sr,
            "pitch_timeline": []
        }

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

    # Vocal Jitter: Micro-frequency perturbation around smooth intonation contour
    if len(valid_pitches) > 6:
        smoothed_f0 = np.convolve(valid_pitches, np.ones(5) / 5.0, mode="same")
        residual = np.abs(valid_pitches[2:-2] - smoothed_f0[2:-2])
        jitter_pct = float(np.mean(residual) / (avg_pitch + 1e-6)) * 100.0
    else:
        jitter_pct = 0.5

    # 5. HuBERT Deep Acoustic Emotion Model
    feature_extractor, model = load_hubert_model()
    inputs = feature_extractor(y, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        raw_logits = model(**inputs).logits.cpu().numpy()[0]
        # Gentle de-biasing of neutral prior dominance (-0.6)
        adj_logits = np.array(raw_logits, dtype=float)
        adj_logits[0] -= 0.6  # remove neutral prior dominance

        # Temperature scaling (T=0.85) sharpens acoustic distinctiveness
        scaled_logits = adj_logits / 0.85
        exp_l = np.exp(scaled_logits - np.max(scaled_logits))
        probs = exp_l / np.sum(exp_l)
    del inputs
    gc.collect()

    id2label = model.config.id2label
    raw_hubert_emotions = {}
    for idx, prob in enumerate(probs):
        raw_label = id2label.get(idx, f"label_{idx}").lower()
        mapped = AUDIO_LABEL_MAP.get(raw_label, raw_label)
        raw_hubert_emotions[mapped] = float(prob) * 100.0

    # 5b. Deep Acoustic Prosody & Biometric Calibration Layer
    # Physical acoustics directly measure biological vocal tract, subglottal pressure, and respiratory state
    scores = {"neutral": 20.0, "joy": 5.0, "sadness": 5.0, "anger": 5.0, "fear": 5.0}

    # 1. Sadness Boost: quiet volume (< 0.020) and/or flat monotone (< 20.0 Hz)
    s_boost = 0.0
    if avg_energy < 0.020:
        s_boost += 20.0 + max(0.0, (0.020 - avg_energy) / 0.012 * 35.0)
    if pitch_std < 20.0 and avg_energy < 0.026:
        s_boost += 15.0 + max(0.0, (20.0 - pitch_std) / 12.0 * 25.0)
    scores["sadness"] += min(65.0, s_boost)

    # 2. Fear / Anxiety Boost: genuine micro-tremor (jitter > 1.35%) OR acute vocal tension with tremor (pitch_std > 35 Hz + jitter > 1.05% + non-booming volume)
    f_boost = 0.0
    if jitter_pct > 1.35:
        f_boost += 25.0 + min(40.0, (jitter_pct - 1.35) / 0.8 * 40.0)
    elif pitch_std > 35.0 and jitter_pct > 1.05 and avg_energy < 0.034:
        f_boost += 25.0 + min(40.0, (pitch_std - 35.0) / 25.0 * 20.0 + (jitter_pct - 1.05) / 0.5 * 20.0)
    scores["fear"] += min(65.0, f_boost)

    # 3. Anger / Sternness Boost: assertive volume (> 0.033) AND (elevated pitch variation > 34 or brightness > 1950)
    a_boost = 0.0
    if avg_energy > 0.033:
        a_boost += 20.0 + min(35.0, (avg_energy - 0.033) / 0.018 * 35.0)
        if pitch_std > 34.0 or avg_brightness > 1950:
            a_boost += 15.0 + min(20.0, max((pitch_std - 34.0) / 20.0 * 20.0, (avg_brightness - 1950) / 400.0 * 20.0))
    scores["anger"] += min(65.0, a_boost)

    # 4. Joy / Exuberance Boost: elevated melodic pitch (> 170 Hz) + expressive modulation (> 28 Hz) + good volume (> 0.024) + low jitter (< 1.05%)
    j_boost = 0.0
    if median_pitch > 170.0 and pitch_std > 28.0 and avg_energy > 0.024 and jitter_pct < 1.05:
        j_boost += 25.0 + min(40.0, (median_pitch - 170.0) / 50.0 * 20.0 + (pitch_std - 28.0) / 25.0 * 20.0)
    scores["joy"] += min(65.0, j_boost)

    # Neutral Prior Balancing: scale down neutral when clear acoustic emotional signals are present
    max_emo_boost = max(s_boost, f_boost, a_boost, j_boost)
    if max_emo_boost > 20.0:
        scores["neutral"] = max(4.0, 20.0 - (max_emo_boost - 20.0) * 0.4)
    else:
        # True calm baseline
        scores["neutral"] += 65.0

    tot_p = sum(scores.values()) or 1.0
    prosody_probs = {k: (v / tot_p) * 100.0 for k, v in scores.items()}

    # HuBERT Alignment & Bias Normalization
    h_emotions = dict(raw_hubert_emotions)
    if s_boost > 20.0:
        h_emotions["sadness"] = max(h_emotions.get("sadness", 0.0), h_emotions.get("neutral", 0.0) * 0.70)
        h_emotions["neutral"] *= 0.35
    elif f_boost > 20.0:
        h_emotions["fear"] = max(25.0, h_emotions.get("neutral", 0.0) * 0.60, h_emotions.get("anger", 0.0) * 0.60)
        h_emotions["neutral"] *= 0.35
        h_emotions["anger"] *= 0.40
    elif a_boost > 20.0:
        h_emotions["anger"] = max(h_emotions.get("anger", 0.0), h_emotions.get("neutral", 0.0) * 0.70)
        h_emotions["neutral"] *= 0.35
    elif j_boost > 20.0:
        h_emotions["joy"] = max(h_emotions.get("joy", 0.0), h_emotions.get("neutral", 0.0) * 0.70)
        h_emotions["neutral"] *= 0.35

    # Multi-representation Acoustic Fusion (HuBERT Deep Transformer 35% + Librosa Physical Prosody 65%)
    audio_emotions = {}
    for emo in ["joy", "sadness", "anger", "fear", "neutral"]:
        h_val = h_emotions.get(emo, 0.0)
        p_val = prosody_probs.get(emo, 0.0)
        audio_emotions[emo] = (h_val * 0.35) + (p_val * 0.65)

    tot_audio = sum(audio_emotions.values()) or 1.0
    audio_emotions = {k: round((v / tot_audio) * 100, 1) for k, v in audio_emotions.items()}

    # 6. Explainable Voice Confidence Metric (0 - 100)
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

    if pitch_std < 35.0 and jitter_pct < 2.5:
        reasons.append("🛡️ **Rock-Solid Pitch Control**: Razor-sharp fundamental frequency stability with minimal jitter.")
    elif pitch_std <= 75.0 and jitter_pct < 3.2:
        reasons.append("✨ **Expressive Cadence**: Natural emotional inflection without nervous tremor.")
    else:
        reasons.append("⚠️ **Micro-Pitch Instability**: Frequency flutter or tremor suggests stress, anxiety, or high emotional charge.")

    # Speech Coach Feedback
    coaching_tips = []
    if avg_energy < 0.022:
        coaching_tips.append("📢 **Diaphragm Amplification**: Inhale fully from your abdomen to project sound without vocal strain.")
    if pitch_std > 75.0:
        coaching_tips.append("🧘 **Tempo Regulation**: Slow your speaking cadence slightly to stabilize pitch modulation.")
    if not coaching_tips:
        coaching_tips.append("🏆 **Exceptional Delivery**: Masterful equilibrium of acoustic projection, pitch control, and cadence.")

    # Distress Detection
    is_distressed = (pitch_std > 60.0 and avg_energy < 0.028) or (jitter_pct > 2.2 and avg_energy < 0.025) or (audio_emotions.get("fear", 0) > 50.0)

    return {
        "audio_emotions": audio_emotions,
        "confidence_score": confidence_score,
        "reasons": reasons,
        "coaching_tips": coaching_tips,
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
        "valence": int(audio_emotions.get("joy", 0) - audio_emotions.get("sadness", 0) - audio_emotions.get("anger", 0) * 0.5 - audio_emotions.get("fear", 0) * 0.5),
        "waveform": y,
        "sample_rate": sr,
        "pitch_timeline": pitch_timeline
    }

def fuse_emotions(audio_result, text_result):
    EMOTION_KEYS = ["joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral"]

    # 1. High-Precision Full-Distribution Text Probabilities
    if "raw_emotion_scores" in text_result:
        text_probs = text_result["raw_emotion_scores"]
    else:
        text_probs = {}
        for s in text_result.get("sentences", []):
            key = s.get("top_emotion_key", "neutral")
            text_probs[key] = text_probs.get(key, 0) + s.get("confidence", 0)
        total_text = sum(text_probs.values()) or 1.0
        text_probs = {k: (v / total_text) * 100 for k, v in text_probs.items()}

    audio_probs = audio_result.get("audio_emotions", {})

    # Quality Gate Check in Multimodal Fusion
    if audio_result.get("status") == "insufficient_audio":
        if text_probs and max(text_probs.values(), default=0) > 20:
            fused = dict(text_probs)
        else:
            return {
                "top_emotion": "Insufficient Speech",
                "top_emotion_key": "neutral",
                "top_emotion_score": 0,
                "fused_distribution": {"neutral": 100.0},
                "icon": "⚠️",
                "color": "#a1a1aa",
                "glow": "rgba(161, 161, 170, 0.25)",
                "circumplex_quadrant": "Neutral / Unassessed",
                "circumplex_desc": "Insufficient audio or text signal to assess emotional state.",
                "valence": 0.0,
                "arousal": 0.0,
                "disagreement": False,
                "disagreement_note": "Audio was too short or quiet to extract speech features."
            }

    # 2. Dynamic Multimodal Fusion Weighting (Affective Prosody First)
    # Human affect is conveyed primarily through vocal prosody (pitch, loudness, pace, timbre).
    # When text is conversational/neutral (reading lists or plain sentences), voice acoustics dominate (80% audio, 20% text)
    # so that neutral word choice does not suppress the speaker's true voice tone.
    # When text has explicit emotional vocabulary (non-neutral > 25%), both modalities reinforce each other (55% audio, 45% text).
    top_text_key = max(text_probs, key=text_probs.get) if text_probs else "neutral"
    text_has_strong_emotion = (top_text_key != "neutral") and (text_probs.get(top_text_key, 0) > 25.0)

    if text_has_strong_emotion:
        w_audio, w_text = 0.55, 0.45
    else:
        w_audio, w_text = 0.80, 0.20

    fused = {}
    for key in EMOTION_KEYS:
        t = text_probs.get(key, 0.0)
        a = audio_probs.get(key, 0.0)
        fused[key] = round(a * w_audio + t * w_text, 1)

    total_fused = sum(fused.values()) or 1.0
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

    # 2D Circumplex Model Coordinates (Russell 1980 - Normalized -1.0 to +1.0)
    v_raw = (fused.get("joy", 0) * 1.0 + fused.get("surprise", 0) * 0.35) - (
        fused.get("anger", 0) * 0.90 + fused.get("sadness", 0) * 0.85 + fused.get("fear", 0) * 0.70 + fused.get("disgust", 0) * 0.60
    )
    valence = round(max(-1.0, min(1.0, v_raw / 100.0)), 2)

    a_raw = (fused.get("anger", 0) * 0.90 + fused.get("fear", 0) * 0.80 + fused.get("surprise", 0) * 0.75 + fused.get("joy", 0) * 0.60) - (
        fused.get("sadness", 0) * 0.70 + fused.get("neutral", 0) * 0.50 + fused.get("disgust", 0) * 0.20
    )
    arousal = round(max(-1.0, min(1.0, a_raw / 100.0)), 2)

    if valence >= 0 and arousal >= 0:
        circumplex_quadrant = "Q1: Exuberant / Enthusiastic"
        circumplex_desc = "High Energy & Positive Sentiment (Engaging, passionate, uplifting delivery)"
    elif valence < 0 and arousal >= 0:
        circumplex_quadrant = "Q2: Tense / Agitated"
        circumplex_desc = "High Energy & Negative Sentiment (Elevated urgency, frustration, or vocal tension)"
    elif valence < 0 and arousal < 0:
        circumplex_quadrant = "Q3: Subdued / Melancholic"
        circumplex_desc = "Low Energy & Negative Sentiment (Fatigue, emotional low, or withdrawal markers)"
    else:
        circumplex_quadrant = "Q4: Serene / Composed"
        circumplex_desc = "Low Energy & Positive Sentiment (Grounding presence, calm reflection, poised assurance)"

    return {
        "top_emotion": f"{icon} {label}",
        "top_key": top_fused_key,
        "top_color": color,
        "top_glow": glow,
        "top_emotion_score": fused[top_fused_key],
        "fused_emotions": fused,
        "valence": valence,
        "arousal": arousal,
        "circumplex_quadrant": circumplex_quadrant,
        "circumplex_desc": circumplex_desc
    }

def generate_executive_coaching(audio_res, text_res, fused):
    """
    Synthesizes multi-modal audio biometrics and sentiment into an
    Executive Communication & Vocal Wellness Report.
    """
    ac = audio_res.get("acoustics", {})
    top_key = fused.get("top_key", "neutral")
    valence = fused.get("valence", 0.0)
    arousal = fused.get("arousal", 0.0)
    confidence = audio_res.get("confidence_score", 75)

    # Primary Communication Archetype
    if valence >= 0.2 and arousal >= 0.2:
        archetype = "Inspirational Motivator"
        primary_insight = "Your delivery radiates energetic positivity and active engagement, naturally commanding audience enthusiasm."
    elif valence >= 0.05 and arousal < 0.1:
        archetype = "Empathetic Diplomat"
        primary_insight = "You project reassuring calmness, poise, and steady composure ideal for executive consultation and consensus."
    elif valence < -0.1 and arousal >= 0.2:
        archetype = "Urgent / High-Stakes Advocate"
        primary_insight = "Your voice carries sharp urgency and acoustic arousal; deliberate grounding pauses will enhance trust and authority."
    elif valence < -0.1 and arousal < 0.0:
        archetype = "Reflective / Subdued Thinker"
        primary_insight = "Your vocal tone is introspective and quiet; adding diaphragmatic projection will amplify your stage presence."
    else:
        archetype = "Analytical Synthesizer"
        primary_insight = "Balanced, objective vocal delivery with neutral modulation suitable for technical briefings and board reviews."

    # 3 Targeted Actionable Protocols
    protocols = []

    # Pillar 1: Cadence & Breath
    pitch_stab = ac.get("pitch_stability", "Moderate")
    if "Unsteady" in pitch_stab or audio_res.get("is_distressed"):
        protocols.append({
            "title": "🌬️ Diaphragmatic Breath Anchor",
            "action": "Detected subtle micro-tremor in vocal fold oscillation. Take a deep 3-second belly breath before starting sentences to anchor your fundamental frequency."
        })
    else:
        protocols.append({
            "title": "⏱️ Strategic Micro-Pausing",
            "action": "Maintain your steady breath rhythm. Insert intentional 0.8-second micro-pauses before key takeaways to command room attention and eliminate perceived rush."
        })

    # Pillar 2: Vocal Timbre & Inflection
    if ac.get("tone") == "Bright & Crisp":
        protocols.append({
            "title": "🎻 Chest Resonance Modulation",
            "action": "Your voice has high acoustic clarity. Slightly lowering your larynx into chest resonance will add warmth, depth, and gravitas to persuasive arguments."
        })
    else:
        protocols.append({
            "title": "✨ Articulatory Energy & Crisp Plosives",
            "action": "Warm vocal resonance detected. Brighten sentence conclusions by crisply finishing consonant stops (t, d, k) to prevent cadence trailing off."
        })

    # Pillar 3: Emotional Resonance & Audience Impact
    if top_key in ["fear", "anxious"]:
        protocols.append({
            "title": "🛡️ Nervous Tension Release & Grounding",
            "action": "Acoustic micro-tremor or vocal flutter detected. Place both feet firmly on the floor and lengthen your exhales to anchor vocal fold stability."
        })
    elif top_key in ["anger", "disgust"]:
        protocols.append({
            "title": "🧘 Emotional De-escalation Cadence",
            "action": "Negative acoustic tension detected. Soften your vocal attack on opening consonants to sound approachable, constructive, and firmly in control."
        })
    elif top_key == "sadness":
        protocols.append({
            "title": "⚡ Vocal Energy & Diaphragm Lift",
            "action": "Subdued monotone delivery detected. Inhale deeply from your abdomen and intentionally inflect upward on key nouns to revitalize vocal engagement."
        })
    elif top_key == "joy":
        protocols.append({
            "title": "🌟 Authentic Alignment Protocol",
            "action": "High positive resonance detected! Channel this natural warmth directly into your call-to-action for maximum audience buy-in."
        })
    else:
        protocols.append({
            "title": "🎯 Dynamic Pitch Variance Expansion",
            "action": "Vocal pitch is steady. Inflect upward by half an octave on rhetorical questions and drop pitch downward on concluding statements to avoid monotone drift."
        })

    strain_label = "Optimal / Low" if not audio_res.get("is_distressed") else "Elevated (Rest Advised)"
    wellness = {
        "archetype": archetype,
        "primary_insight": primary_insight,
        "protocols": protocols,
        "vocal_strain": strain_label,
        "emotional_equilibrium": f"{min(98, max(55, int(100 - abs(valence) * 20 - abs(arousal) * 20)))}%",
        "presence_score": f"{min(99, max(60, int(confidence * 0.7 + (1.0 - abs(arousal - 0.2)) * 30)))}%"
    }
    return wellness
