import streamlit as st
import tempfile
import os
import json
import importlib
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
import streamlit.components.v1 as components

# Custom-built audio recorder component with full-circle click hit-area & centered mic
build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recorder_component")
_custom_audio_recorder = components.declare_component("audio_recorder", path=build_dir)

def audio_recorder(
    text: str = "",
    energy_threshold: float = 0.01,
    pause_threshold: float = 0.8,
    neutral_color: str = "#ffffff",
    recording_color: str = "#f43f5e",
    icon_name: str = "microphone",
    icon_size: str = "2x",
    sample_rate = None,
    key = None,
):
    data = _custom_audio_recorder(
        text=text,
        start_threshold=energy_threshold,
        end_threshold=energy_threshold,
        pause_threshold=pause_threshold,
        neutral_color=neutral_color,
        recording_color=recording_color,
        icon_name=icon_name,
        icon_size=icon_size,
        sample_rate=sample_rate,
        key=key,
        default=None,
    )
    return bytes(json.loads(data)) if data else None

from analyzer import analyze_audio, fuse_emotions
from transcriber import transcribe_and_analyze

# 1. Page Configuration
st.set_page_config(
    page_title="EMOSENSE — Voice Emotion & Mood Analyzer",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Premium Minimalist Dark UI with Electric-Purple Accents & Animated Waveforms
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global Dark Canvas & Responsive Layout */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #07070a !important;
        color: #f1f5f9 !important;
        overflow-x: hidden;
    }

    /* Ambient Dynamic Glowing Background with Animated Waveforms & Subtle Particles */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background: 
            radial-gradient(circle at 50% 12%, rgba(139, 92, 246, 0.12) 0%, transparent 55%),
            radial-gradient(circle at 15% 30%, rgba(99, 102, 241, 0.08) 0%, transparent 40%),
            radial-gradient(circle at 85% 35%, rgba(168, 85, 247, 0.08) 0%, transparent 45%),
            radial-gradient(circle at 50% 90%, rgba(124, 58, 237, 0.06) 0%, transparent 50%),
            #07070a;
        pointer-events: none;
        z-index: 0;
    }

    /* Animated Flowing Audio-Wave Lines (SVG Vector Simulation) */
    .ambient-wave-canvas {
        position: fixed;
        top: 60px;
        left: 0;
        width: 100vw;
        height: 260px;
        pointer-events: none;
        z-index: 1;
        opacity: 0.55;
        overflow: hidden;
    }
    .wave-line-1 {
        position: absolute;
        width: 200vw;
        height: 100%;
        left: -50vw;
        background: radial-gradient(ellipse at center, rgba(168, 85, 247, 0.35) 0%, rgba(139, 92, 246, 0) 70%);
        mask-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 120" preserveAspectRatio="none"><path d="M0,0 C150,90 350,-40 500,45 C650,130 900,10 1200,50 L1200,120 L0,120 Z" fill="white"/></svg>');
        -webkit-mask-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 120" preserveAspectRatio="none"><path d="M0,0 C150,90 350,-40 500,45 C650,130 900,10 1200,50 L1200,120 L0,120 Z" fill="white"/></svg>');
        mask-size: 100% 100%;
        -webkit-mask-size: 100% 100%;
        animation: waveFlow 18s ease-in-out infinite alternate;
    }
    .wave-line-2 {
        position: absolute;
        width: 200vw;
        height: 100%;
        left: -40vw;
        background: radial-gradient(ellipse at center, rgba(192, 132, 252, 0.25) 0%, rgba(99, 102, 241, 0) 65%);
        mask-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 120" preserveAspectRatio="none"><path d="M0,50 C200,10 450,120 700,30 C950,-50 1100,80 1200,20 L1200,120 L0,120 Z" fill="white"/></svg>');
        -webkit-mask-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 120" preserveAspectRatio="none"><path d="M0,50 C200,10 450,120 700,30 C950,-50 1100,80 1200,20 L1200,120 L0,120 Z" fill="white"/></svg>');
        mask-size: 100% 100%;
        -webkit-mask-size: 100% 100%;
        animation: waveFlowReverse 22s ease-in-out infinite alternate;
    }

    @keyframes waveFlow {
        0% { transform: translateX(0) scaleY(0.9); }
        50% { transform: translateX(-6%) scaleY(1.1); }
        100% { transform: translateX(4%) scaleY(0.95); }
    }
    @keyframes waveFlowReverse {
        0% { transform: translateX(0) scaleY(1.05); }
        50% { transform: translateX(6%) scaleY(0.88); }
        100% { transform: translateX(-5%) scaleY(1.12); }
    }

    /* Subtle Floating Light Particles */
    .particle-dust {
        position: fixed;
        width: 3px;
        height: 3px;
        border-radius: 50%;
        background: rgba(216, 180, 254, 0.6);
        box-shadow: 0 0 8px rgba(192, 132, 252, 0.9);
        pointer-events: none;
        z-index: 1;
        animation: floatParticle 14s infinite linear;
    }
    @keyframes floatParticle {
        0% { transform: translateY(100vh) translateX(0); opacity: 0; }
        15% { opacity: 0.8; }
        85% { opacity: 0.8; }
        100% { transform: translateY(-10vh) translateX(40px); opacity: 0; }
    }

    /* Centered Container */
    .block-container {
        max-width: 960px !important;
        padding-top: 1.5rem !important;
        padding-bottom: 3.5rem !important;
        margin: 0 auto !important;
        position: relative;
        z-index: 10;
    }

    /* Centered EMOSENSE Brand Header matching Pixel-Perfect Reference */
    .brand-hero {
        text-align: center;
        padding: 10px 0 20px 0;
        position: relative;
    }
    .mic-header-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 72px;
        height: 72px;
        margin-bottom: 12px;
        filter: drop-shadow(0 0 16px rgba(192, 132, 252, 0.45));
    }
    .brand-title {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 2.85rem;
        font-weight: 800;
        letter-spacing: 0.38em;
        text-indent: 0.38em;
        text-transform: uppercase;
        margin: 0;
        line-height: 1.1;
        /* Metallic Chrome Silver Gradient Text */
        background: linear-gradient(180deg, #ffffff 0%, #cbd5e1 45%, #94a3b8 70%, #f8fafc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        filter: drop-shadow(0 0 25px rgba(255, 255, 255, 0.25));
    }
    .brand-tagline {
        font-family: 'Space Grotesk', sans-serif !important;
        color: #94a3b8 !important;
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 0.35em;
        text-indent: 0.35em;
        text-transform: uppercase;
        margin-top: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-direction: column;
    }
    .brand-underline {
        width: 48px;
        height: 1.5px;
        background: linear-gradient(90deg, transparent, #c084fc, transparent);
        margin-top: 8px;
        border-radius: 2px;
    }

    /* ==========================================================
       UNIFIED MIC CARD  (label + recorder button = ONE element)
       The card is a keyed st.container -> .st-key-mic_card
       ========================================================== */
    .st-key-mic_card {
        background: rgba(14, 13, 20, 0.82);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        box-sizing: border-box;
        height: 96px;
        min-height: 96px;
        padding: 0 18px 0 24px;
        display: flex !important;
        flex-direction: column;
        justify-content: center;
        gap: 0 !important;
        box-shadow: 0 10px 32px 0 rgba(0, 0, 0, 0.55), inset 0 1px 0 rgba(255, 255, 255, 0.08);
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        overflow: hidden;
    }
    .st-key-mic_card:hover {
        border-color: rgba(168, 85, 247, 0.35);
        box-shadow: 0 12px 40px 0 rgba(147, 51, 234, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.15);
        transform: translateY(-2px);
    }
    /* Keep the card contents vertically centred and on one row */
    .st-key-mic_card > div[data-testid="stVerticalBlock"] {
        height: 100% !important;
        min-height: 96px !important;
        justify-content: center !important;
    }
    .st-key-mic_card div[data-testid="stHorizontalBlock"] {
        flex-wrap: nowrap !important;
        align-items: center !important;
        justify-content: space-between !important;
        gap: 12px !important;
        width: 100% !important;
        min-height: 94px !important;
        margin: 0 !important;
    }
    .st-key-mic_card div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child,
    .st-key-mic_card div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child {
        flex: 1 1 0% !important;
        min-width: 0 !important;
    }
    /* Recorder button: fixed circular hit-area with its iframe centered exactly */
    .st-key-mic_card div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:last-child,
    .st-key-mic_card div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child {
        flex: 0 0 62px !important;
        width: 62px !important;
        min-width: 62px !important;
        max-width: 62px !important;
        height: 62px !important;
        min-height: 62px !important;
        max-height: 62px !important;
        box-sizing: border-box !important;
        padding: 0 !important;
        margin: 0 !important;
        border-radius: 50% !important;
        background: rgba(168, 85, 247, 0.045) !important;
        border: 1.5px solid rgba(168, 85, 247, 0.58) !important;
        box-shadow: 0 0 13px rgba(147, 51, 234, 0.20), inset 0 0 8px rgba(192, 132, 252, 0.08) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        align-self: center !important;
        overflow: hidden !important;
        line-height: 0 !important;
        position: relative !important;
    }
    .st-key-mic_card div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:last-child:hover,
    .st-key-mic_card div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child:hover {
        border-color: rgba(216, 180, 254, 0.9) !important;
        box-shadow: 0 0 22px rgba(168, 85, 247, 0.38), inset 0 0 10px rgba(192, 132, 252, 0.12) !important;
    }
    /* Streamlit adds wrapper divs around custom components; center every wrapper */
    .st-key-mic_card div[data-testid="stHorizontalBlock"] > div:last-child > div,
    .st-key-mic_card div[data-testid="stHorizontalBlock"] > div:last-child [data-testid="stCustomComponentV1"] {
        width: 100% !important;
        height: 100% !important;
        min-width: 0 !important;
        min-height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-sizing: border-box !important;
    }
    .st-key-mic_card iframe {
        display: block !important;
        width: 100% !important;
        height: 100% !important;
        min-width: 62px !important;
        max-width: 62px !important;
        min-height: 62px !important;
        max-height: 62px !important;
        margin: 0 !important;
        padding: 0 !important;
        border: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
        flex: 1 1 100% !important;
        cursor: pointer !important;
    }

    /* Upload panel: keep the upload button and file-limit text fully visible */
    .st-key-upload_card,
    .st-key-upload_card > div[data-testid="stVerticalBlock"] {
        width: 100% !important;
        min-height: 96px !important;
        height: 96px !important;
        justify-content: center !important;
        margin: 0 !important;
        padding: 0 !important;
        transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    .st-key-upload_card:hover {
        transform: translateY(-2px) !important;
    }
    .st-key-upload_card [data-testid="stFileUploader"] {
        width: 100% !important;
        min-width: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    [data-testid="stFileUploader"] {
        background-color: transparent !important;
        width: 100% !important;
        min-width: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        width: 100% !important;
        min-width: 0 !important;
        height: 96px !important;
        min-height: 96px !important;
        box-sizing: border-box !important;
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        justify-content: flex-start !important;
        gap: 14px !important;
        padding: 12px 18px !important;
        overflow: hidden !important;
        background: rgba(14, 13, 20, 0.82) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 20px !important;
        box-shadow: 0 10px 32px 0 rgba(0, 0, 0, 0.55), inset 0 1px 0 rgba(255, 255, 255, 0.08) !important;
        transition: border-color 0.25s ease, box-shadow 0.25s ease, transform 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: rgba(168, 85, 247, 0.35) !important;
        box-shadow: 0 12px 32px rgba(147, 51, 234, 0.22), inset 0 1px 0 rgba(255, 255, 255, 0.12) !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] {
        order: 2 !important;
        flex: 1 1 auto !important;
        width: auto !important;
        min-width: 0 !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: flex-start !important;
        justify-content: center !important;
        gap: 2px !important;
        overflow: hidden !important;
        line-height: 1.25 !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] > div:first-child {
        color: #d7d4df !important;
        font-weight: 600 !important;
        font-size: 0.78rem !important;
        line-height: 1.25 !important;
        white-space: normal !important;
        overflow-wrap: anywhere !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] > small {
        display: block !important;
        max-width: 100% !important;
        min-width: 0 !important;
        color: #a5a0b0 !important;
        font-size: 0.68rem !important;
        line-height: 1.3 !important;
        white-space: normal !important;
        overflow-wrap: anywhere !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        order: 1 !important;
        flex: 0 0 auto !important;
        white-space: nowrap !important;
        background: rgba(22, 20, 30, 0.95) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        padding: 9px 15px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 6px !important;
        margin: 0 !important;
        box-shadow: 0 4px 14px rgba(0,0,0,0.28) !important;
    }
    [data-testid="stFileUploaderDropzone"] button:hover {
        background: rgba(147, 51, 234, 0.20) !important;
        border-color: rgba(192, 132, 252, 0.55) !important;
    }

    @media (max-width: 700px) {
        [data-testid="stFileUploaderDropzone"] {
            gap: 9px !important;
            padding: 10px 12px !important;
        }
        [data-testid="stFileUploaderDropzone"] button {
            padding: 8px 10px !important;
            font-size: 0.74rem !important;
        }
        [data-testid="stFileUploaderDropzoneInstructions"] > small {
            font-size: 0.62rem !important;
        }
    }

    /* Sleek Dark Oceanic ANALYZE MY VOICE Button from Reference Image */
    .stButton>button {
        width: 100% !important;
        background: radial-gradient(ellipse at center, rgba(30, 26, 42, 0.95) 0%, rgba(12, 12, 18, 0.98) 100%) !important;
        color: #f1f5f9 !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 0.88rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.22em !important;
        text-transform: uppercase !important;
        border-radius: 9999px !important;
        border: 1px solid rgba(168, 85, 247, 0.35) !important;
        padding: 16px 28px !important;
        box-shadow: 0 0 22px rgba(147, 51, 234, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
        cursor: pointer !important;
    }
    .stButton>button:hover {
        background: radial-gradient(ellipse at center, rgba(46, 38, 68, 0.98) 0%, rgba(18, 16, 26, 1) 100%) !important;
        border-color: rgba(192, 132, 252, 0.75) !important;
        box-shadow: 0 0 35px rgba(168, 85, 247, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.25) !important;
        transform: translateY(-2px) scale(1.008);
        color: #ffffff !important;
    }
    .stButton>button:active {
        transform: translateY(1px) scale(0.995);
    }

    /* Result Dashboard Hero Window */
    .result-hero {
        background: linear-gradient(180deg, rgba(24, 20, 36, 0.85) 0%, rgba(13, 13, 18, 0.9) 100%);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(168, 85, 247, 0.35);
        border-radius: 20px;
        padding: 26px 30px;
        box-shadow: 0 16px 45px rgba(0, 0, 0, 0.6), 0 0 40px rgba(147, 51, 234, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        margin: 24px 0 20px 0;
        transition: all 0.3s ease;
    }
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 14px;
        border-radius: 9999px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #d8b4fe !important;
        background: rgba(147, 51, 234, 0.15);
        border: 1px solid rgba(168, 85, 247, 0.35);
        box-shadow: 0 0 12px rgba(147, 51, 234, 0.2);
    }
    .verdict-title {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 2.8rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #ffffff !important;
        margin: 12px 0 18px 0;
        display: flex;
        align-items: center;
        gap: 14px;
        text-shadow: 0 0 25px rgba(192, 132, 252, 0.35);
    }

    /* Metric Grid */
    .metric-grid-lux {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin-top: 18px;
    }
    .metric-card-lux {
        background: rgba(15, 15, 22, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 14px;
        padding: 15px 18px;
        transition: all 0.25s ease;
    }
    .metric-card-lux:hover {
        border-color: rgba(168, 85, 247, 0.4);
        background: rgba(22, 18, 32, 0.85);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(147, 51, 234, 0.15);
    }
    .metric-lux-label {
        color: #94a3b8 !important;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .metric-lux-value {
        color: #ffffff !important;
        font-size: 1.45rem;
        font-weight: 700;
        margin-top: 4px;
        font-family: 'Space Grotesk', sans-serif;
    }
    .metric-lux-sub {
        color: #cbd5e1 !important;
        font-size: 0.75rem;
        margin-top: 2px;
    }

    /* Streamlit Tabs with Purple Violet Glow */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(14, 14, 20, 0.85);
        backdrop-filter: blur(12px);
        padding: 6px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        color: #94a3b8 !important;
        padding: 9px 18px;
        font-weight: 600;
        font-size: 0.86rem;
        transition: all 0.25s ease;
        border: 1px solid transparent;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #e2e8f0 !important;
        background: rgba(255, 255, 255, 0.03);
    }
    .stTabs [aria-selected="true"] {
        background: rgba(147, 51, 234, 0.18) !important;
        border: 1px solid rgba(168, 85, 247, 0.45) !important;
        color: #ffffff !important;
        box-shadow: 0 0 16px rgba(147, 51, 234, 0.25) !important;
    }

    /* Smooth Progress Bar with Purple Glow */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #7c3aed 0%, #a855f7 100%) !important;
        box-shadow: 0 0 12px rgba(168, 85, 247, 0.5) !important;
        border-radius: 9999px !important;
    }
    .stProgress > div > div > div {
        background-color: rgba(255, 255, 255, 0.06) !important;
        border-radius: 9999px !important;
    }

    /* Emotion Progress Card & Custom Spectrum Styling */
    .emotion-row {
        background: rgba(18, 17, 26, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 10px 16px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 16px;
        transition: all 0.2s ease;
    }
    .emotion-row:hover {
        background: rgba(26, 22, 38, 0.9);
        border-color: rgba(168, 85, 247, 0.35);
        box-shadow: 0 4px 18px rgba(147, 51, 234, 0.15);
    }
    .emotion-label {
        width: 140px;
        font-weight: 600;
        font-size: 0.88rem;
        color: #f1f5f9;
        display: flex;
        align-items: center;
        gap: 8px;
        flex-shrink: 0;
    }
    .emotion-bar-track {
        flex: 1;
        height: 10px;
        background: rgba(255, 255, 255, 0.08);
        border-radius: 9999px;
        overflow: hidden;
        position: relative;
    }
    .emotion-bar-fill {
        height: 100%;
        border-radius: 9999px;
        transition: width 0.8s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .emotion-value {
        width: 55px;
        text-align: right;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.84rem;
        font-weight: 600;
        color: #e2e8f0;
        flex-shrink: 0;
    }

    /* Footer Slogan */
    .footer-slogan {
        text-align: center;
        margin-top: 32px;
        font-family: 'Space Grotesk', sans-serif;
        color: #71717a;
        font-size: 0.68rem;
        letter-spacing: 0.35em;
        text-indent: 0.35em;
        text-transform: uppercase;
    }
</style>

<!-- Ambient Dynamic Elements (Waveforms & Particles) -->
<div class="ambient-wave-canvas">
    <div class="wave-line-1"></div>
    <div class="wave-line-2"></div>
</div>
<div class="particle-dust" style="top: 25%; left: 18%; animation-delay: 0s;"></div>
<div class="particle-dust" style="top: 45%; left: 82%; animation-delay: 3s;"></div>
<div class="particle-dust" style="top: 70%; left: 35%; animation-delay: 6s;"></div>
<div class="particle-dust" style="top: 85%; left: 65%; animation-delay: 9s;"></div>
""", unsafe_allow_html=True)

# 3. Persistent Audio State & History
if "active_audio" not in st.session_state:
    st.session_state["active_audio"] = None
if "audio_label" not in st.session_state:
    st.session_state["audio_label"] = ""
if "rec_history" not in st.session_state:
    # List of dicts: {"id": "rec_1", "name": "rec_1", "timestamp": "...", "bytes": b"...", "type": "mic|file"}
    st.session_state["rec_history"] = []
if "rec_counter" not in st.session_state:
    st.session_state["rec_counter"] = 0
if "last_processed_audio_hash" not in st.session_state:
    st.session_state["last_processed_audio_hash"] = None

def save_to_history(audio_bytes, source_label="mic"):
    """Saves audio to history. Returns True only if it was a NEW recording/file."""
    import hashlib
    import datetime
    if not audio_bytes:
        return False
    
    # Calculate simple hash to avoid duplicating identical continuous frames
    h = hashlib.md5(audio_bytes[:5000] if len(audio_bytes) > 5000 else audio_bytes).hexdigest()
    if st.session_state.get("last_processed_audio_hash") == h:
        return False
    
    st.session_state["last_processed_audio_hash"] = h
    st.session_state["rec_counter"] += 1
    rec_num = st.session_state["rec_counter"]
    rec_name = f"rec_{rec_num}"
    
    now_str = datetime.datetime.now().strftime("%I:%M:%S %p")
    entry = {
        "id": f"rec_{rec_num}",
        "name": rec_name,
        "label": f"{rec_name} ({now_str})",
        "time": now_str,
        "source": source_label,
        "bytes": audio_bytes,
        "size_kb": round(len(audio_bytes) / 1024, 1)
    }
    # Prepend newest, retain maximum 10 recordings
    st.session_state["rec_history"].insert(0, entry)
    if len(st.session_state["rec_history"]) > 10:
        st.session_state["rec_history"] = st.session_state["rec_history"][:10]
    return True

# 4. Bulletproof PDF Report Generator (Safe Margins, No Horizontal Space Exceptions)
def generate_pdf_dossier(audio_res, text_res, fused):
    try:
        def clean_ascii(val):
            if val is None:
                return ""
            s = str(val).replace("**", "").replace("*", "").replace("`", "")
            return "".join(c for c in s if 32 <= ord(c) <= 126).strip()

        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        epw = pdf.epw

        # Title
        pdf.set_font("helvetica", "B", 16)
        pdf.set_text_color(20, 20, 20)
        pdf.cell(w=epw, h=10, text="EMOSENSE - VOICE ANALYSIS REPORT", align="C", new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("helvetica", "", 8)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(w=epw, h=5, text="VOICE EMOTION & ACOUSTIC ASSESSMENT", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)

        # Section 1: Executive Summary
        pdf.set_font("helvetica", "B", 10)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(w=epw, h=6, text="1. PRIMARY VOICE SUMMARY", new_x="LMARGIN", new_y="NEXT")
        pdf.set_x(pdf.l_margin)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + epw, pdf.get_y())
        pdf.ln(3)

        pdf.set_font("helvetica", "", 9)
        demo = audio_res.get("demographics", {})
        ac = audio_res.get("acoustics", {})

        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(w=epw, h=5, text=f"Dominant Emotion: {clean_ascii(fused.get('top_emotion'))} (Confidence: {fused.get('top_emotion_score', 0)}%)")
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(w=epw, h=5, text=f"Speaking Confidence: {audio_res.get('confidence_score', 0)}% (Steadiness Score)")
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(w=epw, h=5, text=f"Demographics: {clean_ascii(demo.get('gender'))} (Confidence: {clean_ascii(demo.get('gender_confidence'))}) | Age Bracket: {clean_ascii(demo.get('age_group'))}")
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(w=epw, h=5, text=f"Mood Level (Valence): {audio_res.get('valence', 0):+d} (-100 to +100 scale)")
        pdf.ln(4)

        # Section 2: Acoustic Diagnostics
        pdf.set_font("helvetica", "B", 10)
        pdf.cell(w=epw, h=6, text="2. ACOUSTIC BIOMETRICS", new_x="LMARGIN", new_y="NEXT")
        pdf.set_x(pdf.l_margin)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + epw, pdf.get_y())
        pdf.ln(3)

        pdf.set_font("helvetica", "", 8.5)
        pdf.set_x(pdf.l_margin)
        pitch_display = clean_ascii(ac.get('median_pitch', ac.get('avg_pitch', 'N/A')))
        pdf.multi_cell(w=epw, h=5, text=f"F0 Median Pitch: {pitch_display} | Avg Pitch: {clean_ascii(ac.get('avg_pitch', 'N/A'))}")
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(w=epw, h=5, text=f"Pitch Stability: {clean_ascii(ac.get('pitch_stability', 'N/A'))} | Vocal Micro-Jitter: {clean_ascii(ac.get('jitter', 'N/A'))}")
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(w=epw, h=5, text=f"Spectral Centroid (Voice Brightness): {clean_ascii(ac.get('spectral_centroid', 'N/A'))} | Energy RMS: {clean_ascii(ac.get('energy', 'N/A'))}")
        pdf.ln(4)

        # Section 3: Transcription
        pdf.set_font("helvetica", "B", 10)
        pdf.cell(w=epw, h=6, text="3. SPEECH TRANSCRIPTION", new_x="LMARGIN", new_y="NEXT")
        pdf.set_x(pdf.l_margin)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + epw, pdf.get_y())
        pdf.ln(3)

        pdf.set_font("helvetica", "", 8.5)
        transcript = clean_ascii(text_res.get("transcript", "No transcript generated."))
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(w=epw, h=5, text=f"Spoken Text: \"{transcript}\"")
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(w=epw, h=5, text=f"Detected Language: {clean_ascii(text_res.get('language_detected', 'auto')).upper()} | Words Spoken: {text_res.get('word_count', 0)}")
        pdf.ln(4)

        # Section 4: Coaching Tips
        pdf.set_font("helvetica", "B", 10)
        pdf.cell(w=epw, h=6, text="4. VOICE TIPS & COACHING ADVICE", new_x="LMARGIN", new_y="NEXT")
        pdf.set_x(pdf.l_margin)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + epw, pdf.get_y())
        pdf.ln(3)

        pdf.set_font("helvetica", "", 8.5)
        for r in audio_res.get("reasons", []):
            clean_r = clean_ascii(r)
            if clean_r:
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(w=epw, h=5, text=f"- {clean_r}")

        for tip in audio_res.get("coaching_tips", []):
            clean_tip = clean_ascii(tip)
            if clean_tip:
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(w=epw, h=5, text=f"* Tip: {clean_tip}")

        return bytes(pdf.output())
    except Exception as e:
        fallback_pdf = FPDF()
        fallback_pdf.add_page()
        fallback_pdf.set_font("helvetica", size=10)
        fallback_pdf.cell(w=fallback_pdf.epw, h=8, text=f"EmoSense Voice Summary - Result: {fused.get('top_emotion', 'Complete')}", new_x="LMARGIN", new_y="NEXT")
        return bytes(fallback_pdf.output())

# =========================================================
# LAYOUT 1: CENTERED EMOSENSE LOGO & SUBTITLE MATCHING DESIGN
# =========================================================
st.markdown("""
<div class="brand-hero">
    <div class="mic-header-icon">
        <svg width="68" height="68" viewBox="0 0 80 80" fill="none" xmlns="http://www.w3.org/2000/svg">
            <!-- Center Glowing Microphone with Audio Waves -->
            <path d="M40 18C36.6863 18 34 20.6863 34 24V38C34 41.3137 36.6863 44 40 44C43.3137 44 46 41.3137 46 38V24C46 20.6863 43.3137 18 40 18Z" stroke="#f1f5f9" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M28 34V38C28 44.6274 33.3726 50 40 50C46.6274 50 52 44.6274 52 38V34" stroke="#f1f5f9" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M40 50V60M32 60H48" stroke="#f1f5f9" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
            <!-- Left Sound Wave Bars -->
            <line x1="22" y1="33" x2="22" y2="39" stroke="#c084fc" stroke-width="2" stroke-linecap="round" opacity="0.8"/>
            <line x1="16" y1="30" x2="16" y2="42" stroke="#d8b4fe" stroke-width="2" stroke-linecap="round"/>
            <line x1="10" y1="35" x2="10" y2="37" stroke="#a855f7" stroke-width="2" stroke-linecap="round" opacity="0.6"/>
            <!-- Right Sound Wave Bars -->
            <line x1="58" y1="33" x2="58" y2="39" stroke="#c084fc" stroke-width="2" stroke-linecap="round" opacity="0.8"/>
            <line x1="64" y1="30" x2="64" y2="42" stroke="#d8b4fe" stroke-width="2" stroke-linecap="round"/>
            <line x1="70" y1="35" x2="70" y2="37" stroke="#a855f7" stroke-width="2" stroke-linecap="round" opacity="0.6"/>
        </svg>
    </div>
    <div class="brand-title">EMOSENSE</div>
    <div class="brand-tagline">
        <span>VOICE EMOTION & MOOD ANALYZER</span>
        <div class="brand-underline"></div>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# LAYOUT 2: LANGUAGE SELECTION SECTION
# =========================================================
col_lang1, col_lang2 = st.columns([1.3, 1.7])
with col_lang1:
    st.markdown("""
    <div style="font-size:0.82rem; font-weight:600; color:#cbd5e1; margin-bottom:6px; display:flex; align-items:center; gap:6px;">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#a855f7" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="10"></circle>
            <line x1="2" x2="22" y1="12" y2="12"></line>
            <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path>
        </svg>
        Select Language
    </div>
    """, unsafe_allow_html=True)
    lang_choice = st.selectbox(
        "Select Language",
        options=["dual", "en", "ta"],
        format_func=lambda x: "🎙️ Dual Speech (Tamil + English)" if x == "dual" else ("🇬🇧 English Only" if x == "en" else "🇮🇳 Tamil Only (தமிழ்)"),
        label_visibility="collapsed"
    )

with col_lang2:
    mode_desc = (
        "Automatically detects code-switching (Tamil + English in same speech)."
        if lang_choice == "dual" else
        ("Dedicated English acoustic and semantic inference." if lang_choice == "en" else "Dedicated Tamil native transcription and tone analysis.")
    )
    st.markdown(f"""
    <div style="padding-top: 24px; color: #a1a1aa; font-size: 0.8rem; line-height: 1.4;">
        <span style="color:#e2e8f0; font-weight:600;">Mode:</span> {mode_desc}
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

# =========================================================
# LAYOUT 3: TWO INTEGRATED PANELS (MIC CARD & UPLOAD CARD)
# =========================================================
col_in1, col_in2 = st.columns(2, gap="large", vertical_alignment="center")

with col_in1:
    # One keyed container = ONE card (styled via .st-key-mic_card)
    # Label + recorder icon live inside it, so they read as a single element.
    with st.container(key="mic_card"):
        mic_text_col, mic_button_col = st.columns([1, 0.16], gap="small", vertical_alignment="center")
        with mic_text_col:
            st.markdown("""
            <div style="display: flex; align-items: center; gap: 14px;">
                <div style="color: #c084fc; display: flex; align-items: center; justify-content: center;">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"></path>
                        <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
                        <line x1="12" x2="12" y1="19" y2="22"></line>
                    </svg>  
                </div>
                <div style="display: flex; flex-direction: column; justify-content: center; min-height: 85px;">
                    <div style="font-size: 0.95rem; font-weight: 700; color: #ffffff; line-height: 1.2;">Speak into Microphone</div>
                    <div style="font-size: 0.74rem; color: #94a3b8; line-height: 1.2; margin-top: 4px;">Tap to Record (5–15s)</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with mic_button_col:
            rec = audio_recorder(
                text="",
                recording_color="#a78bfa",
                neutral_color="#d8d8df",
                icon_size="2x"
            )

    # Only react to a NEW recording, so "Load rec_x" from history isn't overwritten on rerun
    if rec and save_to_history(rec, source_label="Live Mic"):
        st.session_state["active_audio"] = rec
        st.session_state["audio_label"] = f"Live Mic ({st.session_state['rec_history'][0]['name']})"

with col_in2:
    with st.container(key="upload_card"):
        upl = st.file_uploader(
            "Upload an Audio File",
            type=["wav", "mp3", "m4a"],
            label_visibility="collapsed"
        )
    if upl:
        uploaded_bytes = upl.getvalue()
        if save_to_history(uploaded_bytes, source_label=f"Upload: {upl.name}"):
            st.session_state["active_audio"] = uploaded_bytes
            st.session_state["audio_label"] = f"File: {upl.name}"

# =========================================================
# RECORDING HISTORY DRAWER (PAST 10 RECORDINGS)
# =========================================================
if st.session_state["rec_history"]:
    with st.expander(f"🕒 Recent Recordings History ({len(st.session_state['rec_history'])} saved)", expanded=False):
        st.caption("Select any previous recording to replay or run analysis:")
        
        hist_cols = st.columns(min(len(st.session_state["rec_history"]), 5))
        for idx, item in enumerate(st.session_state["rec_history"]):
            c_target = hist_cols[idx % len(hist_cols)]
            with c_target:
                is_current = st.session_state.get("active_audio") == item["bytes"]
                btn_border = "border: 1px solid #c084fc;" if is_current else ""
                
                st.markdown(f"""
                <div style="background: rgba(18, 18, 24, 0.85); border-radius: 12px; padding: 10px 12px; margin-bottom: 8px; border: 1px solid rgba(255,255,255,0.08); {btn_border}">
                    <div style="font-size: 0.82rem; font-weight: 700; color: #ffffff;">{item['name']}</div>
                    <div style="font-size: 0.7rem; color: #a1a1aa; font-family: 'JetBrains Mono', monospace;">{item['time']}</div>
                    <div style="font-size: 0.68rem; color: #c084fc; margin-top: 2px;">{item['source'][:14]} &bull; {item['size_kb']}KB</div>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"Load {item['name']}", key=f"btn_load_{item['id']}_{idx}", use_container_width=True):
                    st.session_state["active_audio"] = item["bytes"]
                    st.session_state["audio_label"] = f"{item['name']} ({item['time']})"
                    st.rerun()

# =========================================================
# LAYOUT 4: AUDIO PLAYBACK COMPONENT
# =========================================================
if st.session_state.get("active_audio") is not None:
    st.markdown("<div style='margin-top: 18px;'></div>", unsafe_allow_html=True)
    st.audio(st.session_state["active_audio"], format="audio/wav")
    
    lbl = st.session_state.get("audio_label", "Audio Ready")
    st.markdown(f"""
    <div style="margin-top: 6px; font-size: 0.8rem; color: #a1a1aa; display: flex; align-items: center; gap: 8px;">
        <span style="color:#d8b4fe;">●</span>
        <span>Selected Audio:</span>
        <span style="color: #ffffff; background: rgba(255,255,255,0.08); padding: 2px 10px; border-radius: 6px; font-family: 'JetBrains Mono', monospace; font-size: 0.76rem;">{lbl}</span>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# LAYOUT 5: PROMINENT PURPLE GRADIENT 'ANALYZE MY VOICE' BUTTON
# =========================================================
st.markdown("<div style='margin-top: 22px;'></div>", unsafe_allow_html=True)

has_audio = st.session_state.get("active_audio") is not None
if not has_audio:
    st.markdown("""
    <div style="text-align: center; color: #71717a; font-size: 0.82rem; margin-bottom: 8px;">
        Record from your microphone or upload an audio file above to activate analysis.
    </div>
    """, unsafe_allow_html=True)

analyze_clicked = st.button(
    "  ANALYZE MY VOICE",
    disabled=not has_audio,
    use_container_width=True
)

st.markdown("""
<div class="footer-slogan">
    LISTEN &nbsp;&bull;&nbsp; UNDERSTAND &nbsp;&bull;&nbsp; IMPROVE
</div>
""", unsafe_allow_html=True)

# =========================================================
# LAYOUT 6: EMOTION ANALYSIS RESULTS DASHBOARD
# =========================================================
if analyze_clicked and has_audio:
    raw_audio_bytes = st.session_state.get("active_audio")
    if not raw_audio_bytes or len(raw_audio_bytes) < 4000:
        st.warning("⚠️ The recorded audio is empty or too brief (< 0.5s). Please record again and speak for at least 3–5 seconds.")
    else:
        with st.spinner("Analyzing your voice biometrics, emotion frequencies, and spoken words..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(raw_audio_bytes)
                tmp_path = tmp.name

        try:
            audio_res = analyze_audio(tmp_path)
            text_res = transcribe_and_analyze(tmp_path, language=lang_choice)
            fused = fuse_emotions(audio_res, text_res)
            demo = audio_res.get("demographics", {})
            ac = audio_res.get("acoustics", {})

            # 1. Main Focused Hero Window
            pitch_display = ac.get("median_pitch", ac.get("avg_pitch", "N/A"))
            valence_val = audio_res.get("valence", 0)

            st.markdown(f"""
            <div class="result-hero">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span class="badge-pill">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>
                        VOICE RESULT
                    </span>
                    <span style="font-size: 0.82rem; color: #c084fc; font-family: 'JetBrains Mono', monospace;">
                        F0 PITCH: {pitch_display}
                    </span>
                </div>
                <div class="verdict-title">
                    <span>{fused.get('top_emotion', 'Neutral')}</span>
                    <span style="font-size: 1.1rem; color: #d8b4fe; font-weight: 500; font-family: 'JetBrains Mono', monospace;">
                        ({fused.get('top_emotion_score', 0)}% sure)
                    </span>
                </div>
                <div class="metric-grid-lux">
                    <div class="metric-card-lux">
                        <div class="metric-lux-label">Speaking Confidence</div>
                        <div class="metric-lux-value">{audio_res.get('confidence_score', 0)}%</div>
                        <div class="metric-lux-sub">Voice Steadiness</div>
                    </div>
                    <div class="metric-card-lux">
                        <div class="metric-lux-label">Voice Tone</div>
                        <div class="metric-lux-value">{demo.get('gender', 'N/A')}</div>
                        <div class="metric-lux-sub">{demo.get('gender_confidence', '85%')} match</div>
                    </div>
                    <div class="metric-card-lux">
                        <div class="metric-lux-label">Age Guess</div>
                        <div class="metric-lux-value">{demo.get('age_group', 'Adult')}</div>
                        <div class="metric-lux-sub">Acoustic Resonance</div>
                    </div>
                    <div class="metric-card-lux">
                        <div class="metric-lux-label">Mood Level</div>
                        <div class="metric-lux-value">
                            {valence_val:+d}
                        </div>
                        <div class="metric-lux-sub">-100 (Sad) to +100 (Happy)</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if audio_res.get("is_distressed"):
                st.warning("⚠️ **Voice Tremor Alert**: High micro-tremor and low acoustic intensity detected, which may indicate elevated emotional stress or fatigue.")

            # 2. Detailed Exploration Tabs
            t_lab, t_speech, t_dist, t_coach, t_exp = st.tabs([
                "🎵 Sound Wave & Pitch",
                "💬 Spoken Words & Sentences",
                "📊 All Emotions Breakdown",
                "💡 Voice Tips & Coaching",
                "📥 Download Report"
            ])

            # TAB 1: ACOUSTIC LAB
            with t_lab:
                st.markdown("#### 🎵 Sound Wave & Pitch Graph")
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4.4), sharex=True, facecolor='#09090e')
                plt.subplots_adjust(hspace=0.28)

                y_sig = audio_res.get("waveform", np.zeros(100))
                sr = audio_res.get("sample_rate", 16000)
                times = np.linspace(0, len(y_sig) / sr, num=len(y_sig))

                # Ax1: Waveform with purple glow aesthetic
                ax1.set_facecolor('#0d0d14')
                ax1.plot(times, y_sig, color="#c084fc", alpha=0.9, linewidth=0.8)
                ax1.set_ylabel("Loudness", color="#a1a1aa", fontsize=8)
                ax1.set_title("Sound Wave (Loudness Over Time)", color="#ffffff", fontsize=10, fontweight="bold")
                ax1.grid(True, linestyle="--", alpha=0.1, color="#a855f7")
                ax1.tick_params(colors="#94a3b8", labelsize=8)

                # Ax2: Pitch tracking timeline
                pitch_line = audio_res.get("pitch_timeline", [])
                pitch_times = np.linspace(0, len(y_sig) / sr, num=len(pitch_line))

                ax2.set_facecolor('#0d0d14')
                ax2.plot(pitch_times, pitch_line, color="#e9d5ff", linewidth=1.6, label="Voice Pitch (Hz)")
                ax2.set_ylabel("Pitch (Hz)", color="#a1a1aa", fontsize=8)
                ax2.set_xlabel("Time (seconds)", color="#a1a1aa", fontsize=8)
                ax2.set_title("Voice Pitch Tracking (Higher = Higher Voice)", color="#ffffff", fontsize=10, fontweight="bold")
                ax2.grid(True, linestyle="--", alpha=0.1, color="#a855f7")
                ax2.tick_params(colors="#94a3b8", labelsize=8)
                ax2.legend(loc="upper right", facecolor="#181824", edgecolor="none", labelcolor="#e9d5ff", fontsize=8)

                fig.tight_layout()
                st.pyplot(fig)

                p1, p2, p3, p4 = st.columns(4)
                p1.metric("Voice Pitch", ac.get("median_pitch", ac.get("avg_pitch", "N/A")))
                p2.metric("Voice Steadiness", ac.get("pitch_stability", "N/A"))
                p3.metric("Voice Tremor", ac.get("jitter", "N/A"))
                p4.metric("Voice Tone", ac.get("tone", "N/A"))

            # TAB 2: TRANSCRIPTION & TIMELINE
            with t_speech:
                st.markdown("#### 🗣️ What You Said")
                st.info(f"\"{text_res.get('transcript', '')}\"")
                detected_lang_str = str(text_res.get('language_detected') or 'auto').upper()
                st.caption(f"Language Mode: `{lang_choice.upper()}` | Detected: `{detected_lang_str}` | Word Count: `{text_res.get('word_count', 0)} words`")

                if text_res.get("sentences"):
                    st.markdown("#### 💬 Sentence-by-Sentence Breakdown")
                    for i, seg in enumerate(text_res["sentences"]):
                        with st.expander(f"{seg.get('icon', '💬')} [{seg.get('start_time', 0)}s → {seg.get('end_time', 0)}s] \"{seg.get('text', '')[:65]}...\"", expanded=True):
                            st.write(f"**Spoken Words:** *\"{seg.get('text', '')}\"*")
                            st.write(f"**Emotion in this sentence:** **{seg.get('top_emotion', 'Neutral')}** (`{seg.get('confidence', 0)}%` sure)")

            # TAB 3: EMOTION SPECTRUM
            with t_dist:
                st.markdown("#### 🌌 Complete Emotion Breakdown")
                st.caption("Acoustic frequency & NLP semantic distribution across all primary emotions:")

                # Thematic chromatic palette mapped individually per emotion
                EMO_PALETTE = {
                    "joy":       ("linear-gradient(90deg, #ec4899 0%, #f43f5e 100%)", "box-shadow: 0 0 12px rgba(244, 63, 94, 0.45);", "😄"),
                    "sadness":   ("linear-gradient(90deg, #6366f1 0%, #3b82f6 100%)", "box-shadow: 0 0 12px rgba(59, 130, 246, 0.45);", "🌧️"),
                    "anger":     ("linear-gradient(90deg, #f97316 0%, #ef4444 100%)", "box-shadow: 0 0 12px rgba(239, 68, 68, 0.45);", "⚡"),
                    "fear":      ("linear-gradient(90deg, #8b5cf6 0%, #6366f1 100%)", "box-shadow: 0 0 12px rgba(139, 92, 246, 0.45);", "🌪️"),
                    "disgust":   ("linear-gradient(90deg, #10b981 0%, #059669 100%)", "box-shadow: 0 0 12px rgba(16, 185, 129, 0.45);", "🥀"),
                    "surprise":  ("linear-gradient(90deg, #f59e0b 0%, #eab308 100%)", "box-shadow: 0 0 12px rgba(245, 158, 11, 0.45);", "✨"),
                    "neutral":   ("linear-gradient(90deg, #a855f7 0%, #c084fc 100%)", "box-shadow: 0 0 12px rgba(168, 85, 247, 0.45);", "⚖️"),
                }

                st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
                
                fused_items = sorted(
                    fused.get("fused_emotions", {}).items(),
                    key=lambda x: x[1],
                    reverse=True
                )

                for emo_k, val in fused_items:
                    k_clean = emo_k.lower().strip()
                    bg_grad, glow_style, emo_icon = EMO_PALETTE.get(
                        k_clean,
                        ("linear-gradient(90deg, #9333ea 0%, #c084fc 100%)", "box-shadow: 0 0 10px rgba(192, 132, 252, 0.4);", "🔹")
                    )
                    pct = max(0.5, min(100.0, float(val)))
                    
                    st.markdown(f"""
                    <div class="emotion-row">
                        <div class="emotion-label">
                            <span>{emo_icon}</span>
                            <span>{emo_k.title()}</span>
                        </div>
                        <div class="emotion-bar-track">
                            <div class="emotion-bar-fill" style="width: {pct}%; background: {bg_grad}; {glow_style}"></div>
                        </div>
                        <div class="emotion-value">{val:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

            # TAB 4: VOCAL DIAGNOSTICS & COACH
            with t_coach:
                st.markdown("#### 💡 Helpful Voice Coaching")
                d_left, d_right = st.columns(2)
                with d_left:
                    st.markdown("##### 🔍 What We Noticed in Your Voice:")
                    for r in audio_res.get("reasons", []):
                        st.markdown(r)
                with d_right:
                    st.markdown("##### 🎯 Suggestions to Improve:")
                    for tip in audio_res.get("coaching_tips", []):
                        st.info(tip)

            # TAB 5: EXPORT DOSSIER (PDF & JSON)
            with t_exp:
                st.markdown("#### 📥 Download Your Voice Report")
                st.write("Save a copy of your analysis as a PDF or raw data file:")

                exp_col1, exp_col2 = st.columns(2)

                with exp_col1:
                    pdf_bytes = generate_pdf_dossier(audio_res, text_res, fused)
                    st.download_button(
                        label="📥 DOWNLOAD PDF REPORT",
                        data=pdf_bytes,
                        file_name="emosense_voice_report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

                with exp_col2:
                    dossier_dict = {
                        "project": "EmoSense Voice Analyzer",
                        "verdict": fused.get("top_emotion", "Neutral"),
                        "confidence_score": audio_res.get("confidence_score", 0),
                        "demographics": demo,
                        "acoustics": ac,
                        "transcript": text_res.get("transcript", ""),
                        "language": text_res.get("language_detected", "auto"),
                        "fused_emotions": fused.get("fused_emotions", {})
                    }
                    st.download_button(
                        label="📥 DOWNLOAD RAW DATA (.JSON)",
                        data=json.dumps(dossier_dict, indent=2),
                        file_name="emosense_data.json",
                        mime="application/json",
                        use_container_width=True
                    )

        except Exception as exc:
            import traceback
            st.error(f"Error during scan: {exc}")
            st.code(traceback.format_exc())
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)