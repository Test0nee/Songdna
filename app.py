import streamlit as st
import asyncio
from shazamio import Shazam
import google.generativeai as genai
import tempfile
import os
import requests
import json
import pandas as pd
import numpy as np
import librosa
import plotly.graph_objects as go

# ---------- SPOTIFY AUTH ----------
SPOTIFY_CLIENT_ID = st.secrets.get("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = st.secrets.get("SPOTIFY_CLIENT_SECRET")


def get_spotify_token():
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
        return None
    url = "https://accounts.spotify.com/api/token"
    data = {"grant_type": "client_credentials"}
    try:
        r = requests.post(url, data=data, auth=(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET))
        return r.json().get("access_token")
    except Exception:
        return None


# ---------- 1. CONFIGURATION ----------
st.set_page_config(
    page_title="SunoSonic Studio",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- 2. ULTRA MODERN UI (CSS) ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700;900&family=JetBrains+Mono:wght@400;700&display=swap');
    
    /* GLOBAL THEME */
    .stApp {
        background-color: #020617;
        background-image: 
            radial-gradient(at 0% 0%, rgba(56, 189, 248, 0.10) 0px, transparent 55%),
            radial-gradient(at 100% 0%, rgba(129, 140, 248, 0.18) 0px, transparent 55%);
        font-family: 'Inter', sans-serif;
        color: #ffffff;
    }
    
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 5rem !important;
        max-width: 1100px !important;
    }
    
    header[data-testid="stHeader"], footer, [data-testid="stSidebar"] {display: none;}
    
    /* TOP BRANDING */
    .brand-wrap {
        text-align: center;
        margin-bottom: 24px;
    }
    .brand-logo {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 48px;
        height: 48px;
        border-radius: 999px;
        background: radial-gradient(circle at 30% 0%, #22d3ee, #4c1d95);
        box-shadow: 0 0 40px rgba(56,189,248,0.7);
        margin-bottom: 10px;
    }
    .brand-logo-inner {
        width: 20px;
        height: 20px;
        border-radius: 999px;
        border: 2px solid rgba(15,23,42,0.9);
        box-shadow: 0 0 0 2px rgba(15,23,42,0.6);
        background: radial-gradient(circle, #f9fafb 0%, #1e293b 55%, #020617 100%);
    }
    .brand-title {
        font-size: 2.6rem;
        font-weight: 900;
        letter-spacing: 0.24em;
        text-transform: uppercase;
        margin: 0;
        text-shadow: 0 0 35px rgba(59,130,246,0.5);
    }
    .brand-subtitle {
        font-size: 0.78rem;
        letter-spacing: 0.32em;
        text-transform: uppercase;
        color: #6b7280;
        margin-top: 4px;
    }
    
    .top-action {
        text-align: center;
        margin: 10px 0 24px 0;
    }

    /* HERO SECTION */
    .hero-wrapper {
        position: relative;
        border-radius: 28px;
        overflow: hidden;
        margin-bottom: 25px;
        box-shadow: 0 28px 80px -40px rgba(15,23,42,0.9);
        border: 1px solid rgba(148,163,184,0.3);
        height: 420px; /* taller hero */
        width: calc(100% + 120px); /* wider than content */
        margin-left: -60px;
        background: radial-gradient(circle at 0% 0%, #1e293b, #020617);
    }
    .hero-bg { 
        position: absolute;
        inset: 0;
        width: 100%; 
        height: 100%; 
        object-fit: cover; 
        transform: scale(1.15); /* zoom to hide square feel */
        filter: blur(24px) saturate(1.25) brightness(0.7);
        transform-origin: center;
    }
    .hero-overlay {
        position: absolute; 
        inset: 0;
        background:
            linear-gradient(to right, rgba(15,23,42,0.96) 0%, rgba(15,23,42,0.88) 40%, rgba(15,23,42,0.45) 68%, rgba(15,23,42,0.0) 100%),
            radial-gradient(circle at 0% 0%, rgba(59,130,246,0.25), transparent 55%),
            radial-gradient(circle at 100% 0%, rgba(236,72,153,0.25), transparent 55%);
        display: flex; 
        align-items: center; 
        padding: 30px 46px;
    }

    .hero-inner {
        display: flex;
        align-items: center;
        justify-content: space-between;
        width: 100%;
        gap: 28px;
    }

    .hero-meta {
        max-width: 65%;
    }

    .hero-cover-wrap {
        width: 180px;
        height: 180px;
        border-radius: 24px;
        padding: 6px;
        background: radial-gradient(circle at 0% 0%, rgba(56,189,248,0.4), rgba(236,72,153,0.2));
        box-shadow: 0 20px 40px rgba(15,23,42,0.95);
        border: 1px solid rgba(148,163,184,0.5);
    }
    .hero-cover-inner {
        width: 100%;
        height: 100%;
        border-radius: 18px;
        overflow: hidden;
        background: #020617;
    }
    .hero-cover-inner img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        display: block;
    }

    .verified-badge {
        background: rgba(37, 99, 235, 0.18); 
        color: #bfdbfe;
        border: 1px solid rgba(59, 130, 246, 0.65); 
        padding: 6px 12px;
        border-radius: 100px; 
        font-size: 0.75rem; 
        font-weight: 700;
        text-transform: uppercase; 
        display: inline-flex; 
        align-items: center; 
        gap: 6px;
        backdrop-filter: blur(14px);
    }
    .verified-dot {
        width: 8px;
        height: 8px;
        border-radius: 999px;
        background: #22c55e;
        box-shadow: 0 0 10px rgba(34,197,94,0.9);
    }
    .artist-title { 
        font-size: 3.1rem; 
        font-weight: 900; 
        line-height: 0.95; 
        margin: 10px 0 4px 0; 
        letter-spacing: -2px; 
        background: linear-gradient(to right, #ffffff, #e5e7eb); 
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent; 
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .song-subtitle { 
        font-size: 1.25rem; 
        color: #cbd5f5; 
        margin-bottom: 18px; 
        letter-spacing: -0.3px; 
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .meta-pill { 
        background: rgba(15,23,42,0.96); 
        border: 1px solid rgba(148,163,184,0.55); 
        padding: 6px 14px; 
        border-radius: 999px; 
        font-size: 0.8rem; 
        color: #e2e8f0; 
        margin-right: 8px; 
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .meta-pill span.icon {
        font-size: 0.9rem;
    }

    .hero-play-wrap {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .hero-play-btn {
        width: 60px;
        height: 60px;
        border-radius: 999px;
        background: radial-gradient(circle at 30% 0%, #22c55e, #16a34a);
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 18px 35px rgba(22,163,74,0.9);
        border: 2px solid rgba(15,23,42,0.8);
    }
    .hero-play-btn span {
        margin-left: 3px;
        width: 0;
        height: 0;
        border-top: 9px solid transparent;
        border-bottom: 9px solid transparent;
        border-left: 14px solid #052e16;
    }
    .hero-duration {
        font-size: 0.8rem;
        color: #e5e7eb;
        opacity: 0.8;
    }

    /* GLASS PANELS */
    .glass-panel {
        background: #020617; 
        border: 1px solid #1f2937;
        border-radius: 20px; 
        padding: 24px; 
        height: 100%; 
        position: relative;
    }
    .glow-cyan { 
        box-shadow: 0 0 40px -10px rgba(56, 189, 248, 0.18); 
        border-top: 1px solid rgba(56, 189, 248, 0.35); 
    }
    .glow-pink { 
        box-shadow: 0 0 40px -10px rgba(236, 72, 153, 0.18); 
        border-top: 1px solid rgba(236, 72, 153, 0.35); 
    }
    .glow-purple { 
        box-shadow: 0 0 40px -10px rgba(129, 140, 248, 0.25); 
        border-top: 1px solid rgba(129, 140, 248, 0.4); 
    }
    
    .panel-header { 
        display: flex; 
        align-items: center; 
        gap: 10px; 
        margin-bottom: 20px; 
        font-size: 0.8rem; 
        font-weight: 700; 
        letter-spacing: 2px; 
        text-transform: uppercase; 
        color: #6b7280; 
    }
    
    .stat-grid { 
        display: grid; 
        grid-template-columns: 1fr 1fr; 
        gap: 15px; 
    }
    .stat-box { 
        background: rgba(15,23,42,0.95); 
        padding: 15px; 
        border-radius: 12px; 
        border: 1px solid rgba(31,41,55,1); 
    }
    .stat-label { 
        font-size: 0.7rem; 
        color: #6b7280; 
        margin-bottom: 4px; 
        letter-spacing: 1px; 
    }
    .stat-value { 
        font-size: 1.1rem; 
        font-weight: 600; 
        color: #f9fafb; 
    }
    .small-tag { 
        font-size: 0.75rem; 
        padding: 4px 10px; 
        background: #020617; 
        border-radius: 6px; 
        color: #9ca3af; 
        border: 1px solid #1f2937; 
        margin-right: 5px; 
    }

    /* PROMPT BOX */
    .prompt-container { 
        font-family: 'JetBrains Mono', monospace; 
        background: #020617; 
        border: 1px solid #1f2937; 
        color: #22d3ee; 
        padding: 20px; 
        border-radius: 12px; 
        font-size: 0.9rem; 
        line-height: 1.6; 
    }
    .tip-item { 
        display: flex; 
        gap: 15px; 
        margin-bottom: 15px; 
    }
    .tip-num { 
        background: #111827; 
        width: 24px; 
        height: 24px; 
        display: flex; 
        align-items: center; 
        justify-content: center; 
        border-radius: 50%; 
        font-size: 0.7rem; 
    }

    /* LYRIC STUDIO */
    .lyric-area textarea { 
        background: #020617 !important; 
        color: #e5e7eb !important; 
        border: 1px solid #1f2937 !important; 
        font-family: 'Inter', sans-serif; 
    }
    .lyric-output { 
        background: #020617; 
        border: 1px solid #1f2937; 
        padding: 20px; 
        border-radius: 12px; 
        font-family: 'JetBrains Mono', monospace; 
        white-space: pre-wrap; 
        color: #a78bfa; 
        height: 300px; 
        overflow-y: auto; 
    }

    /* MODERN STREAMLIT AUDIO PLAYER */
    [data-testid="stAudio"] > div {
        background: #020617;
        border-radius: 999px;
        padding: 10px 18px;
        box-shadow: 0 14px 40px rgba(15,23,42,0.9);
        border: 1px solid #1f2937;
    }
    [data-testid="stAudio"] audio {
        width: 100%;
        filter: saturate(1.1);
    }

    audio::-webkit-media-controls-panel {
        background: #020617;
    }
    audio::-webkit-media-controls-current-time-display,
    audio::-webkit-media-controls-time-remaining-display {
        color: #e5e7eb;
    }

    /* UPLOAD */
    .upload-area { 
        border: 2px dashed #1f2937; 
        border-radius: 20px; 
        padding: 60px; 
        text-align: center; 
    }

    /* STRUCTURAL DYNAMICS PANEL */
    .viz-panel {
        background: #020617;
        border-radius: 24px;
        border: 1px solid #1f2937;
        padding: 22px 22px 18px 22px;
        margin-top: 10px;
        box-shadow: 0 24px 60px rgba(15,23,42,0.85);
        position: relative;
        overflow: hidden;
    }
    .viz-panel::before {
        content: "";
        position: absolute;
        inset: 0;
        background:
            radial-gradient(circle at 0% 0%, rgba(56,189,248,0.15), transparent 55%),
            radial-gradient(circle at 100% 0%, rgba(236,72,153,0.15), transparent 55%);
        opacity: 0.7;
        pointer-events: none;
    }
    .viz-inner {
        position: relative;
        z-index: 1;
    }
    .viz-header-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 12px;
    }
    .viz-title-wrap {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .viz-icon {
        width: 32px;
        height: 32px;
        border-radius: 12px;
        background: radial-gradient(circle at 0% 0%, #22d3ee, #6366f1);
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 0 20px rgba(59,130,246,0.8);
        border: 1px solid rgba(15,23,42,0.9);
    }
    .viz-icon span {
        font-size: 16px;
    }
    .viz-title {
        font-size: 0.95rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #e5e7eb;
        font-weight: 600;
    }
    .viz-header-line {
        height: 2px;
        background: linear-gradient(90deg, #22d3ee, #6366f1, #ec4899);
        border-radius: 999px;
        opacity: 0.9;
        margin-bottom: 14px;
    }
    .viz-chart-frame {
        border-radius: 18px;
        background: radial-gradient(circle at 0% 0%, rgba(15,23,42,0.5), rgba(15,23,42,0.95));
        padding: 10px 10px 4px 10px;
        border: 1px solid rgba(31,41,55,1);
    }
    .viz-footer {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-top: 10px;
        font-size: 0.8rem;
        color: #9ca3af;
        gap: 10px;
        flex-wrap: wrap;
    }
    .viz-toggles {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .pill-toggle {
        padding: 4px 12px;
        border-radius: 999px;
        border: 1px solid #1f2937;
        background: #020617;
        color: #9ca3af;
        font-size: 0.75rem;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .pill-toggle-dot {
        width: 10px;
        height: 10px;
        border-radius: 999px;
        background: #4b5563;
    }
    .pill-toggle.active {
        color: #e5e7eb;
        border-color: rgba(148,163,184,0.9);
        background: radial-gradient(circle at 0% 0%, rgba(56,189,248,0.18), rgba(15,23,42,1));
    }
    .pill-toggle.active .pill-toggle-dot.l {
        background: #22d3ee;
        box-shadow: 0 0 10px rgba(34,211,238,0.9);
    }
    .pill-toggle.active .pill-toggle-dot.r {
        background: #6366f1;
        box-shadow: 0 0 10px rgba(99,102,241,0.9);
    }
    .pill-toggle.active .pill-toggle-dot.rms {
        background: #ec4899;
        box-shadow: 0 0 10px rgba(236,72,153,0.9);
    }
    .viz-buttons {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .viz-btn {
        padding: 5px 12px;
        border-radius: 999px;
        border: 1px solid #1f2937;
        background: #020617;
        color: #e5e7eb;
        font-size: 0.75rem;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        cursor: default;
    }
    .viz-btn.primary {
        border-color: rgba(148,163,184,0.9);
        background: radial-gradient(circle at 0% 0%, rgba(56,189,248,0.18), rgba(15,23,42,1));
    }
    .viz-btn span.icon {
        font-size: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- 3. KNOWLEDGE BASE ----------
SUNO_TAGS = {
    "Structure": [
        "[Intro]",
        "[Verse]",
        "[Chorus]",
        "[Pre-Chorus]",
        "[Bridge]",
        "[Outro]",
        "[Drop]",
        "[Build]",
        "[Instrumental Break]",
        "[Hook]",
        "[Solo]",
    ],
    "Mood": [
        "Uplifting",
        "Melancholic",
        "Dark",
        "Euphoric",
        "Chill",
        "Aggressive",
        "Dreamy",
        "Nostalgic",
        "Epic",
        "Mysterious",
    ],
    "Vocals": [
        "[Male Vocals]",
        "[Female Vocals]",
        "[Duet]",
        "[Choir]",
        "[Whispered]",
        "[Belting]",
        "[Auto-tune]",
        "[Spoken Word]",
        "[Rapping]",
    ],
    "Production": [
        "[Reverb]",
        "[Delay]",
        "[Lo-fi]",
        "[Acoustic]",
        "[Synthesizer]",
        "[Bass Boosted]",
        "[Orchestral]",
        "[Minimal]",
    ],
}

# ---------- 4. LOGIC AND HELPERS ----------
api_key = st.secrets.get("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)


def run_async(coroutine):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coroutine)


async def fetch_artist_image(artist):
    """
    Get a wide artist or cover image.
    First try Spotify artist, then Deezer, finally fallback to Unsplash.
    """
    if not artist:
        return "https://images.unsplash.com/photo-1514525253440-b393452e8d26?w=1600"

    clean_artist = artist.split(",")[0].split("&")[0].split("feat")[0].strip()
    token = get_spotify_token()

    if token:
        try:
            headers = {"Authorization": f"Bearer {token}"}
            url = f"https://api.spotify.com/v1/search?q={clean_artist}&type=artist&limit=1"
            r = requests.get(url, headers=headers, timeout=5)
            data = r.json()
            items = data.get("artists", {}).get("items", [])
            if items:
                images = items[0].get("images", [])
                if images:
                    return images[0]["url"]
        except Exception:
            pass

    try:
        url = f"https://api.deezer.com/search/artist?q={clean_artist}"
        r = requests.get(url, timeout=5)
        data = r.json()
        if "data" in data and data["data"]:
            return data["data"][0]["picture_xl"]
    except Exception:
        pass

    return "https://images.unsplash.com/photo-1514525253440-b393452e8d26?w=1600"


def extract_audio_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        duration_sec = librosa.get_duration(y=y, sr=sr)
        minutes = int(duration_sec // 60)
        seconds = int(round(duration_sec % 60))
        duration_str = f"{minutes}:{seconds:02d}"

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = round(float(tempo)) if float(tempo) > 0 else 0

        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_vals = np.sum(chroma, axis=1)
        pitches = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        key_idx = int(np.argmax(chroma_vals))
        key = pitches[key_idx]

        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        avg_centroid = float(np.mean(spectral_centroids))
        if avg_centroid > 3000:
            brightness = "Bright / Aggressive"
        elif avg_centroid > 2000:
            brightness = "Balanced"
        else:
            brightness = "Dark / Mellow"

        rms = librosa.feature.rms(y=y)[0]
        avg_energy = float(np.mean(rms))
        if avg_energy > 0.1:
            intensity = "High Energy"
        elif avg_energy > 0.05:
            intensity = "Moderate Energy"
        else:
            intensity = "Low / Chill"

        harmonic, percussive = librosa.effects.hpss(y)

        S = np.abs(librosa.stft(harmonic, n_fft=2048, hop_length=512))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        voice_band = (freqs >= 300) & (freqs <= 3400)

        if np.any(voice_band):
            voice_energy = float(np.mean(S[voice_band]))
            non_voice_energy = float(np.mean(S[~voice_band])) + 1e-9
            voice_ratio = voice_energy / non_voice_energy
        else:
            voice_ratio = 1.0

        if voice_ratio > 1.15 and intensity != "Low / Chill":
            vocals_flag = "Likely Vocals"
        elif voice_ratio < 1.05 and intensity == "Low / Chill":
            vocals_flag = "Probably Instrumental / Background"
        else:
            vocals_flag = "Unclear, mixed or subtle vocals"

        perc_level = float(np.mean(np.abs(percussive)))
        harm_level = float(np.mean(np.abs(harmonic))) + 1e-9
        perc_ratio = perc_level / harm_level

        if perc_ratio > 1.2 and intensity != "Low / Chill":
            style_hint = "Band style with drums and rhythm section, possibly rock or pop with clear percussion."
        elif intensity == "Low / Chill" and brightness.startswith("Dark"):
            style_hint = "Lo-fi, ambient or chill ballad style, soft and relaxed."
        else:
            style_hint = "Modern production with a blend of electronic and acoustic elements."

        return {
            "success": True,
            "bpm": f"{bpm} BPM" if bpm > 0 else "Unknown BPM",
            "key": key,
            "timbre": brightness,
            "energy": intensity,
            "style_hint": style_hint,
            "vocals": vocals_flag,
            "voice_ratio": round(voice_ratio, 2),
            "duration": duration_str,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def identify_song(file_path):
    shazam = Shazam()
    try:
        out = await shazam.recognize(file_path)
        if "track" in out:
            track = out["track"]
            return {
                "found": True,
                "source": "shazam",
                "title": track.get("title"),
                "artist": track.get("subtitle"),
                "album_art": track.get("images", {}).get("coverart"),
                "genre": track.get("genres", {}).get("primary", "Electronic"),
            }
        return {"found": False}
    except Exception:
        return {"found": False, "error": "Shazam failed"}


def analyze_gemini_json(song_data):
    if not api_key:
        return None

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
    except Exception:
        model = genai.GenerativeModel("gemini-1.5-flash")

    if song_data.get("source") == "librosa":
        prompt = f"""
You are an AI assistant specialized in music analysis and AI music prompting.

A user uploaded a completely unknown track, possibly created in Suno or another AI tool.
You only know these audio features:

- Tempo: {song_data.get('bpm')}
- Key: {song_data.get('key')}
- Timbre: {song_data.get('timbre')}
- Energy: {song_data.get('energy')}
- Style hint: {song_data.get('style_hint', '')}
- Duration: {song_data.get('duration', 'Unknown')}
- Vocal presence heuristic: {song_data.get('vocals', 'Unknown')} (voice band ratio {song_data.get('voice_ratio', 0)})

Use the vocal heuristic as a strong hint.

Based on all this, infer and return ONLY valid JSON in this structure:

{{
  "mood": "single word",
  "tempo": "{song_data.get('bpm')}",
  "key": "{song_data.get('key')} (estimated)",
  "genre": "concise genre",
  "instruments": ["instrument 1", "instrument 2"],
  "vocal_type": "description",
  "vocal_style": "description",
  "suno_prompt": "one or two sentences as a style prompt",
  "tips": [
    "tip 1",
    "tip 2",
    "tip 3"
  ]
}}
"""
    else:
        prompt = f"""
You are an AI assistant specialized in music analysis and AI music prompting.

Analyze the song "{song_data['title']}" by "{song_data['artist']}".
Infer mood, genre, tempo, key, instruments, vocal type/style and a Suno-ready style prompt.

Return ONLY valid JSON in this structure:

{{
  "mood": "single word",
  "tempo": "number + ' BPM'",
  "key": "musical key",
  "genre": "concise genre",
  "instruments": ["instrument 1", "instrument 2"],
  "vocal_type": "description",
  "vocal_style": "description",
  "suno_prompt": "one or two sentences as a style prompt",
  "tips": [
    "tip 1",
    "tip 2",
    "tip 3"
  ]
}}
"""

    try:
        response = model.generate_content(prompt)
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_text)
    except Exception:
        return {
            "mood": "Unknown",
            "tempo": song_data.get("bpm", ""),
            "key": song_data.get("key", ""),
            "genre": song_data.get("genre", "Unknown"),
            "instruments": [],
            "vocal_type": "Unknown",
            "vocal_style": "",
            "suno_prompt": f"{song_data.get('genre', 'Unknown')} track at {song_data.get('bpm', '')}, {song_data.get('energy', '')} energy.",
            "tips": [],
        }


def format_lyrics_with_tags(raw_lyrics, song_analysis):
    if not api_key:
        return "Please set GEMINI_API_KEY in Streamlit secrets."
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
    except Exception:
        model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = f"""
Act as a Suno.ai meta-tagging expert.

CONTEXT:
- Genre: {song_analysis.get('genre', 'Pop')}
- Mood: {song_analysis.get('mood', 'General')}
- Official Tag Dictionary: {SUNO_TAGS}

TASK:
Insert structural tags like [Intro], [Verse], [Chorus], [Bridge], [Outro], etc, into the following lyrics.
Return only the tagged lyrics.

LYRICS:
{raw_lyrics}
"""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {e}"


# ---------- 5. MAIN APPLICATION ----------
def main():
    if "song_data" not in st.session_state:
        st.session_state.song_data = None
        st.session_state.analysis = None
        st.session_state.formatted_lyrics = ""
        st.session_state.uploaded_bytes = None

    if not api_key:
        st.warning(
            "Gemini API key is not configured. Audio analysis will work, but AI prompts and lyric tagging will be limited."
        )

    # BRAND HEADER
    st.markdown(
        """
        <div class="brand-wrap">
            <div class="brand-logo"><div class="brand-logo-inner"></div></div>
            <h1 class="brand-title">SUNOSONIC</h1>
            <div class="brand-subtitle">AI AUDIO INTELLIGENCE STUDIO</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # STATE 1: UPLOAD
    if not st.session_state.song_data:
        st.markdown(
            '<div class="top-action"><span style="font-size:0.8rem; letter-spacing:0.18em; text-transform:uppercase; color:#9ca3af;">Upload a track to begin analysis</span></div>',
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(" ", type=["mp3", "wav", "ogg"])

        if not uploaded_file:
            st.info("👆 Drop an audio file above to begin.")

        if uploaded_file:
            with st.spinner("🎧 Analyzing audio DNA..."):
                st.session_state.uploaded_bytes = uploaded_file.getvalue()

                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                    tmp.write(st.session_state.uploaded_bytes)
                    tmp_path = tmp.name

                audio_stats = extract_audio_features(tmp_path)

                if not audio_stats.get("success"):
                    st.error("Audio file is corrupted or unreadable.")
                    os.remove(tmp_path)
                    return

                shazam_result = run_async(identify_song(tmp_path))

                if shazam_result.get("found"):
                    result = {
                        "found": True,
                        "source": "shazam",
                        "title": shazam_result.get("title"),
                        "artist": shazam_result.get("artist"),
                        "album_art": shazam_result.get("album_art"),
                        "genre": shazam_result.get("genre", "Electronic"),
                        "bpm": audio_stats["bpm"],
                        "key": audio_stats["key"],
                        "timbre": audio_stats["timbre"],
                        "energy": audio_stats["energy"],
                        "style_hint": audio_stats["style_hint"],
                        "vocals": audio_stats["vocals"],
                        "voice_ratio": audio_stats["voice_ratio"],
                        "duration": audio_stats["duration"],
                    }
                else:
                    st.toast("Metadata not found. Engaging deep audio scan...", icon="🧬")
                    result = {
                        "found": True,
                        "source": "librosa",
                        "title": "Unknown Track (Deep Scan)",
                        "artist": "Audio Fingerprint",
                        "album_art": "https://images.unsplash.com/photo-1511379938547-c1f69419868d?w=1600",
                        "genre": "Analyzing Signal...",
                        "bpm": audio_stats["bpm"],
                        "key": audio_stats["key"],
                        "timbre": audio_stats["timbre"],
                        "energy": audio_stats["energy"],
                        "style_hint": audio_stats["style_hint"],
                        "vocals": audio_stats["vocals"],
                        "voice_ratio": audio_stats["voice_ratio"],
                        "duration": audio_stats["duration"],
                    }

                result["artist_bg"] = run_async(fetch_artist_image(result["artist"]))

                st.session_state.song_data = result
                st.session_state.analysis = analyze_gemini_json(result)
                os.remove(tmp_path)
                st.rerun()

    # STATE 2: DASHBOARD
    else:
        data = st.session_state.song_data
        ai = st.session_state.analysis or {}

        cols = st.columns([1, 1, 1])
        with cols[1]:
            if st.button("← ANALYZE NEW TRACK", use_container_width=True):
                st.session_state.song_data = None
                st.session_state.analysis = None
                st.session_state.formatted_lyrics = ""
                st.session_state.uploaded_bytes = None
                st.rerun()

        hero_bg = data.get("artist_bg") or "https://images.unsplash.com/photo-1511379938547-c1f69419868d?w=1600"
        cover_img = data.get("album_art") or hero_bg

        st.markdown(
            f"""
            <div class="hero-wrapper">
                <img src="{hero_bg}" class="hero-bg">
                <div class="hero-overlay">
                    <div class="hero-inner">
                        <div class="hero-cover-wrap">
                            <div class="hero-cover-inner">
                                <img src="{cover_img}">
                            </div>
                        </div>
                        <div class="hero-meta">
                            <div class="verified-badge">
                                <span class="verified-dot"></span>
                                <span>{data['source'].upper()} ANALYSIS</span>
                            </div>
                            <div class="artist-title">{data['artist']}</div>
                            <div class="song-subtitle">{data['title']}</div>
                            <div class="meta-tags">
                                <span class="meta-pill"><span class="icon">🎵</span>{ai.get('genre', data.get('genre', 'Unknown'))}</span>
                                <span class="meta-pill"><span class="icon">⏱</span>{ai.get('tempo', data.get('bpm', '--'))}</span>
                                <span class="meta-pill"><span class="icon">🎹</span>{ai.get('key', data.get('key', '--'))}</span>
                                <span class="meta-pill"><span class="icon">⏰</span>{data.get('duration', '--:--')}</span>
                            </div>
                        </div>
                        <div class="hero-play-wrap">
                            <div class="hero-play-btn"><span></span></div>
                            <div class="hero-duration">{data.get('duration', '--:--')}</div>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.session_state.get("uploaded_bytes"):
            st.audio(st.session_state.uploaded_bytes, format="audio/mp3")

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            instruments_html = "".join(
                [f'<span class="small-tag">{inst}</span>' for inst in ai.get("instruments", [])]
            )
            st.markdown(
                f"""
                <div class="glass-panel glow-cyan">
                    <div class="panel-header"><span style="color:#38bdf8">⚡</span> SONIC PROFILE</div>
                    <div class="stat-grid">
                        <div class="stat-box">
                            <div class="stat-label">MOOD</div>
                            <div class="stat-value">{ai.get('mood', '-')}</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-label">ENERGY</div>
                            <div class="stat-value">{data.get('energy', 'N/A')}</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-label">VOCALS</div>
                            <div class="stat-value">{data.get('vocals', 'Unknown')}</div>
                        </div>
                    </div>
                    <div style="margin-top:10px; font-size:0.75rem; color:#9ca3af;">
                        BPM: {data.get('bpm', '--')} · Key: {data.get('key', '--')} · Timbre: {data.get('timbre', '--')}
                    </div>
                    <div style="margin-top:15px">{instruments_html}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
                <div class="glass-panel glow-pink">
                    <div class="panel-header"><span style="color:#ec4899">🎙</span> VOCAL ARCHITECTURE</div>
                    <div class="stat-grid">
                        <div class="stat-box">
                            <div class="stat-label">TYPE</div>
                            <div class="stat-value">{ai.get('vocal_type', '-')}</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-label">STYLE</div>
                            <div class="stat-value">{ai.get('vocal_style', 'Modern')}</div>
                        </div>
                    </div>
                    <div style="margin-top:15px; font-size:0.9rem; color:#e5e7eb; font-style:italic;">"{ai.get('vocal_style', '-')}"</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # VISUALIZER MODERN PANEL
        st.markdown("<br>", unsafe_allow_html=True)

        spikiness = 2.0 if "High" in data.get("energy", "") else 0.6
        chart_data = pd.DataFrame(
            np.random.randn(240, 3) * spikiness,
            columns=["L", "R", "RMS"],
        )

        fig = go.Figure()
        colors = {"L": "#22d3ee", "R": "#6366f1", "RMS": "#ec4899"}

        for ch in ["L", "R", "RMS"]:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(chart_data))),
                    y=chart_data[ch],
                    mode="lines",
                    name=ch,
                    line=dict(color=colors[ch], width=2.5),
                    hoverinfo="skip",
                )
            )

        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            xaxis=dict(
                showgrid=False,
                showline=False,
                zeroline=False,
                showticklabels=False,
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="rgba(148,163,184,0.2)",
                zeroline=False,
                showticklabels=False,
            ),
        )

        st.markdown(
            """
            <div class="viz-panel">
              <div class="viz-inner">
                <div class="viz-header-row">
                    <div class="viz-title-wrap">
                        <div class="viz-icon"><span>📈</span></div>
                        <div class="viz-title">STRUCTURAL DYNAMICS & RMS (SIMULATED)</div>
                    </div>
                </div>
                <div class="viz-header-line"></div>
                <div class="viz-chart-frame">
            """,
            unsafe_allow_html=True,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False},
        )

        st.markdown(
            """
                </div>
                <div class="viz-footer">
                    <div class="viz-toggles">
                        <span class="pill-toggle active">
                            <span class="pill-toggle-dot l"></span>L
                        </span>
                        <span class="pill-toggle active">
                            <span class="pill-toggle-dot r"></span>R
                        </span>
                        <span class="pill-toggle active">
                            <span class="pill-toggle-dot rms"></span>RMS
                        </span>
                    </div>
                    <div class="viz-buttons">
                        <span class="viz-btn primary"><span class="icon">🔍</span>Zoom</span>
                        <span class="viz-btn"><span class="icon">⬇️</span>Export Data</span>
                        <span class="viz-btn"><span class="icon">⚙️</span>Settings</span>
                    </div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # PROMPT AND TIPS
        st.markdown("<br>", unsafe_allow_html=True)

        p_col, t_col = st.columns([1.5, 1])
        with p_col:
            st.markdown(
                f'<div class="glass-panel glow-purple"><div class="panel-header"><span style="color:#818cf8">🎹</span> SUNO AI STYLE PROMPT</div><div class="prompt-container">{ai.get("suno_prompt", "Prompt not available.")}</div></div>',
                unsafe_allow_html=True,
            )

        with t_col:
            tips_list = ai.get("tips", [])
            tips_html = "".join(
                [
                    f'<div class="tip-item"><div class="tip-num">{i+1}</div><div>{tip}</div></div>'
                    for i, tip in enumerate(tips_list)
                ]
            )
            st.markdown(
                f'<div class="glass-panel"><div class="panel-header">💡 PRO TIPS</div>{tips_html or "No tips generated."}</div>',
                unsafe_allow_html=True,
            )

        # LYRIC STUDIO
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="glass-panel glow-purple" style="border: 1px solid #1f2937;">',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="panel-header">📝 LYRIC STUDIO <span class="small-tag" style="margin-left:10px; color:#a78bfa; border-color:#4c1d95;">AUTO TAGGER</span></div>',
            unsafe_allow_html=True,
        )

        l_col1, l_col2 = st.columns(2)
        with l_col1:
            st.caption("Paste raw lyrics here")
            raw_input = st.text_area(
                "raw",
                height=300,
                placeholder="Type or paste lyrics...",
                label_visibility="collapsed",
                key="raw_lyrics_input",
            )

            with st.expander("📚 Suno meta tags reference"):
                for cat, tags in SUNO_TAGS.items():
                    st.markdown(f"**{cat}**")
                    st.markdown(" ".join([f"`{t}`" for t in tags]))

            if st.button("✨ Apply Suno meta tags", use_container_width=True):
                if raw_input:
                    with st.spinner(
                        "AI is structuring your lyrics based on the song style..."
                    ):
                        st.session_state.formatted_lyrics = format_lyrics_with_tags(
                            raw_input, ai
                        )
                else:
                    st.warning("Please paste some lyrics first.")

        with l_col2:
            st.caption("Formatted output")
            if st.session_state.formatted_lyrics:
                st.code(
                    st.session_state.formatted_lyrics,
                    language="markdown",
                    line_numbers=False,
                )
            else:
                st.markdown(
                    '<div class="lyric-output" style="color:#555; display:flex; align-items:center; justify-content:center;">Result will appear here...</div>',
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
