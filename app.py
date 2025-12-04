import streamlit as st
import streamlit.components.v1 as components
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
from PIL import Image
from io import BytesIO

# --- SPOTIFY AUTH ---
SPOTIFY_CLIENT_ID = st.secrets.get("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = st.secrets.get("SPOTIFY_CLIENT_SECRET")

def get_spotify_token():
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
        return None
    url = "https://accounts.spotify.com/api/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {"grant_type": "client_credentials"}
    try:
        r = requests.post(url, headers=headers, data=data, auth=(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET))
        if r.status_code == 200:
            return r.json().get("access_token")
    except Exception:
        return None

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="SunoSonic Studio",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. ULTRA-MODERN UI (CSS) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700;900&family=JetBrains+Mono:wght@400;700&display=swap');
    
    .stApp {
        background-color: #020617;
        background-image: 
            radial-gradient(at 0% 0%, rgba(56, 189, 248, 0.10) 0px, transparent 55%),
            radial-gradient(at 100% 0%, rgba(129, 140, 248, 0.18) 0px, transparent 55%);
        font-family: 'Inter', sans-serif;
        color: #ffffff;
    }
    
    .block-container { padding-top: 1.5rem !important; padding-bottom: 5rem !important; max-width: 1100px !important; }
    header[data-testid="stHeader"], footer, [data-testid="stSidebar"] {display: none;}
    
    /* BRANDING */
    .brand-wrap { text-align: center; margin-bottom: 24px; }
    .brand-logo { display: inline-flex; align-items: center; justify-content: center; width: 48px; height: 48px; border-radius: 999px; background: radial-gradient(circle at 30% 0%, #22d3ee, #4c1d95); box-shadow: 0 0 40px rgba(56,189,248,0.7); margin-bottom: 10px; }
    .brand-logo-inner { width: 20px; height: 20px; border-radius: 999px; border: 2px solid rgba(15,23,42,0.9); box-shadow: 0 0 0 2px rgba(15,23,42,0.6); background: radial-gradient(circle, #f9fafb 0%, #1e293b 55%, #020617 100%); }
    .brand-title { font-size: 2.6rem; font-weight: 900; letter-spacing: 0.24em; text-transform: uppercase; margin: 0; text-shadow: 0 0 35px rgba(59,130,246,0.5); }
    .brand-subtitle { font-size: 0.78rem; letter-spacing: 0.32em; text-transform: uppercase; color: #6b7280; margin-top: 4px; }
    .top-action { text-align: center; margin: 10px 0 24px 0; }

    /* HERO */
    .hero-wrapper { position: relative; border-radius: 28px; overflow: hidden; margin-bottom: 25px; box-shadow: 0 28px 80px -40px rgba(15,23,42,0.9); border: 1px solid rgba(148,163,184,0.3); height: 320px; background: radial-gradient(circle at 0% 0%, #1e293b, #020617); }
    .hero-bg { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; transform: scale(1.15); filter: blur(24px) saturate(1.25) brightness(0.7); }
    .hero-overlay { position: absolute; inset: 0; background: linear-gradient(to right, rgba(15,23,42,0.96) 0%, rgba(15,23,42,0.88) 40%, rgba(15,23,42,0.45) 68%, rgba(15,23,42,0.0) 100%); display: flex; align-items: center; padding: 30px 46px; }
    .hero-inner { display: flex; align-items: center; justify-content: space-between; width: 100%; gap: 28px; }
    .hero-meta { max-width: 65%; }
    .hero-cover-wrap { width: 180px; height: 180px; border-radius: 24px; padding: 6px; background: radial-gradient(circle at 0% 0%, rgba(56,189,248,0.4), rgba(236,72,153,0.2)); box-shadow: 0 20px 40px rgba(15,23,42,0.95); border: 1px solid rgba(148,163,184,0.5); }
    .hero-cover-inner { width: 100%; height: 100%; border-radius: 18px; overflow: hidden; background: #020617; }
    .hero-cover-inner img { width: 100%; height: 100%; object-fit: cover; display: block; }
    .verified-badge { background: rgba(37, 99, 235, 0.18); color: #bfdbfe; border: 1px solid rgba(59, 130, 246, 0.65); padding: 6px 12px; border-radius: 100px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; display: inline-flex; align-items: center; gap: 6px; backdrop-filter: blur(14px); }
    .verified-dot { width: 8px; height: 8px; border-radius: 999px; background: #22c55e; box-shadow: 0 0 10px rgba(34,197,94,0.9); }
    .artist-title { font-size: 3.1rem; font-weight: 900; line-height: 0.95; margin: 10px 0 4px 0; letter-spacing: -2px; background: linear-gradient(to right, #ffffff, #e5e7eb); -webkit-background-clip: text; -webkit-text-fill-color: transparent; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .song-subtitle { font-size: 1.25rem; color: #cbd5f5; margin-bottom: 18px; letter-spacing: -0.3px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .meta-pill { background: rgba(15,23,42,0.96); border: 1px solid rgba(148,163,184,0.55); padding: 6px 14px; border-radius: 999px; font-size: 0.8rem; color: #e2e8f0; margin-right: 8px; display: inline-flex; align-items: center; gap: 6px; }

    /* PANELS */
    .glass-panel { background: #020617; border: 1px solid #1f2937; border-radius: 20px; padding: 24px; height: 100%; position: relative; }
    .glow-cyan { box-shadow: 0 0 40px -10px rgba(56, 189, 248, 0.18); border-top: 1px solid rgba(56, 189, 248, 0.35); }
    .glow-pink { box-shadow: 0 0 40px -10px rgba(236, 72, 153, 0.18); border-top: 1px solid rgba(236, 72, 153, 0.35); }
    .glow-purple { box-shadow: 0 0 40px -10px rgba(129, 140, 248, 0.25); border-top: 1px solid rgba(129, 140, 248, 0.4); }
    .panel-header { display: flex; align-items: center; gap: 10px; margin-bottom: 20px; font-size: 0.8rem; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; color: #6b7280; }
    .stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
    .stat-box { background: rgba(15,23,42,0.95); padding: 15px; border-radius: 12px; border: 1px solid rgba(31,41,55,1); }
    .stat-label { font-size: 0.7rem; color: #6b7280; margin-bottom: 4px; letter-spacing: 1px; }
    .stat-value { font-size: 1.1rem; font-weight: 600; color: #f9fafb; }
    .small-tag { font-size: 0.75rem; padding: 4px 10px; background: #020617; border-radius: 6px; color: #9ca3af; border: 1px solid #1f2937; margin-right: 5px; }

    /* PROMPT & LYRIC */
    .prompt-container { font-family: 'JetBrains Mono', monospace; background: #020617; border: 1px solid #1f2937; color: #22d3ee; padding: 20px; border-radius: 12px; font-size: 0.9rem; line-height: 1.6; }
    .tip-item { display: flex; gap: 15px; margin-bottom: 15px; }
    .tip-num { background: #111827; width: 24px; height: 24px; display: flex; align-items: center; justify-content: center; border-radius: 50%; font-size: 0.7rem; }
    .lyric-area textarea { background: #020617 !important; color: #e5e7eb !important; border: 1px solid #1f2937 !important; font-family: 'Inter', sans-serif; }
    .lyric-output { background: #020617; border: 1px solid #1f2937; padding: 20px; border-radius: 12px; font-family: 'JetBrains Mono', monospace; white-space: pre-wrap; color: #a78bfa; height: 300px; overflow-y: auto; }

    /* AUDIO */
    [data-testid="stAudio"] > div { background: #020617; border-radius: 999px; padding: 10px 18px; box-shadow: 0 14px 40px rgba(15,23,42,0.9); border: 1px solid #1f2937; }
    [data-testid="stAudio"] audio { width: 100%; filter: saturate(1.1); }
    .upload-area { border: 2px dashed #1f2937; border-radius: 20px; padding: 60px; text-align: center; }
    </style>
""", unsafe_allow_html=True)

# --- 3. KNOWLEDGE BASE ---
SUNO_TAGS = {
    "Structure": ["[Intro]", "[Verse]", "[Chorus]", "[Pre-Chorus]", "[Bridge]", "[Drop]", "[Hook]", "[Instrumental Break]", "[Outro]"],
    "Vocals": ["[Male Vocals]", "[Female Vocals]", "[Duet]", "[Choir]", "[Whispered]", "[Belting]", "[Auto-tune]", "[Spoken Word]", "[Rap Flow]"],
    "Mood": ["[Mood: Uplifting]", "[Mood: Melancholic]", "[Mood: Aggressive]", "[Energy: High]", "[Energy: Low]", "[Atmosphere: Dark]"]
}

# --- 4. LOGIC ---
api_key = st.secrets.get("GEMINI_API_KEY")
if api_key: genai.configure(api_key=api_key)

def run_async(coroutine):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coroutine)

async def fetch_artist_image(artist):
    if not artist: return "https://images.unsplash.com/photo-1514525253440-b393452e8d26?w=1600"
    clean_artist = artist.split(",")[0].split("&")[0].split("feat")[0].strip()
    token = get_spotify_token()
    if token:
        try:
            url = f"https://api.spotify.com/v1/search?q={clean_artist}&type=artist&limit=1"
            r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=5)
            data = r.json()
            if data.get("artists", {}).get("items"): return data["artists"]["items"][0]["images"][0]["url"]
        except: pass
    try:
        url = f"https://api.deezer.com/search/artist?q={clean_artist}"
        r = requests.get(url, timeout=5)
        if r.json().get('data'): return r.json()['data'][0]['picture_xl']
    except: pass
    return "https://images.unsplash.com/photo-1514525253440-b393452e8d26?w=1600"

# --- ADVANCED AUDIO ANALYSIS (NOW WITH SECTIONS) ---
def extract_audio_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None, duration=180) # Limit to 3 min for speed
        duration_sec = librosa.get_duration(y=y, sr=sr)
        duration_str = f"{int(duration_sec // 60)}:{int(round(duration_sec % 60)):02d}"
        
        # 1. Basics
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = round(float(tempo)) if float(tempo) > 0 else 0
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        key_idx = int(np.argmax(np.sum(chroma, axis=1)))
        key = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"][key_idx]
        
        # 2. Energy & Brightness
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        brightness = "Bright" if float(np.mean(spectral_centroids)) > 3000 else "Dark"
        rms = librosa.feature.rms(y=y)[0]
        intensity = "High" if float(np.mean(rms)) > 0.1 else "Low"
        
        # 3. Structure Detection (Simple Onset-based)
        # We look for large peaks in "Novelty" to guess section changes
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        peaks = librosa.util.peak_pick(onset_env, 3, 3, 3, 5, 0.5, 10)
        times = librosa.frames_to_time(peaks, sr=sr)
        
        # Filter sections to avoid having one every second (min 15 sec apart)
        sections = []
        last_time = 0
        for t in times:
            if t - last_time > 15:
                sections.append(t)
                last_time = t
        
        # 4. Waveform Data for Plotly (Resampled for performance)
        hop_length = 1024 # Larger hop = less points = faster chart
        rms_resampled = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        rms_norm = rms_resampled / np.max(rms_resampled)
        
        # Fake "Left/Right" by adding slight noise to RMS
        l_channel = rms_norm + (np.random.rand(len(rms_norm)) * 0.1)
        r_channel = rms_norm - (np.random.rand(len(rms_norm)) * 0.1)

        return {
            "success": True, "bpm": f"{bpm} BPM", "key": key, "timbre": brightness,
            "energy": intensity, "duration": duration_str,
            "waveform_l": l_channel.tolist(),
            "waveform_r": r_channel.tolist(),
            "sections": sections # List of timestamps in seconds
        }
    except Exception as e: return {"success": False, "error": str(e)}

async def identify_song(file_path):
    shazam = Shazam()
    try:
        out = await shazam.recognize(file_path)
        if "track" in out:
            track = out["track"]
            return {
                "found": True, "source": "shazam", "title": track.get("title"),
                "artist": track.get("subtitle"), "album_art": track.get("images", {}).get("coverart"),
                "genre": track.get("genres", {}).get("primary", "Electronic")
            }
        return {"found": False}
    except: return {"found": False}

def analyze_gemini_json(song_data):
    if not api_key: return None
    try: model = genai.GenerativeModel("gemini-2.5-flash")
    except: model = genai.GenerativeModel("gemini-1.5-flash")
    
    prompt = f"""
    Analyze song: {song_data.get('title', 'Unknown')} by {song_data.get('artist', 'Unknown')}.
    Audio: {song_data.get('bpm')}, {song_data.get('key')}, {song_data.get('energy')}.
    Return JSON: {{ "mood": "...", "genre": "...", "instruments": ["..."], "vocal_type": "...", "vocal_style": "...", "suno_prompt": "...", "tips": ["..."] }}
    """
    try: return json.loads(model.generate_content(prompt).text.replace("```json", "").replace("```", "").strip())
    except: return None

def format_lyrics_with_tags(raw, analysis):
    if not api_key: return "Error"
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = f"Insert Suno tags {SUNO_TAGS} into these lyrics for {analysis.get('genre')}: {raw}"
    try: return model.generate_content(prompt).text
    except: return "Error"

# --- 5. MAIN ---
def main():
    if "song_data" not in st.session_state: st.session_state.song_data = None
    if "analysis" not in st.session_state: st.session_state.analysis = None
    if "formatted_lyrics" not in st.session_state: st.session_state.formatted_lyrics = ""
    if "uploaded_bytes" not in st.session_state: st.session_state.uploaded_bytes = None

    st.markdown("""<div class="brand-wrap"><div class="brand-logo"><div class="brand-logo-inner"></div></div><h1 class="brand-title">SUNOSONIC</h1><div class="brand-subtitle">AI AUDIO INTELLIGENCE STUDIO</div></div>""", unsafe_allow_html=True)

    if not st.session_state.song_data:
        st.markdown('<div class="top-action"><span style="font-size:0.8rem; letter-spacing:0.18em; text-transform:uppercase; color:#9ca3af;">Upload a track to begin analysis</span></div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(" ", type=["mp3", "wav", "ogg"])
        if uploaded_file:
            with st.spinner("🎧 Decoding Audio DNA..."):
                st.session_state.uploaded_bytes = uploaded_file.getvalue()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                    tmp.write(st.session_state.uploaded_bytes)
                    tmp_path = tmp.name
                
                audio_stats = extract_audio_features(tmp_path)
                if not audio_stats.get("success"):
                    st.error("Audio corrupt."); os.remove(tmp_path); return
                
                shazam_result = run_async(identify_song(tmp_path))
                result = shazam_result if shazam_result.get("found") else {"found": True, "source": "librosa", "title": "Unknown Track", "artist": "Deep Scan", "album_art": "", "genre": "Analyzing..."}
                result.update(audio_stats)
                result["artist_bg"] = run_async(fetch_artist_image(result["artist"]))
                
                st.session_state.song_data = result
                st.session_state.analysis = analyze_gemini_json(result)
                os.remove(tmp_path)
                st.rerun()

    else:
        data = st.session_state.song_data
        ai = st.session_state.analysis or {}
        
        # HERO
        st.markdown(f"""
            <div class="hero-wrapper">
                <img src="{data.get('artist_bg')}" class="hero-bg">
                <div class="hero-overlay">
                    <div class="hero-inner">
                        <div class="hero-cover-wrap"><div class="hero-cover-inner"><img src="{data.get('album_art') or data.get('artist_bg')}"></div></div>
                        <div class="hero-meta">
                            <div class="verified-badge"><span class="verified-dot"></span><span>{data['source'].upper()} ANALYSIS</span></div>
                            <div class="artist-title">{data['artist']}</div>
                            <div class="song-subtitle">{data['title']}</div>
                            <div class="meta-tags"><span class="meta-pill">🎵 {ai.get('genre', 'Unknown')}</span><span class="meta-pill">⏱ {data.get('bpm')}</span><span class="meta-pill">🎹 {data.get('key')}</span></div>
                        </div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.get("uploaded_bytes"): st.audio(st.session_state.uploaded_bytes, format="audio/mp3")
        
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            inst_html = "".join([f'<span class="small-tag">{i}</span>' for i in ai.get("instruments", [])])
            st.markdown(f"""<div class="glass-panel glow-cyan"><div class="panel-header"><span style="color:#38bdf8">⚡</span> SONIC PROFILE</div><div class="stat-grid"><div class="stat-box"><div class="stat-label">MOOD</div><div class="stat-value">{ai.get('mood')}</div></div><div class="stat-box"><div class="stat-label">ENERGY</div><div class="stat-value">{data.get('energy')}</div></div></div><div style="margin-top:15px">{inst_html}</div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="glass-panel glow-pink"><div class="panel-header"><span style="color:#ec4899">🎙</span> VOCAL ARCHITECTURE</div><div class="stat-grid"><div class="stat-box"><div class="stat-label">TYPE</div><div class="stat-value">{ai.get('vocal_type')}</div></div><div class="stat-box"><div class="stat-label">STYLE</div><div class="stat-value">{ai.get('vocal_style')}</div></div></div></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # --- INTERACTIVE VISUALIZER ---
        # Using real data from Librosa
        y_data = data.get("waveform_l", [])
        x_data = np.linspace(0, 100, len(y_data))
        sections = data.get("sections", [])
        
        fig = go.Figure()
        
        # Waveform
        fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines', name='Energy', line=dict(color='#22d3ee', width=1.5), fill='tozeroy', hovertemplate='<b>Energy</b>: %{y:.2f}<br>Time: %{x:.1f}%<extra></extra>'))
        
        # Structure Markers (Sections)
        max_y = max(y_data) if y_data else 1
        for i, sec_time in enumerate(sections):
            # Normalize time to 0-100 scale approximately for the chart
            sec_x = (sec_time / 180) * 100 # assuming 180s duration
            if sec_x > 100: break
            
            # Draw vertical line
            fig.add_shape(type="line", x0=sec_x, y0=0, x1=sec_x, y1=max_y, line=dict(color="rgba(255, 255, 255, 0.3)", width=1, dash="dot"))
            # Add Annotation
            fig.add_annotation(x=sec_x, y=max_y, text=f"SEC {i+1}", showarrow=False, yshift=10, font=dict(color="#ec4899", size=10))

        fig.update_layout(
            paper_bgcolor='#020617', plot_bgcolor='#020617',
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', showticklabels=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', showticklabels=False),
            height=200, showlegend=False, hovermode="x unified"
        )
        
        # Custom Tooltip HTML injection isn't possible in standard Plotly Python, 
        # so we rely on 'hovertemplate' above which gives us the "Rich Info on Hover" requirement.
        
        plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn', config={'displayModeBar': False})
        
        components.html(f"""
        <!DOCTYPE html><html><head><link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
        <style>body{{margin:0;background-color:transparent;font-family:'Inter',sans-serif;overflow:hidden;}}.viz-panel{{background:#020617;border-radius:20px;border:1px solid #1f2937;padding:20px;box-shadow:0 24px 60px rgba(15,23,42,0.85);position:relative;}}.viz-header{{display:flex;align-items:center;gap:10px;margin-bottom:10px;}}.viz-icon{{width:28px;height:28px;border-radius:8px;background:linear-gradient(135deg,#22d3ee,#6366f1);display:flex;align-items:center;justify-content:center;font-size:14px;box-shadow:0 0 15px rgba(34,211,238,0.6);}}.viz-title{{color:#e5e7eb;font-weight:700;font-size:12px;letter-spacing:2px;text-transform:uppercase;}}.viz-chart-container{{border-radius:12px;background:rgba(15,23,42,0.6);border:1px solid #374151;overflow:hidden;}}</style></head>
        <body><div class="viz-panel"><div class="viz-header"><div class="viz-icon">📈</div><div class="viz-title">STRUCTURAL DYNAMICS (META TAGS DETECTED)</div></div><div class="viz-chart-container">{plot_html}</div></div></body></html>
        """, height=320, scrolling=False)

        st.markdown("<br>", unsafe_allow_html=True)
        # ... (Rest of UI) ...
        p_col, t_col = st.columns([1.5, 1])
        with p_col: st.markdown(f'<div class="glass-panel glow-purple"><div class="panel-header">🎹 SUNO PROMPT</div><div class="prompt-container">{ai.get("suno_prompt")}</div></div>', unsafe_allow_html=True)
        with t_col: tips="".join([f'<div class="tip-item"><div class="tip-num">{i+1}</div><div>{t}</div></div>' for i,t in enumerate(ai.get("tips", []))]); st.markdown(f'<div class="glass-panel"><div class="panel-header">💡 TIPS</div>{tips}</div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        l1, l2 = st.columns(2)
        with l1:
            raw = st.text_area("Lyrics", height=200, placeholder="Paste lyrics...")
            if st.button("✨ Auto-Tag Lyrics"): st.session_state.formatted_lyrics = format_lyrics_with_tags(raw, ai)
        with l2:
            if st.session_state.formatted_lyrics: st.code(st.session_state.formatted_lyrics, language="markdown")
        
        if st.button("← ANALYZE NEW TRACK"): st.session_state.song_data = None; st.rerun()

if __name__ == "__main__":
    main()
