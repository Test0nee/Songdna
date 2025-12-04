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
from PIL import Image
from io import BytesIO

# --- CONFIGURATION ---
st.set_page_config(
    page_title="SunoSonic Studio",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- SPOTIFY AUTH ---
SPOTIFY_CLIENT_ID = st.secrets.get("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = st.secrets.get("SPOTIFY_CLIENT_SECRET")

def get_spotify_token():
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET: return None
    try:
        r = requests.post(
            "https://accounts.spotify.com/api/token", 
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={"grant_type": "client_credentials"},
            auth=(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET),
            timeout=5
        )
        if r.status_code == 200: return r.json().get("access_token")
    except: return None
    return None

# --- UI CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono:wght@400;700&display=swap');
    
    .stApp {
        background-color: #050505;
        background-image: radial-gradient(circle at 50% 0%, #1e1b4b 0%, #020617 60%);
        font-family: 'Inter', sans-serif;
        color: #fff;
    }
    
    .block-container { padding-top: 2rem; max-width: 1200px; }
    header, footer, [data-testid="stSidebar"] { display: none !important; }

    /* HERO */
    .hero-wrapper {
        position: relative; border-radius: 24px; overflow: hidden; margin-bottom: 30px;
        box-shadow: 0 25px 50px -12px rgba(0,0,0,0.7); height: 350px;
        background: #0f172a; border: 1px solid rgba(255,255,255,0.1);
    }
    .hero-bg { width: 100%; height: 100%; object-fit: cover; opacity: 0.6; filter: blur(20px) saturate(1.2); transform: scale(1.1); }
    .hero-overlay {
        position: absolute; inset: 0;
        background: linear-gradient(180deg, rgba(2,6,23,0) 0%, rgba(2,6,23,0.8) 60%, #020617 100%);
        display: flex; align-items: flex-end; padding: 40px;
    }
    .hero-content { width: 100%; display: flex; justify-content: space-between; align-items: flex-end; }
    .hero-text h1 { font-size: 4rem; font-weight: 900; margin: 0; line-height: 1; letter-spacing: -2px; text-shadow: 0 4px 20px rgba(0,0,0,0.5); }
    .hero-text h2 { font-size: 1.5rem; color: #94a3b8; margin: 10px 0 20px 0; font-weight: 400; }
    
    .pill {
        display: inline-block; padding: 6px 14px; border-radius: 50px;
        background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.1);
        font-size: 0.8rem; margin-right: 8px; backdrop-filter: blur(10px);
    }

    /* GLASS PANELS */
    .glass-panel {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px; padding: 24px;
        backdrop-filter: blur(12px);
        margin-bottom: 20px;
    }
    .panel-title { 
        font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1.5px; 
        color: #94a3b8; font-weight: 700; margin-bottom: 15px; display: flex; align-items: center; gap: 8px;
    }

    /* VISUALIZER CONTAINER (DAW STYLE) */
    .viz-container {
        background: #0b0f19;
        border: 1px solid #1e293b;
        border-radius: 16px;
        padding: 4px;
        box-shadow: inset 0 0 20px rgba(0,0,0,0.5);
    }

    /* GRID STATS */
    .stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
    .stat-card { background: rgba(255,255,255,0.03); padding: 15px; border-radius: 12px; }
    .stat-label { font-size: 0.7rem; color: #64748b; margin-bottom: 5px; }
    .stat-val { font-size: 1.1rem; font-weight: 600; }

    /* LYRICS */
    .stTextArea textarea { background: #0b0f19 !important; border: 1px solid #1e293b !important; color: #cbd5e1 !important; }
    .code-block { background: #0b0f19; padding: 20px; border-radius: 12px; font-family: 'JetBrains Mono', monospace; color: #a5b4fc; border: 1px solid #1e293b; }
    </style>
""", unsafe_allow_html=True)

# --- 3. KNOWLEDGE BASE ---
SUNO_TAGS = {
    "Structure": ["[Intro]", "[Verse]", "[Chorus]", "[Bridge]", "[Drop]", "[Build]", "[Outro]", "[Hook]"],
    "Mood": ["Uplifting", "Dark", "Euphoric", "Chill", "Aggressive", "Dreamy", "Epic"],
    "Vocals": ["[Male Vocals]", "[Female Vocals]", "[Duet]", "[Choir]", "[Whispered]", "[Belting]", "[Auto-tune]"]
}

# --- 4. HELPERS ---
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
    if not artist: return None
    # 1. Try Spotify
    clean_artist = artist.split(',')[0].strip()
    token = get_spotify_token()
    if token:
        try:
            url = f"https://api.spotify.com/v1/search?q={clean_artist}&type=artist&limit=1"
            r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=3)
            if r.status_code == 200:
                items = r.json().get("artists", {}).get("items", [])
                if items and items[0].get("images"):
                    return items[0]["images"][0]["url"]
        except: pass
    # 2. Fallback
    return "https://images.unsplash.com/photo-1470225620780-dba8ba36b745?w=1200"

# --- 5. AUDIO ENGINE (ROBUST) ---
def safe_load_audio(file_path):
    """ Tries multiple ways to load audio to bypass FFmpeg issues """
    errors = []
    # Method 1: Librosa Standard
    try:
        y, sr = librosa.load(file_path, sr=None, duration=180)
        return y, sr, None
    except Exception as e:
        errors.append(f"Librosa: {str(e)}")
    
    # Method 2: Soundfile (No FFmpeg needed for WAV/FLAC, fails on MP3)
    try:
        import soundfile as sf
        y, sr = sf.read(file_path)
        if len(y.shape) > 1: y = y.mean(axis=1) # Convert stereo to mono
        return y, sr, None
    except Exception as e:
        errors.append(f"Soundfile: {str(e)}")

    return None, None, " | ".join(errors)

def extract_audio_features(file_path):
    y, sr, error = safe_load_audio(file_path)
    
    if error:
        return {"success": False, "error": error}

    # Analysis
    duration = librosa.get_duration(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = round(tempo) if tempo > 0 else 120
    
    # Key
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    key = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"][np.argmax(np.sum(chroma, axis=1))]
    
    # Energy
    rms = librosa.feature.rms(y=y)[0]
    energy_score = np.mean(rms)
    energy = "High" if energy_score > 0.1 else "Mid" if energy_score > 0.05 else "Low"
    
    # Visualizer Data (Downsampled)
    hop = 512
    viz_rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    viz_rms = viz_rms / (np.max(viz_rms) + 1e-9) # Normalize 0-1
    
    # Structure Detection (Simple)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    peaks = librosa.util.peak_pick(onset_env, 3, 3, 3, 5, 0.5, 10)
    section_times = librosa.frames_to_time(peaks, sr=sr)
    
    # Filter sections (min 15s apart)
    filtered_sections = []
    last = 0
    for t in section_times:
        if t - last > 15:
            filtered_sections.append(t)
            last = t

    return {
        "success": True,
        "bpm": bpm,
        "key": key,
        "energy": energy,
        "duration": f"{int(duration//60)}:{int(duration%60):02d}",
        "waveform": viz_rms.tolist(),
        "sections": filtered_sections
    }

async def identify_song(file_path):
    shazam = Shazam()
    try:
        out = await shazam.recognize(file_path)
        if 'track' in out:
            return {
                "found": True,
                "title": out['track']['title'],
                "artist": out['track']['subtitle'],
                "img": out['track']['images'].get('coverart'),
                "genre": out['track']['genres']['primary']
            }
    except: pass
    return {"found": False}

def analyze_gemini(data):
    if not api_key: return None
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = f"""
    Analyze song: {data.get('title','Unknown')} by {data.get('artist','Unknown')}.
    Tech: {data.get('bpm')} BPM, Key {data.get('key')}, {data.get('energy')} Energy.
    Output JSON: {{ "mood": "...", "genre": "...", "instruments": ["..."], "vocal_type": "...", "vocal_style": "...", "suno_prompt": "...", "tips": ["..."] }}
    """
    try: return json.loads(model.generate_content(prompt).text.replace("```json","").replace("```",""))
    except: return None

def format_lyrics(raw, style):
    if not api_key: return "Error: No API Key"
    model = genai.GenerativeModel("gemini-2.5-flash")
    return model.generate_content(f"Add Suno tags {SUNO_TAGS} to these lyrics for a {style} song:\n{raw}").text

# --- 6. MAIN APP ---
def main():
    if 'data' not in st.session_state: st.session_state.data = None
    if 'ai' not in st.session_state: st.session_state.ai = None
    if 'lyrics' not in st.session_state: st.session_state.lyrics = ""

    st.markdown("<h1 style='text-align:center; letter-spacing:-2px;'>SUNOSONIC</h1>", unsafe_allow_html=True)

    # UPLOAD
    if not st.session_state.data:
        uploaded = st.file_uploader("Drop audio file", type=['mp3','wav','ogg'])
        if uploaded:
            with st.spinner("🎧 Decoding DNA..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                    tmp.write(uploaded.getvalue())
                    tmp_path = tmp.name
                
                # 1. Analyze Audio
                stats = extract_audio_features(tmp_path)
                if not stats['success']:
                    st.error(f"Analysis Failed: {stats['error']}")
                    st.info("Try installing FFmpeg on your system to fix this.")
                    os.remove(tmp_path)
                    return

                # 2. Identify
                meta = run_async(identify_song(tmp_path))
                
                # 3. Combine
                full_data = {**stats, **(meta if meta['found'] else {"title":"Unknown","artist":"Deep Scan","img":None,"genre":"Unknown"})}
                full_data['artist_bg'] = run_async(fetch_artist_image(full_data['artist']))
                
                st.session_state.data = full_data
                st.session_state.ai = analyze_gemini(full_data)
                os.remove(tmp_path)
                st.rerun()

    # DASHBOARD
    else:
        d = st.session_state.data
        ai = st.session_state.ai or {}
        
        # HERO
        bg = d.get('artist_bg') or "https://images.unsplash.com/photo-1470225620780-dba8ba36b745"
        st.markdown(f"""
            <div class="hero-wrapper">
                <img src="{bg}" class="hero-bg">
                <div class="hero-overlay">
                    <div class="hero-content">
                        <div class="hero-text">
                            <div class="pill">✓ VERIFIED ANALYSIS</div>
                            <h1>{d['artist']}</h1>
                            <h2>{d['title']}</h2>
                            <div>
                                <span class="pill">🎵 {ai.get('genre','Unknown')}</span>
                                <span class="pill">⏱ {d['bpm']} BPM</span>
                                <span class="pill">🎹 {d['key']}</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # GRIDS
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
                <div class="glass-panel">
                    <div class="panel-title">⚡ SONIC PROFILE</div>
                    <div class="stat-grid">
                        <div class="stat-card"><div class="stat-label">MOOD</div><div class="stat-val">{ai.get('mood','-')}</div></div>
                        <div class="stat-card"><div class="stat-label">ENERGY</div><div class="stat-val">{d['energy']}</div></div>
                    </div>
                    <div style="margin-top:15px">{' '.join([f'<span class="pill" style="font-size:0.7rem">{i}</span>' for i in ai.get('instruments',[])])}</div>
                </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
                <div class="glass-panel">
                    <div class="panel-title">🎙 VOCAL PROFILE</div>
                    <div class="stat-grid">
                        <div class="stat-card"><div class="stat-label">TYPE</div><div class="stat-val">{ai.get('vocal_type','-')}</div></div>
                        <div class="stat-card"><div class="stat-label">STYLE</div><div class="stat-val">{ai.get('vocal_style','-')}</div></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # DAW VISUALIZER
        st.markdown('<div class="panel-title" style="margin-left:5px">📈 STRUCTURAL DYNAMICS</div>', unsafe_allow_html=True)
        
        # Prepare Plotly Data
        y = np.array(d['waveform'])
        x = np.linspace(0, 100, len(y))
        
        fig = go.Figure()
        # Main Waveform (Fill)
        fig.add_trace(go.Scatter(x=x, y=y, fill='tozeroy', line=dict(color='#22d3ee', width=1), name='Energy'))
        # Mirror for "Stereo" look
        fig.add_trace(go.Scatter(x=x, y=-y, fill='tozeroy', line=dict(color='#818cf8', width=1), name='Stereo'))
        
        # Section Markers
        for i, sec in enumerate(d['sections']):
            sec_x = (sec / (len(y)*512/22050)) * 100 # Approx scaling
            if sec_x > 100: break
            fig.add_vline(x=sec_x, line_width=1, line_dash="dot", line_color="rgba(255,255,255,0.3)")
            fig.add_annotation(x=sec_x, y=0.8, text=f"SEC {i+1}", showarrow=False, font=dict(color="#ec4899", size=10))

        fig.update_layout(
            height=200, margin=dict(l=0,r=0,t=20,b=0),
            paper_bgcolor="#0b0f19", plot_bgcolor="#0b0f19",
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', showticklabels=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', showticklabels=False, range=[-1.1, 1.1]),
            showlegend=False, hovermode="x unified"
        )
        
        # Containerize Plotly
        st.markdown('<div class="viz-container">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # PROMPT & LYRICS
        c3, c4 = st.columns([1.5, 1])
        with c3:
            st.markdown(f'<div class="glass-panel"><div class="panel-title">🎹 SUNO PROMPT</div><div class="code-block">{ai.get("suno_prompt","...")}</div></div>', unsafe_allow_html=True)
        with c4:
            tips = "".join([f"<li style='margin-bottom:8px; color:#94a3b8'>{t}</li>" for t in ai.get("tips",[])])
            st.markdown(f'<div class="glass-panel"><div class="panel-title">💡 TIPS</div><ul>{tips}</ul></div>', unsafe_allow_html=True)

        # LYRIC TAGGER
        st.markdown('<div class="glass-panel" style="border-color:#4f46e5">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">📝 LYRIC TAGGER</div>', unsafe_allow_html=True)
        l1, l2 = st.columns(2)
        with l1:
            raw = st.text_area("Input Lyrics", height=250, placeholder="Paste your lyrics here...")
            if st.button("✨ Auto-Structure Lyrics", use_container_width=True):
                st.session_state.lyrics = format_lyrics(raw, ai.get('genre'))
        with l2:
            if st.session_state.lyrics: st.code(st.session_state.lyrics, language="markdown")
            else: st.info("Result will appear here")
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("RESET"):
            st.session_state.data = None
            st.rerun()

if __name__ == "__main__":
    main()
