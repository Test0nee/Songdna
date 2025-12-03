import streamlit as st
import asyncio
from shazamio import Shazam
import google.generativeai as genai
import tempfile
import os
import requests
import json
import random
import pandas as pd
import numpy as np

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
    /* IMPORT FONTS */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700;900&family=JetBrains+Mono:wght@400;700&display=swap');
    
    /* GLOBAL THEME */
    .stApp {
        background-color: #050505;
        background-image: 
            radial-gradient(at 0% 0%, rgba(56, 189, 248, 0.08) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(236, 72, 153, 0.08) 0px, transparent 50%);
        font-family: 'Inter', sans-serif;
        color: #ffffff;
    }
    
    /* RESET STREAMLIT PADDING */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 5rem !important;
        max-width: 1100px !important;
    }
    
    /* HIDE STREAMLIT ELEMENTS */
    header[data-testid="stHeader"] {display: none;}
    footer {display: none;}
    [data-testid="stSidebar"] {display: none;}
    
    /* -----------------------
       HERO SECTION 
       ----------------------- */
    .hero-wrapper {
        position: relative;
        border-radius: 24px;
        overflow: hidden;
        margin-bottom: 25px;
        box-shadow: 0 20px 50px -20px rgba(0,0,0,0.7);
        border: 1px solid rgba(255,255,255,0.05);
        height: 380px;
    }
    
    .hero-bg {
        width: 100%;
        height: 100%;
        object-fit: cover;
        transition: transform 0.5s ease;
    }
    
    .hero-overlay {
        position: absolute;
        inset: 0;
        background: linear-gradient(180deg, rgba(0,0,0,0) 0%, rgba(0,0,0,0.6) 50%, #050505 100%);
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
        padding: 40px;
    }
    
    .verified-badge {
        background: rgba(56, 189, 248, 0.2);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.4);
        padding: 6px 12px;
        border-radius: 100px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        margin-bottom: 15px;
        backdrop-filter: blur(10px);
    }
    
    .artist-title {
        font-size: 5rem;
        font-weight: 900;
        line-height: 0.9;
        margin-bottom: 5px;
        letter-spacing: -3px;
        background: linear-gradient(to right, #fff, #aaa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .song-subtitle {
        font-size: 2rem;
        font-weight: 400;
        color: #94a3b8;
        margin-bottom: 25px;
        letter-spacing: -1px;
    }
    
    .meta-tags {
        display: flex;
        gap: 10px;
        align-items: center;
    }
    
    .meta-pill {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.1);
        padding: 8px 16px;
        border-radius: 50px;
        font-size: 0.85rem;
        color: #e2e8f0;
        backdrop-filter: blur(5px);
    }
    
    /* -----------------------
       GLASS CARDS
       ----------------------- */
    .glass-panel {
        background: #0a0a0a;
        border: 1px solid #1f1f1f;
        border-radius: 20px;
        padding: 24px;
        height: 100%;
        position: relative;
        overflow: hidden;
    }
    
    .glow-cyan { box-shadow: 0 0 40px -10px rgba(56, 189, 248, 0.1); border-top: 1px solid rgba(56, 189, 248, 0.2); }
    .glow-pink { box-shadow: 0 0 40px -10px rgba(236, 72, 153, 0.1); border-top: 1px solid rgba(236, 72, 153, 0.2); }
    .glow-purple { box-shadow: 0 0 40px -10px rgba(168, 85, 247, 0.1); border-top: 1px solid rgba(168, 85, 247, 0.2); }
    
    .panel-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #64748b;
    }
    
    .stat-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 15px;
    }
    
    .stat-box {
        background: rgba(255,255,255,0.03);
        padding: 15px;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.03);
    }
    
    .stat-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 4px;
        letter-spacing: 1px;
    }
    
    .stat-value {
        font-size: 1.1rem;
        font-weight: 600;
        color: #fff;
    }
    
    .tag-container {
        margin-top: 20px;
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }
    
    .small-tag {
        font-size: 0.75rem;
        padding: 4px 10px;
        background: #1e1e1e;
        border-radius: 6px;
        color: #aaa;
        border: 1px solid #333;
    }
    
    /* -----------------------
       PROMPT & TIPS
       ----------------------- */
    .prompt-container {
        font-family: 'JetBrains Mono', monospace;
        background: #050505;
        border: 1px solid #333;
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
        align-items: flex-start;
    }
    .tip-num {
        background: #222;
        color: #fff;
        width: 24px;
        height: 24px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        font-size: 0.7rem;
        flex-shrink: 0;
    }
    .tip-text {
        color: #a1a1aa;
        font-size: 0.9rem;
        line-height: 1.4;
    }
    
    /* -----------------------
       UPLOAD ZONE
       ----------------------- */
    .upload-area {
        border: 2px dashed #333;
        border-radius: 20px;
        padding: 60px;
        text-align: center;
        background: rgba(255,255,255,0.01);
        transition: all 0.3s ease;
    }
    .upload-area:hover {
        border-color: #555;
        background: rgba(255,255,255,0.02);
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. LOGIC & HELPERS ---

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
    if not artist: return "https://images.unsplash.com/photo-1514525253440-b393452e8d26?w=1200"
    clean_artist = artist.split(',')[0].split('&')[0].split('feat')[0].strip()
    try:
        # Use Deezer API to get artist picture
        url = f"https://api.deezer.com/search/artist?q={clean_artist}"
        r = requests.get(url, timeout=5)
        data = r.json()
        if 'data' in data and data['data']:
            return data['data'][0]['picture_xl']
    except:
        return "https://images.unsplash.com/photo-1514525253440-b393452e8d26?w=1200"
    return "https://images.unsplash.com/photo-1514525253440-b393452e8d26?w=1200"

async def identify_song(file_path):
    shazam = Shazam()
    try:
        out = await shazam.recognize(file_path)
        if 'track' in out:
            track = out['track']
            return {
                "found": True,
                "title": track.get('title'),
                "artist": track.get('subtitle'),
                "album_art": track.get('images', {}).get('coverart'),
                "genre": track.get('genres', {}).get('primary', 'Electronic')
            }
        return {"found": False}
    except Exception as e:
        return {"found": False, "error": str(e)}

def analyze_gemini_json(song_data):
    if not api_key: return None
    
    # Try 2.5, fallback to 1.5
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
    except:
        model = genai.GenerativeModel('gemini-1.5-flash')

    prompt = f"""
    Analyze the song "{song_data['title']}" by "{song_data['artist']}".
    Return pure JSON. No markdown.
    {{
        "mood": "Single Word (e.g. Euphoric)",
        "key": "e.g. C# Minor",
        "tempo": "e.g. 128 BPM",
        "instruments": ["Synth", "Bass", "Drums"],
        "vocal_type": "Male / Female / Choir",
        "vocal_style": "Short description (e.g. Reverb-heavy)",
        "suno_prompt": "Genre, Tempo, Instruments, Vocal Style (One line for AI generation)",
        "tips": ["Production tip 1", "Production tip 2", "Production tip 3"]
    }}
    """
    try:
        response = model.generate_content(prompt)
        clean_text = response.text.replace('```json', '').replace('```', '').strip()
        return json.loads(clean_text)
    except:
        return None

# --- 4. MAIN APPLICATION ---
def main():
    if 'song_data' not in st.session_state:
        st.session_state.song_data = None
        st.session_state.analysis = None

    # HEADER LOGO
    st.markdown("<h1 style='text-align:center; letter-spacing:-2px; margin-bottom:0;'>SUNOSONIC</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#555; font-size:0.8rem; letter-spacing:4px; margin-bottom:40px;'>AI AUDIO INTELLIGENCE</p>", unsafe_allow_html=True)

    # --- STATE 1: UPLOAD ---
    if not st.session_state.song_data:
        uploaded_file = st.file_uploader(" ", type=['mp3', 'wav', 'ogg'])
        
        # Custom Placeholder Text using Markdown above the invisible uploader labels
        if not uploaded_file:
            st.info("👆 DROP AUDIO FILE ABOVE TO BEGIN")

        if uploaded_file:
            with st.spinner("🎧 ANALYZING AUDIO SPECTRUM..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                
                result = run_async(identify_song(tmp_path))
                os.remove(tmp_path)
                
                if result['found']:
                    result['artist_bg'] = run_async(fetch_artist_image(result['artist']))
                    st.session_state.song_data = result
                    st.session_state.analysis = analyze_gemini_json(result)
                    st.rerun()
                else:
                    st.error("Could not identify track. Please try a clearer snippet.")

    # --- STATE 2: DASHBOARD ---
    else:
        data = st.session_state.song_data
        ai = st.session_state.analysis or {}

        # 1. HERO BANNER
        st.markdown(f"""
            <div class="hero-wrapper">
                <img src="{data['artist_bg']}" class="hero-bg">
                <div class="hero-overlay">
                    <div><span class="verified-badge">✓ Verified Artist</span></div>
                    <div class="artist-title">{data['artist']}</div>
                    <div class="song-subtitle">{data['title']}</div>
                    <div class="meta-tags">
                        <span class="meta-pill">🎵 {data['genre']}</span>
                        <span class="meta-pill">⏱ {ai.get('tempo', '-- BPM')}</span>
                        <span class="meta-pill">🎹 {ai.get('key', '--')}</span>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Audio Player (Standard Streamlit)
        st.audio(data.get('album_art') or "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3", format='audio/mp3')

        st.markdown("<br>", unsafe_allow_html=True)

        # 2. ANALYSIS GRID
        col1, col2 = st.columns(2)
        
        # LEFT: SONIC PROFILE (Cyan Theme)
        with col1:
            st.markdown(f"""
                <div class="glass-panel glow-cyan">
                    <div class="panel-header">
                        <span style="color:#38bdf8">⚡</span> SONIC PROFILE
                    </div>
                    <div class="stat-grid">
                        <div class="stat-box">
                            <div class="stat-label">MOOD</div>
                            <div class="stat-value">{ai.get('mood', '-')}</div>
                        </div>
                         <div class="stat-box">
                            <div class="stat-label">GENRE</div>
                            <div class="stat-value">{data['genre']}</div>
                        </div>
                    </div>
                    <div class="tag-container">
                        {''.join([f'<span class="small-tag">{inst}</span>' for inst in ai.get('instruments', [])])}
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # RIGHT: VOCAL ARCHITECTURE (Pink Theme)
        with col2:
            st.markdown(f"""
                <div class="glass-panel glow-pink">
                    <div class="panel-header">
                        <span style="color:#ec4899">🎙</span> VOCAL ARCHITECTURE
                    </div>
                    <div class="stat-grid">
                        <div class="stat-box">
                            <div class="stat-label">TYPE</div>
                            <div class="stat-value">{ai.get('vocal_type', '-')}</div>
                        </div>
                         <div class="stat-box">
                            <div class="stat-label">PROCESSING</div>
                            <div class="stat-value">Modern</div>
                        </div>
                    </div>
                    <div style="margin-top:20px; font-size:0.9rem; color:#ccc; font-style:italic;">
                        "{ai.get('vocal_style', 'Analysis pending...')}"
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)

        # 3. PROMPT & TIPS ROW
        p_col, t_col = st.columns([1.5, 1])
        
        with p_col:
            st.markdown(f"""
                <div class="glass-panel glow-purple">
                    <div class="panel-header">
                        <span style="color:#a855f7">🎹</span> SUNO AI STYLE PROMPT
                    </div>
                    <div class="prompt-container">
                        {ai.get('suno_prompt', 'Generating prompt...')}
                    </div>
                    <div style="margin-top:10px; font-size:0.75rem; color:#666;">
                        COPY THIS PROMPT DIRECTLY INTO SUNO V3
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with t_col:
            tips_html = ""
            for i, tip in enumerate(ai.get('tips', [])):
                tips_html += f"""
                <div class="tip-item">
                    <div class="tip-num">{i+1}</div>
                    <div class="tip-text">{tip}</div>
                </div>
                """
            
            st.markdown(f"""
                <div class="glass-panel">
                    <div class="panel-header">💡 PRO TIPS</div>
                    {tips_html}
                </div>
            """, unsafe_allow_html=True)

        # 4. LYRIC STUDIO PLACEHOLDER
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
            <div class="glass-panel" style="border: 1px dashed #333; text-align:center; padding: 40px;">
                <h3 style="margin:0; color:#fff;">LYRIC STUDIO</h3>
                <p style="color:#666; font-size:0.8rem; margin-bottom:20px;">GENERATE LYRICS MATCHING THIS STYLE</p>
                <div style="display:inline-block; padding: 10px 20px; background:#222; border-radius:8px; color:#555; font-size:0.8rem;">
                    COMING SOON
                </div>
            </div>
        """, unsafe_allow_html=True)

        # 5. RESET BUTTON
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("⬅ ANALYZE NEW TRACK", use_container_width=True):
            st.session_state.song_data = None
            st.session_state.analysis = None
            st.rerun()

if __name__ == "__main__":
    main()
