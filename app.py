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

# --- 2. MODERN UI (CSS) ---
st.markdown("""
    <style>
    /* RESET & FONTS */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    .stApp {
        background-color: #050505;
        background-image: radial-gradient(circle at 50% 0%, #1a1a2e 0%, #000000 70%);
        font-family: 'Inter', sans-serif;
    }
    
    /* REMOVE DEFAULT PADDING */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 5rem !important;
        max-width: 1200px !important;
    }
    
    /* HIDE ELEMENTS */
    header[data-testid="stHeader"] {display: none;}
    footer {display: none;}
    [data-testid="stSidebar"] {display: none;}
    
    /* GLASS CARDS */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 24px -1px rgba(0, 0, 0, 0.2);
    }
    
    /* HERO SECTION */
    .hero-container {
        position: relative;
        width: 100%;
        height: 300px;
        border-radius: 24px;
        overflow: hidden;
        margin-bottom: 30px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .hero-bg {
        width: 100%;
        height: 100%;
        object-fit: cover;
        opacity: 0.6;
        filter: blur(0px) brightness(0.7);
    }
    .hero-overlay {
        position: absolute;
        bottom: 0;
        left: 0;
        width: 100%;
        padding: 40px;
        background: linear-gradient(to top, rgba(0,0,0,0.9), transparent);
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
    }
    
    /* TYPOGRAPHY */
    .artist-name {
        font-size: 3.5rem;
        font-weight: 800;
        color: white;
        margin: 0;
        letter-spacing: -2px;
        line-height: 1;
        text-shadow: 0 4px 12px rgba(0,0,0,0.5);
    }
    .song-title {
        font-size: 1.5rem;
        color: #cccccc;
        font-weight: 400;
        margin-top: 5px;
        margin-bottom: 20px;
    }
    
    /* BADGES */
    .badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 50px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-right: 8px;
        letter-spacing: 1px;
    }
    .badge-blue { background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }
    .badge-purple { background: rgba(139, 92, 246, 0.2); color: #a78bfa; border: 1px solid rgba(139, 92, 246, 0.3); }
    .badge-pink { background: rgba(236, 72, 153, 0.2); color: #f472b6; border: 1px solid rgba(236, 72, 153, 0.3); }
    
    /* SECTIONS HEADERS */
    .section-header {
        font-size: 0.8rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 700;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* DATA BOXES */
    .data-box {
        background: rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 16px;
        height: 100%;
    }
    .data-label { font-size: 0.7rem; color: #888; text-transform: uppercase; margin-bottom: 5px; }
    .data-value { font-size: 1.1rem; color: white; font-weight: 600; }
    
    /* CUSTOM UPLOAD */
    .stFileUploader > div > div {
        background-color: rgba(255,255,255,0.02);
        border: 1px dashed rgba(255,255,255,0.1);
        border-radius: 16px;
    }
    
    /* SUNO PROMPT BOX */
    .prompt-box {
        font-family: 'Courier New', monospace;
        background: #0a0a0a;
        border-left: 3px solid #a855f7;
        padding: 15px;
        color: #e2e8f0;
        font-size: 0.9rem;
        line-height: 1.5;
        border-radius: 4px;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. HELPER LOGIC ---

# Setup Gemini
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
    if not artist: return "https://images.unsplash.com/photo-1470225620780-dba8ba36b745?w=800&q=80"
    clean_artist = artist.split(',')[0].split('&')[0].split('feat')[0].strip()
    try:
        url = f"https://api.deezer.com/search/artist?q={clean_artist}"
        r = requests.get(url, timeout=5)
        data = r.json()
        if 'data' in data and data['data']:
            return data['data'][0]['picture_xl']
    except:
        return "https://images.unsplash.com/photo-1493225255756-d9584f8606e9?w=800"
    return "https://images.unsplash.com/photo-1493225255756-d9584f8606e9?w=800"

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
                "genre": track.get('genres', {}).get('primary', 'Pop')
            }
        return {"found": False}
    except Exception as e:
        return {"found": False, "error": str(e)}

def analyze_gemini_json(song_data):
    if not api_key: return None
    
    # Using 2.5 Flash for speed, fallback to 2.0 Flash Lite or Pro if needed
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
    except:
        model = genai.GenerativeModel('gemini-1.5-flash')

    # We ask for JSON specifically to populate our new UI cards
    prompt = f"""
    Analyze the song "{song_data['title']}" by "{song_data['artist']}".
    Return valid JSON ONLY. No markdown formatting.
    Structure:
    {{
        "mood": "One or two words (e.g. Uplifting, Dark)",
        "key_tempo": "e.g. C Minor, 128 BPM",
        "instruments": ["Instrument 1", "Instrument 2", "Instrument 3"],
        "vocal_count": "Solo / Duet",
        "vocal_gender": "Male / Female / Mixed",
        "vocal_texture": "One short sentence description.",
        "suno_prompt": "Genre, Tempo, Key Instruments, Vocal Style (One specific line)",
        "production_tips": ["Tip 1", "Tip 2", "Tip 3"]
    }}
    """
    try:
        response = model.generate_content(prompt)
        # Clean response to ensure it's pure JSON
        clean_text = response.text.replace('```json', '').replace('```', '').strip()
        return json.loads(clean_text)
    except Exception as e:
        st.error(f"AI Brain Error: {e}")
        return None

# --- 4. MAIN APP ---
def main():
    # Header
    st.markdown("<h1 style='text-align: center; margin-bottom: 5px;'>SUNOSONIC</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666; letter-spacing: 4px; font-size: 0.8rem; margin-bottom: 40px;'>AI AUDIO INTELLIGENCE STUDIO</p>", unsafe_allow_html=True)

    if 'song_data' not in st.session_state:
        st.session_state.song_data = None
    if 'analysis' not in st.session_state:
        st.session_state.analysis = None

    # --- INPUT STATE ---
    if not st.session_state.song_data:
        uploaded_file = st.file_uploader(" ", type=['mp3', 'wav', 'ogg'])
        
        if uploaded_file:
            with st.spinner("🎧 DECODING AUDIO DNA..."):
                # 1. Save Temp
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                
                # 2. Identify
                result = run_async(identify_song(tmp_path))
                os.remove(tmp_path)
                
                if result['found']:
                    # 3. Get Hero Image
                    result['artist_bg'] = run_async(fetch_artist_image(result['artist']))
                    st.session_state.song_data = result
                    
                    # 4. Get AI Analysis (JSON)
                    st.session_state.analysis = analyze_gemini_json(result)
                    st.rerun()
                else:
                    st.error("Could not identify track. Try a longer clip.")

    # --- RESULT STATE ---
    else:
        data = st.session_state.song_data
        ai = st.session_state.analysis
        
        # 1. TOP HERO SECTION
        st.markdown(f"""
            <div class="hero-container">
                <img src="{data['artist_bg']}" class="hero-bg">
                <div class="hero-overlay">
                    <div style="margin-bottom: 10px;">
                        <span class="badge badge-blue">✓ VERIFIED ARTIST</span>
                    </div>
                    <h1 class="artist-name">{data['artist']}</h1>
                    <h2 class="song-title">{data['title']}</h2>
                    <div>
                        <span class="badge badge-purple">{data['genre']}</span>
                        <span class="badge badge-pink">{ai.get('key_tempo', 'Analyzing...') if ai else '...'}</span>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Audio Player
        st.audio(data.get('album_art') or "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3", format="audio/mp3") # Placeholder if no local file

        if ai:
            # 2. SONIC & VOCAL GRID
            c1, c2 = st.columns(2)
            
            with c1:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-header">⚡ SONIC PROFILE</div>', unsafe_allow_html=True)
                
                sc1, sc2 = st.columns(2)
                with sc1:
                    st.markdown(f"""
                        <div class="data-box">
                            <div class="data-label">MOOD</div>
                            <div class="data-value">{ai.get('mood', '-')}</div>
                        </div>
                    """, unsafe_allow_html=True)
                with sc2:
                    st.markdown(f"""
                        <div class="data-box">
                            <div class="data-label">KEY/TEMPO</div>
                            <div class="data-value">{ai.get('key_tempo', '-')}</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                # Instruments Tags
                inst_html = "".join([f'<span class="badge badge-blue" style="margin-bottom:5px;">{inst}</span>' for inst in ai.get('instruments', [])])
                st.markdown(f"<div>{inst_html}</div>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with c2:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-header">🎙️ VOCAL ARCHITECTURE</div>', unsafe_allow_html=True)
                
                vc1, vc2 = st.columns(2)
                with vc1:
                     st.markdown(f"""
                        <div class="data-box">
                            <div class="data-label">COUNT</div>
                            <div class="data-value">{ai.get('vocal_count', '-')}</div>
                        </div>
                    """, unsafe_allow_html=True)
                with vc2:
                     st.markdown(f"""
                        <div class="data-box">
                            <div class="data-label">GENDER</div>
                            <div class="data-value">{ai.get('vocal_gender', '-')}</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"""
                    <div class="data-box" style="border:none; background:rgba(255,255,255,0.03);">
                        <div class="data-label">VOCAL TEXTURE ANALYSIS</div>
                        <div style="font-style: italic; color: #ccc;">"{ai.get('vocal_texture', '-')}"</div>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # 3. WAVEFORM VISUALIZER (Simulated for aesthetics)
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">🎼 STRUCTURAL DYNAMICS & RMS</div>', unsafe_allow_html=True)
            
            chart_data = pd.DataFrame(
                np.random.randn(50, 3),
                columns=['a', 'b', 'c'])
            # Create a cool neon area chart
            st.area_chart(chart_data, height=120, color=["#3b82f6", "#8b5cf6", "#ec4899"])
            st.markdown('</div>', unsafe_allow_html=True)

            # 4. PROMPT & TIPS
            c3, c4 = st.columns([1.5, 1])
            
            with c3:
                st.markdown('<div class="glass-card" style="height: 100%;">', unsafe_allow_html=True)
                st.markdown('<div class="section-header">🎹 SUNO AI STYLE PROMPT</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="prompt-box">{ai.get("suno_prompt", "Generating...")}</div>', unsafe_allow_html=True)
                st.caption("Copy this directly into Suno v3 for best results.")
                st.markdown('</div>', unsafe_allow_html=True)
                
            with c4:
                st.markdown('<div class="glass-card" style="height: 100%;">', unsafe_allow_html=True)
                st.markdown('<div class="section-header">💡 TIPS & TRICKS</div>', unsafe_allow_html=True)
                tips_html = "".join([f'<li style="margin-bottom:8px; color:#ddd; font-size:0.9rem;">{tip}</li>' for tip in ai.get('production_tips', [])])
                st.markdown(f'<ul style="padding-left: 20px;">{tips_html}</ul>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        # 5. LYRIC STUDIO (Visual Only)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">📝 LYRIC STUDIO <span class="badge badge-purple" style="margin-left:10px; font-size:0.5rem">SUNO COMPATIBLE</span></div>', unsafe_allow_html=True)
        lc1, lc2 = st.columns([3, 1])
        with lc1:
            st.text_input("Enter lyric theme", placeholder="Ex: A cyber-noir detective story set in Tokyo...")
        with lc2:
            st.markdown("<br>", unsafe_allow_html=True)
            st.button("✨ GENERATE LYRICS", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Reset
        if st.button("← ANALYZE NEW TRACK", type="secondary"):
            st.session_state.song_data = None
            st.session_state.analysis = None
            st.rerun()

if __name__ == "__main__":
    main()
