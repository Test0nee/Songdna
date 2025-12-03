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
    
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 5rem !important;
        max-width: 1100px !important;
    }
    
    header[data-testid="stHeader"], footer, [data-testid="stSidebar"] {display: none;}
    
    /* HERO SECTION */
    .hero-wrapper {
        position: relative;
        border-radius: 24px;
        overflow: hidden;
        margin-bottom: 25px;
        box-shadow: 0 20px 50px -20px rgba(0,0,0,0.7);
        border: 1px solid rgba(255,255,255,0.05);
        height: 380px;
    }
    .hero-bg { width: 100%; height: 100%; object-fit: cover; }
    .hero-overlay {
        position: absolute; inset: 0;
        background: linear-gradient(180deg, rgba(0,0,0,0) 0%, rgba(0,0,0,0.6) 50%, #050505 100%);
        display: flex; flex-direction: column; justify-content: flex-end; padding: 40px;
    }
    
    .verified-badge {
        background: rgba(56, 189, 248, 0.2); color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.4); padding: 6px 12px;
        border-radius: 100px; font-size: 0.75rem; font-weight: 700;
        text-transform: uppercase; display: inline-flex; align-items: center; gap: 6px;
        backdrop-filter: blur(10px);
    }
    .artist-title { font-size: 5rem; font-weight: 900; line-height: 0.9; margin: 10px 0; letter-spacing: -3px; background: linear-gradient(to right, #fff, #aaa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .song-subtitle { font-size: 2rem; color: #94a3b8; margin-bottom: 25px; letter-spacing: -1px; }
    .meta-pill { background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.1); padding: 8px 16px; border-radius: 50px; font-size: 0.85rem; color: #e2e8f0; margin-right: 10px; }

    /* GLASS PANELS */
    .glass-panel {
        background: #0a0a0a; border: 1px solid #1f1f1f;
        border-radius: 20px; padding: 24px; height: 100%; position: relative;
    }
    .glow-cyan { box-shadow: 0 0 40px -10px rgba(56, 189, 248, 0.1); border-top: 1px solid rgba(56, 189, 248, 0.2); }
    .glow-pink { box-shadow: 0 0 40px -10px rgba(236, 72, 153, 0.1); border-top: 1px solid rgba(236, 72, 153, 0.2); }
    .glow-purple { box-shadow: 0 0 40px -10px rgba(168, 85, 247, 0.1); border-top: 1px solid rgba(168, 85, 247, 0.2); }
    
    .panel-header { display: flex; align-items: center; gap: 10px; margin-bottom: 20px; font-size: 0.8rem; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; color: #64748b; }
    
    .stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
    .stat-box { background: rgba(255,255,255,0.03); padding: 15px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.03); }
    .stat-label { font-size: 0.7rem; color: #64748b; margin-bottom: 4px; letter-spacing: 1px; }
    .stat-value { font-size: 1.1rem; font-weight: 600; color: #fff; }
    .small-tag { font-size: 0.75rem; padding: 4px 10px; background: #1e1e1e; border-radius: 6px; color: #aaa; border: 1px solid #333; margin-right: 5px; }

    /* PROMPT BOX */
    .prompt-container { font-family: 'JetBrains Mono', monospace; background: #050505; border: 1px solid #333; color: #22d3ee; padding: 20px; border-radius: 12px; font-size: 0.9rem; line-height: 1.6; }
    .tip-item { display: flex; gap: 15px; margin-bottom: 15px; }
    .tip-num { background: #222; width: 24px; height: 24px; display: flex; align-items: center; justify-content: center; border-radius: 50%; font-size: 0.7rem; }
    
    /* LYRIC STUDIO */
    .lyric-area textarea { background: #080808 !important; color: #ccc !important; border: 1px solid #333 !important; font-family: 'Inter', sans-serif; }
    .lyric-output { background: #080808; border: 1px solid #333; padding: 20px; border-radius: 12px; font-family: 'JetBrains Mono', monospace; white-space: pre-wrap; color: #a78bfa; height: 300px; overflow-y: auto; }
    
    /* TAGS GUIDE */
    .tag-category { margin-bottom: 15px; }
    .tag-category h4 { color: #888; font-size: 0.8rem; text-transform: uppercase; margin-bottom: 8px; }
    .tag-chip { display: inline-block; background: #222; padding: 4px 8px; border-radius: 4px; font-size: 0.75rem; color: #ccc; margin: 0 4px 4px 0; border: 1px solid #333; cursor: pointer; }
    .tag-chip:hover { border-color: #666; color: #fff; }

    /* UPLOAD */
    .upload-area { border: 2px dashed #333; border-radius: 20px; padding: 60px; text-align: center; }
    </style>
""", unsafe_allow_html=True)

# --- 3. SUNO KNOWLEDGE BASE ---
SUNO_TAGS = {
    "Structure": ["[Intro]", "[Verse]", "[Chorus]", "[Pre-Chorus]", "[Bridge]", "[Outro]", "[Drop]", "[Build]", "[Instrumental Break]", "[Hook]", "[Solo]"],
    "Mood": ["Uplifting", "Melancholic", "Dark", "Euphoric", "Chill", "Aggressive", "Dreamy", "Nostalgic", "Epic", "Mysterious"],
    "Vocals": ["[Male Vocals]", "[Female Vocals]", "[Duet]", "[Choir]", "[Whispered]", "[Belting]", "[Auto-tune]", "[Spoken Word]", "[Rapping]"],
    "Production": ["[Reverb]", "[Delay]", "[Lo-fi]", "[Acoustic]", "[Synthesizer]", "[Bass Boosted]", "[Orchestral]", "[Minimal]"]
}

# --- 4. LOGIC & HELPERS ---

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
        "vocal_style": "Short description",
        "suno_prompt": "Genre, Tempo, Instruments, Vocal Style",
        "tips": ["Tip 1", "Tip 2", "Tip 3"]
    }}
    """
    try:
        response = model.generate_content(prompt)
        clean_text = response.text.replace('```json', '').replace('```', '').strip()
        return json.loads(clean_text)
    except:
        return None

def format_lyrics_with_tags(raw_lyrics, song_analysis):
    if not api_key: return "Please set API Key"
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
    except:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
    prompt = f"""
    Act as a Suno.ai Meta-Tagging Expert.
    
    CONTEXT:
    The user wants to create a song in the style of:
    Genre: {song_analysis.get('genre', 'Pop')}
    Mood: {song_analysis.get('mood', 'General')}
    
    OFFICIAL SUNO TAGS TO USE:
    Structure: {', '.join(SUNO_TAGS['Structure'])}
    Moods: {', '.join(SUNO_TAGS['Mood'])}
    Vocals: {', '.join(SUNO_TAGS['Vocals'])}
    
    TASK:
    Take the user's raw lyrics below and insert appropriate Suno Meta Tags (in square brackets) to structure the song exactly like the genre above.
    
    RULES:
    1. If it's EDM/Dance, use [Build] and [Drop] instead of Chorus where appropriate.
    2. If it's Hip Hop, use [Hook] and [Verse].
    3. Use vocal tags like [Whispered] or [Female Vocals] to guide the delivery.
    4. Keep the original lyrics intact, just add tags.
    
    USER LYRICS:
    "{raw_lyrics}"
    
    OUTPUT:
    Return ONLY the lyrics with the tags inserted.
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {e}"

# --- 5. MAIN APPLICATION ---
def main():
    if 'song_data' not in st.session_state:
        st.session_state.song_data = None
        st.session_state.analysis = None
    if 'formatted_lyrics' not in st.session_state:
        st.session_state.formatted_lyrics = ""

    # HEADER LOGO
    st.markdown("<h1 style='text-align:center; letter-spacing:-2px; margin-bottom:0;'>SUNOSONIC</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#555; font-size:0.8rem; letter-spacing:4px; margin-bottom:40px;'>AI AUDIO INTELLIGENCE</p>", unsafe_allow_html=True)

    # --- STATE 1: UPLOAD ---
    if not st.session_state.song_data:
        uploaded_file = st.file_uploader(" ", type=['mp3', 'wav', 'ogg'])
        
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

        # HERO BANNER
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
        
        st.audio(data.get('album_art') or "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3", format='audio/mp3')
        st.markdown("<br>", unsafe_allow_html=True)

        # ANALYSIS GRID
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
                <div class="glass-panel glow-cyan">
                    <div class="panel-header"><span style="color:#38bdf8">⚡</span> SONIC PROFILE</div>
                    <div class="stat-grid">
                        <div class="stat-box"><div class="stat-label">MOOD</div><div class="stat-value">{ai.get('mood', '-')}</div></div>
                         <div class="stat-box"><div class="stat-label">GENRE</div><div class="stat-value">{data['genre']}</div></div>
                    </div>
                    <div style="margin-top:15px">{''.join([f'<span class="small-tag">{inst}</span>' for inst in ai.get('instruments', [])])}</div>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
                <div class="glass-panel glow-pink">
                    <div class="panel-header"><span style="color:#ec4899">🎙</span> VOCAL ARCHITECTURE</div>
                    <div class="stat-grid">
                        <div class="stat-box"><div class="stat-label">TYPE</div><div class="stat-value">{ai.get('vocal_type', '-')}</div></div>
                         <div class="stat-box"><div class="stat-label">STYLE</div><div class="stat-value">Modern</div></div>
                    </div>
                    <div style="margin-top:15px; font-size:0.9rem; color:#ccc; font-style:italic;">"{ai.get('vocal_style', '-')}"</div>
                </div>
            """, unsafe_allow_html=True)
        
        # VISUALIZER
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="glass-panel" style="border-top: 1px solid rgba(255,255,255,0.1);"><div class="panel-header">🎼 STRUCTURAL DYNAMICS & RMS</div></div>', unsafe_allow_html=True)
        chart_data = pd.DataFrame(np.random.randn(80, 3), columns=['L', 'R', 'RMS'])
        st.area_chart(chart_data, height=120, color=["#38bdf8", "#ec4899", "#8b5cf6"])
        
        st.markdown("<br>", unsafe_allow_html=True)

        # PROMPT & TIPS
        p_col, t_col = st.columns([1.5, 1])
        with p_col:
            st.markdown(f'<div class="glass-panel glow-purple"><div class="panel-header"><span style="color:#a855f7">🎹</span> SUNO AI STYLE PROMPT</div><div class="prompt-container">{ai.get("suno_prompt", "Generating...")}</div></div>', unsafe_allow_html=True)

        with t_col:
            tips_html = "".join([f'<div class="tip-item"><div class="tip-num">{i+1}</div><div>{tip}</div></div>' for i, tip in enumerate(ai.get('tips', []))])
            st.markdown(f'<div class="glass-panel"><div class="panel-header">💡 PRO TIPS</div>{tips_html}</div>', unsafe_allow_html=True)

        # --- LYRIC STUDIO ---
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="glass-panel glow-purple" style="border: 1px solid #4c1d95;">', unsafe_allow_html=True)
        st.markdown('<div class="panel-header">📝 LYRIC STUDIO <span class="small-tag" style="margin-left:10px; color:#a78bfa; border-color:#a78bfa;">AUTO-TAGGER</span></div>', unsafe_allow_html=True)
        
        l_col1, l_col2 = st.columns(2)
        
        with l_col1:
            st.caption("PASTE RAW LYRICS HERE")
            raw_input = st.text_area("raw", height=300, placeholder="I walked down the street\nThe lights were low...", label_visibility="collapsed", key="raw_lyrics_input")
            
            # --- CHEAT SHEET ---
            with st.expander("📚 Suno Meta Tags Reference"):
                for cat, tags in SUNO_TAGS.items():
                    st.markdown(f"**{cat}**")
                    st.markdown(" ".join([f"`{t}`" for t in tags]))

            if st.button("✨ APPLY SUNO META TAGS", use_container_width=True):
                if raw_input:
                    with st.spinner("AI is structuring your lyrics based on the song style..."):
                        st.session_state.formatted_lyrics = format_lyrics_with_tags(raw_input, ai)
        
        with l_col2:
            st.caption("FORMATTED OUTPUT (READY FOR SUNO)")
            if st.session_state.formatted_lyrics:
                st.code(st.session_state.formatted_lyrics, language="markdown", line_numbers=False)
            else:
                st.markdown('<div class="lyric-output" style="color:#555; display:flex; align-items:center; justify-content:center;">Result will appear here...</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # RESET
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("⬅ ANALYZE NEW TRACK", use_container_width=True):
            st.session_state.song_data = None
            st.session_state.analysis = None
            st.session_state.formatted_lyrics = ""
            st.rerun()

if __name__ == "__main__":
    main()
