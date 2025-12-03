import streamlit as st
import asyncio
from shazamio import Shazam
import google.generativeai as genai
import tempfile
import os
import requests

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="SunoSonic",
    page_icon="🎹",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. SUPERIOR UI (CSS) ---
st.markdown("""
    <style>
    /* 1. FORCE DARK THEME & RESET */
    .stApp {
        background-color: #000000;
        color: #FFFFFF;
    }
    
    /* 2. CENTER CONTENT CORRECTLY */
    .block-container {
        padding-top: 5vh !important;
        padding-bottom: 5vh !important;
        max-width: 900px !important;
    }
    
    /* 3. HIDE JUNK */
    header[data-testid="stHeader"] {display: none;}
    footer {display: none;}
    #MainMenu {visibility: hidden;}
    [data-testid="stSidebar"] {display: none;}

    /* 4. KINETIC BACKGROUND */
    .kinetic-wrapper {
        position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
        z-index: 0; overflow: hidden;
        display: flex; flex-direction: column; justify-content: center; gap: 20px;
        opacity: 0.4; filter: grayscale(100%) contrast(1.1); pointer-events: none;
    }
    .marquee-row { display: flex; gap: 20px; width: 200vw; }
    .marquee-item {
        width: 300px; height: 180px; background-color: #222;
        border-radius: 12px; background-size: cover; background-position: center;
        flex-shrink: 0; border: 1px solid rgba(255,255,255,0.05);
        opacity: 0.7; box-shadow: 0 4px 30px rgba(0,0,0,0.5);
    }
    
    /* ANIMATIONS */
    @keyframes scrollLeft { from {transform: translateX(0);} to {transform: translateX(-50%);} }
    @keyframes scrollRight { from {transform: translateX(-50%);} to {transform: translateX(0);} }
    .scroll-left { animation: scrollLeft 50s linear infinite; }
    .scroll-right { animation: scrollRight 50s linear infinite; }

    /* 5. GLASS PANEL */
    .glass-panel {
        position: relative; z-index: 10;
        background: rgba(18, 18, 18, 0.85);
        backdrop-filter: blur(40px); -webkit-backdrop-filter: blur(40px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 24px; padding: 60px;
        box-shadow: 0 20px 50px rgba(0,0,0,0.5);
        text-align: center;
        transition: all 0.5s ease;
    }

    /* 6. TYPOGRAPHY & ELEMENTS */
    h1 {
        font-family: 'Helvetica Neue', sans-serif; font-weight: 800; letter-spacing: -2px;
        font-size: 3.5rem !important; color: white; margin-bottom: 10px;
    }
    .subtitle {
        color: #888; font-family: monospace; letter-spacing: 4px; 
        margin-bottom: 50px; text-transform: uppercase; font-size: 0.8rem;
    }
    
    /* Uploader */
    .stFileUploader { 
        padding: 30px; 
        border: 2px dashed rgba(255,255,255,0.1); 
        border-radius: 16px; 
        background: rgba(255,255,255,0.02);
        transition: border 0.3s ease;
    }
    .stFileUploader:hover { border-color: rgba(255,255,255,0.4); }
    
    /* Buttons */
    div.stButton > button { 
        width: 100%; border-radius: 12px; font-weight: bold; 
        text-transform: uppercase; letter-spacing: 2px;
        background: #ffffff; color: black; border: none; padding: 18px; 
        transition: all 0.3s ease;
        margin-top: 20px;
    }
    div.stButton > button:hover { 
        transform: translateY(-2px); 
        box-shadow: 0 10px 30px rgba(255,255,255,0.15);
        background: #f0f0f0;
    }
    
    /* Images */
    img { border-radius: 12px; box-shadow: 0 10px 40px rgba(0,0,0,0.5); }
    
    /* Info Box */
    .stAlert { background-color: rgba(255,255,255,0.05); border: none; color: #ccc; }
    </style>
    
    <div class="kinetic-wrapper">
        <div class="marquee-row scroll-left">
            <div class="marquee-item" style="background-image: url('https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?w=500')"></div>
            <div class="marquee-item" style="background-image: url('https://images.unsplash.com/photo-1493225255756-d9584f8606e9?w=500')"></div>
            <div class="marquee-item" style="background-image: url('https://images.unsplash.com/photo-1514525253440-b393452e8d26?w=500')"></div>
            <div class="marquee-item" style="background-image: url('https://images.unsplash.com/photo-1470225620780-dba8ba36b745?w=500')"></div>
            <div class="marquee-item" style="background-image: url('https://images.unsplash.com/photo-1511379938547-c1f69419868d?w=500')"></div>
        </div>
        <div class="marquee-row scroll-right">
            <div class="marquee-item" style="background-image: url('https://images.unsplash.com/photo-1501612780327-45045538702b?w=500')"></div>
            <div class="marquee-item" style="background-image: url('https://images.unsplash.com/photo-1459749411177-0473ef71607b?w=500')"></div>
            <div class="marquee-item" style="background-image: url('https://images.unsplash.com/photo-1498038432885-c6f3f1b912ee?w=500')"></div>
            <div class="marquee-item" style="background-image: url('https://images.unsplash.com/photo-1520523839897-bd0b52f945a0?w=500')"></div>
        </div>
         <div class="marquee-row scroll-left">
            <div class="marquee-item" style="background-image: url('https://images.unsplash.com/photo-1511735111813-97415a4ed839?w=500')"></div>
            <div class="marquee-item" style="background-image: url('https://images.unsplash.com/photo-1506157786151-b8491531f525?w=500')"></div>
            <div class="marquee-item" style="background-image: url('https://images.unsplash.com/photo-1508700115892-45ecd05ae2ad?w=500')"></div>
            <div class="marquee-item" style="background-image: url('https://images.unsplash.com/photo-1485579149621-3123dd979885?w=500')"></div>
        </div>
    </div>
""", unsafe_allow_html=True)

# --- 3. LOGIC ---

# 1. SETUP KEYS
api_key = st.secrets.get("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

# 2. ASYNC WRAPPER
def run_async(coroutine):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coroutine)

# 3. DEEZER FETCH
async def fetch_artist_image(artist):
    if not artist: return None
    clean_artist = artist.split(',')[0].split('&')[0].split('feat')[0].strip()
    try:
        url = f"https://api.deezer.com/search/artist?q={clean_artist}"
        r = requests.get(url, timeout=5)
        data = r.json()
        if 'data' in data and data['data']:
            return data['data'][0]['picture_xl']
    except:
        return None
    return None

# 4. SHAZAM IDENTIFY
async def identify_song(file_path):
    shazam = Shazam()
    try:
        out = await shazam.recognize(file_path)
        if 'track' in out:
            track = out['track']
            images = track.get('images', {})
            return {
                "found": True,
                "title": track.get('title'),
                "artist": track.get('subtitle'),
                "album_art": images.get('coverart'),
                "genre": track.get('genres', {}).get('primary')
            }
        return {"found": False}
    except Exception as e:
        return {"found": False, "error": str(e)}

# 5. GEMINI ANALYZE
def analyze_gemini(song_data):
    if not api_key: return "⚠️ Please add GEMINI_API_KEY to Streamlit Secrets."
    
    # --- UPDATED MODEL TO 2.5 ---
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
    except:
        # Fallback to older model only if 2.5 fails
        model = genai.GenerativeModel('gemini-2.0-flash-lite')
    
    prompt = f"""
    Act as a Music Producer. Analyze "{song_data['title']}" by "{song_data['artist']}".
    
    1. SONIC PROFILE: Sub-genre, Atmosphere, and Key Elements.
    2. SUNO AI PROMPT: Write a specific one-line prompt to generate this exact style.
       Format: [Genre], [Tempo], [Instruments], [Vocal Style]
    3. VOCAL ARCHITECTURE: Describe the vocal gender, processing (reverb/autotune), and delivery.
    """
    try:
        return model.generate_content(prompt).text
    except Exception as e:
        return f"AI Error: {e}"

# --- 4. MAIN UI FLOW ---
def main():
    if 'song_data' not in st.session_state:
        st.session_state.song_data = None

    # Start Glass Panel
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<h1>SUNOSONIC</h1>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">AI AUDIO INTELLIGENCE</div>', unsafe_allow_html=True)

    # STATE 1: UPLOAD
    if not st.session_state.song_data:
        uploaded_file = st.file_uploader("DROP AUDIO FILE HERE", type=['mp3', 'wav', 'ogg'])
        
        if uploaded_file:
            with st.spinner("🎧 ANALYZING AUDIO SPECTROGRAM..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                
                result = run_async(identify_song(tmp_path))
                os.remove(tmp_path)
                
                if result['found']:
                    img = run_async(fetch_artist_image(result['artist']))
                    result['artist_bg'] = img
                    st.session_state.song_data = result
                    st.rerun()
                else:
                    st.error("NO MATCH FOUND. TRY A CLEARER CLIP.")

    # STATE 2: RESULTS
    else:
        data = st.session_state.song_data
        
        bg_img = data.get('artist_bg') or data.get('album_art')
        if bg_img:
            st.image(bg_img, use_container_width=True)
            
        st.markdown(f"<h2 style='text-align:center; font-size: 2.5rem; margin-top:30px;'>{data['artist']}</h2>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align:center; color:#ccc; font-weight:lighter; margin-bottom:30px;'>{data['title']}</h3>", unsafe_allow_html=True)
        
        # AI Analysis
        with st.spinner("GENERATING PROMPT..."):
            analysis = analyze_gemini(data)
            st.info(analysis)

        if st.button("⬅ SCAN NEW TRACK"):
            st.session_state.song_data = None
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
