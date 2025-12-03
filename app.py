import streamlit as st
import asyncio
from shazamio import Shazam
import google.generativeai as genai
import tempfile
import os

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="SunoSonic",
    page_icon="🎹",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. NVGT KINETIC UI (CSS) ---
st.markdown("""
    <style>
    /* RESET & DARK THEME */
    .stApp {
        background-color: #000;
    }
    .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }
    header, footer { display: none !important; }

    /* KINETIC BACKGROUND */
    .kinetic-wrapper {
        position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
        z-index: 0; overflow: hidden;
        display: flex; flex-direction: column; justify-content: center; gap: 15px;
        opacity: 0.5; filter: grayscale(100%) contrast(1.2); pointer-events: none;
    }
    .marquee-row { display: flex; gap: 15px; width: 200vw; }
    .marquee-item {
        width: 250px; height: 150px; background-color: #111;
        border-radius: 8px; background-size: cover; background-position: center;
        flex-shrink: 0; border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* ANIMATIONS */
    @keyframes scrollLeft { from {transform: translateX(0);} to {transform: translateX(-50%);} }
    @keyframes scrollRight { from {transform: translateX(-50%);} to {transform: translateX(0);} }
    .scroll-left { animation: scrollLeft 60s linear infinite; }
    .scroll-right { animation: scrollRight 60s linear infinite; }

    /* GLASS FOREGROUND */
    .glass-panel {
        position: relative; z-index: 10;
        background: rgba(10, 10, 10, 0.75);
        backdrop-filter: blur(25px); -webkit-backdrop-filter: blur(25px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px; padding: 50px;
        max-width: 900px; margin: 80px auto;
        box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5);
    }

    /* TYPOGRAPHY */
    h1 {
        font-family: sans-serif; font-weight: 900; letter-spacing: -3px;
        font-size: 4rem !important; text-align: center; color: white; margin: 0;
    }
    .subtitle {
        text-align: center; color: #888; font-family: monospace; letter-spacing: 4px; margin-bottom: 40px;
    }
    
    /* CUSTOM STREAMLIT ELEMENTS */
    .stFileUploader { padding: 20px; border: 1px dashed rgba(255,255,255,0.2); border-radius: 12px; }
    .stButton button { width: 100%; border-radius: 8px; font-weight: bold; text-transform: uppercase; }
    </style>
    
    <div class="kinetic-wrapper">
        <div class="marquee-row scroll-left">
            <div class="marquee-item" style="background-image: url('https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?w=400')"></div>
            <div class="marquee-item" style="background-image: url('https://images.unsplash.com/photo-1493225255756-d9584f8606e9?w=400')"></div>
            <div class="marquee-item" style="background-image: url('https://images.unsplash.com/photo-1514525253440-b393452e8d26?w=400')"></div>
            <div class="marquee-item" style="background-image: url('https://images.unsplash.com/photo-1470225620780-dba8ba36b745?w=400')"></div>
            <div class="marquee-item" style="background-image: url('https://images.unsplash.com/photo-1511379938547-c1f69419868d?w=400')"></div>
            <div class="marquee-item" style="background-image: url('https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?w=400')"></div>
        </div>
        <div class="marquee-row scroll-right">
            <div class="marquee-item" style="background-image: url('https://images.unsplash.com/photo-1501612780327-45045538702b?w=400')"></div>
            <div class="marquee-item" style="background-image: url('https://images.unsplash.com/photo-1459749411177-0473ef71607b?w=400')"></div>
            <div class="marquee-item" style="background-image: url('https://images.unsplash.com/photo-1498038432885-c6f3f1b912ee?w=400')"></div>
            <div class="marquee-item" style="background-image: url('https://images.unsplash.com/photo-1520523839897-bd0b52f945a0?w=400')"></div>
            <div class="marquee-item" style="background-image: url('https://images.unsplash.com/photo-1501612780327-45045538702b?w=400')"></div>
        </div>
    </div>
""", unsafe_allow_html=True)

# --- 3. LOGIC (THE EAR & BRAIN) ---

# Get API Key from Secrets (Setup on Streamlit Cloud)
api_key = st.secrets.get("GEMINI_API_KEY") 
if api_key:
    genai.configure(api_key=api_key)

# Wrapper for Async functions
def run_async(coroutine):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coroutine)

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
                # ShazamIO gives us the High-Res Artist Image for free!
                "artist_bg": images.get('background'), 
                "album_art": images.get('coverart'),
                "genre": track.get('genres', {}).get('primary')
            }
        return {"found": False}
    except Exception as e:
        return {"found": False, "error": str(e)}

def analyze_gemini(song_data):
    if not api_key: return "⚠️ Please add GEMINI_API_KEY to Streamlit Secrets."
    model = genai.GenerativeModel('gemini-1.5-flash')
    
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
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<h1>SUNOSONIC</h1>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">AI AUDIO INTELLIGENCE</div>', unsafe_allow_html=True)

    if 'song_data' not in st.session_state:
        st.session_state.song_data = None

    # STATE 1: UPLOAD
    if not st.session_state.song_data:
        uploaded_file = st.file_uploader("DROP AUDIO FILE", type=['mp3', 'wav', 'ogg'])
        
        if uploaded_file:
            with st.spinner("🎧 LISTENING & DECODING..."):
                # Save temp file for Shazam
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                
                # Run the "Ear"
                result = run_async(identify_song(tmp_path))
                os.remove(tmp_path)
                
                if result['found']:
                    st.session_state.song_data = result
                    st.rerun()
                else:
                    st.error("NO MATCH FOUND. TRY A CLEARER CLIP.")

    # STATE 2: RESULTS
    else:
        data = st.session_state.song_data
        
        # Images
        bg_img = data.get('artist_bg') or data.get('album_art')
        cover_img = data.get('album_art') or bg_img
        
        # Hero Banner
        st.image(bg_img, use_container_width=True)
        st.markdown(f"<h2 style='text-align:center;'>{data['artist']}</h2>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align:center; color:#ccc;'>{data['title']}</h3>", unsafe_allow_html=True)
        
        # Reset Button
        if st.button("⬅ SCAN NEW TRACK"):
            st.session_state.song_data = None
            st.rerun()
            
        st.markdown("---")
        
        # AI Analysis
        with st.spinner("GENERATING PROMPT..."):
            analysis = analyze_gemini(data)
            st.info(analysis)

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
