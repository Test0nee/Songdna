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
    url = "https://accounts.spotify.com/api/token" # Corrected URL
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {"grant_type": "client_credentials"}
    try:
        r = requests.post(url, headers=headers, data=data, auth=(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET))
        if r.status_code == 200:
            return r.json().get("access_token")
    except Exception:
        return None

# --- CONFIGURATION ---
st.set_page_config(
    page_title="SunoSonic Studio",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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
    .hero-container {
        position: relative; border-radius: 24px; overflow: hidden; margin-bottom: 30px;
        box-shadow: 0 25px 50px -12px rgba(0,0,0,0.7); border: 1px solid rgba(255,255,255,0.1);
        height: 380px; display: flex; align-items: center;
    }
    .hero-bg-blur {
        position: absolute; top: 0; left: 0; width: 100%; height: 100%;
        background-size: cover; background-position: center;
        filter: blur(40px) saturate(1.5) brightness(0.6); z-index: 0; transform: scale(1.1);
    }
    .hero-overlay-gradient {
        position: absolute; top: 0; left: 0; width: 100%; height: 100%;
        background: linear-gradient(90deg, rgba(0,0,0,0.8) 0%, rgba(0,0,0,0.4) 60%, rgba(0,0,0,0.1) 100%);
        z-index: 1;
    }
    .hero-content-flex {
        position: relative; z-index: 2; display: flex; align-items: center;
        padding: 40px; width: 100%; gap: 40px;
    }
    .album-cover-square {
        width: 280px; height: 280px; border-radius: 12px;
        box-shadow: 0 15px 40px rgba(0,0,0,0.6); object-fit: cover;
        border: 1px solid rgba(255,255,255,0.2); flex-shrink: 0;
    }
    .hero-text-col { display: flex; flex-direction: column; justify-content: center; }
    
    .verified-pill {
        display: inline-flex; align-items: center; gap: 6px;
        background: rgba(255,255,255,0.15); border: 1px solid rgba(255,255,255,0.2);
        padding: 6px 12px; border-radius: 50px; font-size: 0.75rem; font-weight: 700;
        text-transform: uppercase; margin-bottom: 16px; width: fit-content; backdrop-filter: blur(10px);
    }
    h1.hero-title {
        font-size: 4.5rem; font-weight: 900; margin: 0; line-height: 1; letter-spacing: -2px;
        text-shadow: 0 4px 30px rgba(0,0,0,0.5);
    }
    h2.hero-subtitle {
        font-size: 2rem; color: rgba(255,255,255,0.8); margin: 10px 0 25px 0; font-weight: 500; letter-spacing: -0.5px;
    }
    .meta-row { display: flex; gap: 10px; }
    .meta-tag {
        background: rgba(0,0,0,0.5); border: 1px solid rgba(255,255,255,0.1);
        padding: 8px 16px; border-radius: 8px; font-size: 0.85rem; color: #e2e8f0; font-weight: 600;
    }

    /* GLASS PANELS */
    .glass-panel {
        background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px; padding: 24px; backdrop-filter: blur(12px); margin-bottom: 20px;
    }
    .panel-title { 
        font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1.5px; 
        color: #94a3b8; font-weight: 700; margin-bottom: 15px; display: flex; align-items: center; gap: 8px;
    }

    /* VISUALIZER */
    .viz-container {
        background: #0b0f19; border: 1px solid #1e293b; border-radius: 16px;
        padding: 4px; box-shadow: inset 0 0 20px rgba(0,0,0,0.5);
    }

    /* STATS & LYRICS */
    .stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
    .stat-card { background: rgba(255,255,255,0.03); padding: 15px; border-radius: 12px; }
    .stat-label { font-size: 0.7rem; color: #64748b; margin-bottom: 5px; }
    .stat-val { font-size: 1.1rem; font-weight: 600; }
    .stTextArea textarea { background: #0b0f19 !important; border: 1px solid #1e293b !important; color: #cbd5e1 !important; }
    .code-block { background: #0b0f19; padding: 20px; border-radius: 12px; font-family: 'JetBrains Mono', monospace; color: #a5b4fc; border: 1px solid #1e293b; }
    
    /* BRANDING */
    .brand-wrap { text-align: center; margin-bottom: 24px; }
    .brand-title { font-size: 2.6rem; font-weight: 900; letter-spacing: 0.24em; text-transform: uppercase; margin: 0; text-shadow: 0 0 35px rgba(59,130,246,0.5); }
    .brand-subtitle { font-size: 0.78rem; letter-spacing: 0.32em; text-transform: uppercase; color: #6b7280; margin-top: 4px; }
    .top-action { text-align: center; margin: 10px 0 24px 0; }
    
    /* REMIX STATION */
    .remix-box {
        background: linear-gradient(90deg, rgba(124, 58, 237, 0.1), rgba(59, 130, 246, 0.1));
        border: 1px solid rgba(139, 92, 246, 0.3);
        border-radius: 16px; padding: 20px; margin-bottom: 20px;
    }
    
    /* GENRE DNA BARS */
    .dna-bar-row { margin-bottom: 12px; }
    .dna-labels { display: flex; justify-content: space-between; font-size: 0.75rem; color: #cbd5e1; margin-bottom: 4px; }
    .dna-track { width: 100%; height: 6px; background: rgba(255,255,255,0.1); border-radius: 10px; overflow: hidden; }
    .dna-fill { height: 100%; border-radius: 10px; }
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
    return "https://images.unsplash.com/photo-1470225620780-dba8ba36b745?w=1200"

def extract_dominant_color(image_url):
    try:
        if not image_url: return (30, 27, 75)
        response = requests.get(image_url, timeout=5)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        img = img.resize((1, 1))
        color = img.getpixel((0, 0))
        return color
    except:
        return (56, 189, 248)

# --- 5. AUDIO ENGINE ---
def safe_load_audio(file_path):
    errors = []
    # Try Librosa first
    try:
        y, sr = librosa.load(file_path, sr=None, duration=180)
        return y, sr, None
    except Exception as e:
        errors.append(f"Librosa: {str(e)}")
    
    # Try Soundfile as backup (Works for wav/flac even without ffmpeg)
    try:
        import soundfile as sf
        y, sr = sf.read(file_path)
        if len(y.shape) > 1: y = y.mean(axis=1)
        return y, sr, None
    except Exception as e:
        errors.append(f"Soundfile: {str(e)}")

    return None, None, " | ".join(errors)

def extract_audio_features(file_path):
    y, sr, error = safe_load_audio(file_path)
    if error: return {"success": False, "error": error}

    duration = librosa.get_duration(y=y, sr=sr)
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        if np.ndim(tempo) > 0: tempo = tempo[0]
        bpm = round(float(tempo)) if float(tempo) > 0 else 120
    except: bpm = 120 
    
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    key = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"][np.argmax(np.sum(chroma, axis=1))]
    
    rms = librosa.feature.rms(y=y)[0]
    energy_score = np.mean(rms)
    energy = "High" if energy_score > 0.1 else "Mid" if energy_score > 0.05 else "Low"
    
    hop = 512
    viz_rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    viz_rms = viz_rms / (np.max(viz_rms) + 1e-9) 
    
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        # Using named args to prevent TypeError in newer Librosa versions
        peaks = librosa.util.peak_pick(onset_env, pre_max=20, post_max=20, pre_avg=20, post_avg=20, delta=0.5, wait=100)
        section_times = librosa.frames_to_time(peaks, sr=sr)
    except: section_times = []
    
    filtered_sections = []
    last = 0
    for t in section_times:
        if t - last > 15:
            filtered_sections.append(t)
            last = t

    return {
        "success": True, "bpm": bpm, "key": key, "energy": energy,
        "duration": f"{int(duration//60)}:{int(duration%60):02d}",
        "waveform": viz_rms.tolist(), "sections": filtered_sections
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

# --- AI BRAIN ---
def analyze_gemini(data, intent="Original Style (Analysis)"):
    if not api_key: return None
    model = genai.GenerativeModel("gemini-2.5-flash")
    
    seg_str = ", ".join([f"Sec {i+1}: {s['end']-s['start']}s (Energy: {s['energy']})" for i,s in enumerate(data.get('segments',[]))])
    
    prompt = f"""
    Act as a Music Producer.
    
    SOURCE AUDIO DNA:
    - Song: {data.get('title','Unknown')} by {data.get('artist','Unknown')}
    - BPM: {data.get('bpm')} | Key: {data.get('key')} | Energy: {data.get('energy')}
    
    USER INTENT: "{intent}"
    
    TASK:
    1. Breakdown the Genre DNA (percentages).
    2. Map the song structure (Intro, Verse, etc).
    3. Create a Suno v3 Prompt.
    
    OUTPUT JSON:
    {{
        "mood": "Mood",
        "genre_breakdown": {{"Main Genre": 50, "Sub Genre": 30, "Influence": 20}},
        "structure_map": [
            {{"label": "Intro", "start": 0, "color": "#a855f7"}}, 
            {{"label": "Verse", "start": 15, "color": "#3b82f6"}}
        ],
        "vocal_type": "Vocal Style",
        "suno_prompt": "Prompt...",
        "tips": ["Tip 1", "Tip 2"]
    }}
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

    st.markdown("<h1 style='text-align:center; letter-spacing:-2px; margin-bottom:10px;'>SUNOSONIC</h1>", unsafe_allow_html=True)

    if not st.session_state.data:
        st.markdown('<div class="top-action"><span style="font-size:0.8rem; letter-spacing:0.18em; text-transform:uppercase; color:#9ca3af;">Upload a track to begin analysis</span></div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Drop audio file", type=['mp3','wav','ogg'])
        if uploaded:
            with st.spinner("🎧 Decoding DNA..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                    tmp.write(uploaded.getvalue())
                    tmp_path = tmp.name
                
                stats = extract_audio_features(tmp_path)
                if not stats['success']:
                    st.error(f"Analysis Failed: {stats['error']}")
                    st.info("Ensure ffmpeg is installed on the server.")
                    os.remove(tmp_path)
                    return

                meta = run_async(identify_song(tmp_path))
                full_data = {**stats, **(meta if meta['found'] else {"title":"Unknown","artist":"Deep Scan","img":None,"genre":"Unknown"})}
                full_data['artist_bg'] = run_async(fetch_artist_image(full_data['artist']))
                
                img_url = full_data.get('img') or full_data.get('artist_bg')
                full_data['color_rgb'] = extract_dominant_color(img_url)

                st.session_state.data = full_data
                st.session_state.ai = analyze_gemini(full_data) 
                os.remove(tmp_path)
                st.rerun()

    else:
        d = st.session_state.data
        ai = st.session_state.ai or {}
        
        # --- DYNAMIC BACKGROUND ---
        rgb = d.get('color_rgb', (30, 27, 75))
        st.markdown(f"""
            <style>
            .stApp {{
                background-image: radial-gradient(
                    circle at 50% 0%, 
                    rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.35) 0%, 
                    #050505 70%
                ) !important;
            }}
            </style>
        """, unsafe_allow_html=True)

        # --- HERO ---
        img_url = d.get('img') or d.get('artist_bg') or "https://images.unsplash.com/photo-1470225620780-dba8ba36b745"
        
        st.markdown(f"""
            <div class="hero-container">
                <div class="hero-bg-blur" style="background-image: url('{img_url}');"></div>
                <div class="hero-overlay-gradient"></div>
                <div class="hero-content-flex">
                    <img src="{img_url}" class="album-cover-square">
                    <div class="hero-text-col">
                        <div class="verified-pill">
                            <span style="color:#4ade80;">●</span> {d.get('source', 'AI Analysis').upper()}
                        </div>
                        <h1 class="hero-title">{d['artist']}</h1>
                        <h2 class="hero-subtitle">{d['title']}</h2>
                        <div class="meta-row">
                            <div class="meta-tag">🎵 {ai.get('genre', 'Unknown')}</div>
                            <div class="meta-tag">⏱ {d['bpm']} BPM</div>
                            <div class="meta-tag">🎹 {d['key']}</div>
                        </div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            # --- GENRE DNA VISUALIZATION ---
            breakdown_html = ""
            colors = ["#38bdf8", "#818cf8", "#c084fc", "#f472b6"]
            if 'genre_breakdown' in ai:
                for idx, (genre, pct) in enumerate(ai['genre_breakdown'].items()):
                    color = colors[idx % len(colors)]
                    breakdown_html += f"""
                    <div class="dna-bar-row">
                        <div class="dna-labels"><span>{genre}</span><span>{pct}%</span></div>
                        <div class="dna-track">
                            <div class="dna-fill" style="width:{pct}%; background:{color};"></div>
                        </div>
                    </div>
                    """
            
            st.markdown(f"""
                <div class="glass-panel">
                    <div class="panel-title">🧬 GENRE DNA</div>
                    <div style="margin-bottom:15px;">{breakdown_html}</div>
                    <div class="stat-grid">
                        <div class="stat-card"><div class="stat-label">MOOD</div><div class="stat-val">{ai.get('mood','-')}</div></div>
                        <div class="stat-card"><div class="stat-label">VOCAL TYPE</div><div class="stat-val">{ai.get('vocal_type','-')}</div></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
        with c2:
            # --- STRUCTURE MAP ---
            st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">📏 STRUCTURE MAP</div>', unsafe_allow_html=True)
            
            structure_map = ai.get('structure_map', [])
            total_dur = d.get('segments', [{}])[-1].get('end', 180)
            
            timeline_html = '<div style="display:flex; height:40px; width:100%; border-radius:8px; overflow:hidden; margin-top:20px;">'
            for i, sec in enumerate(structure_map):
                next_start = structure_map[i+1]['start'] if i+1 < len(structure_map) else total_dur
                duration = next_start - sec['start']
                width_pct = (duration / total_dur) * 100
                timeline_html += f'<div style="width:{width_pct}%; background:{sec.get("color","#555")}; display:flex; align-items:center; justify-content:center; color:white; font-size:10px; border-right:1px solid rgba(0,0,0,0.5);">{sec["label"]}</div>'
            timeline_html += '</div>'
            
            st.markdown(timeline_html, unsafe_allow_html=True)
            st.markdown("<div style='margin-top:15px; color:#94a3b8; font-size:0.8rem;'>AI estimated arrangement based on energy shifts.</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # --- VISUALIZER ---
        st.markdown('<div class="panel-title" style="margin-left:5px">📈 STRUCTURAL DYNAMICS</div>', unsafe_allow_html=True)
        y = np.array(d['waveform'])
        x = np.linspace(0, 100, len(y))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, fill='tozeroy', line=dict(color='#22d3ee', width=1), name='Energy'))
        fig.add_trace(go.Scatter(x=x, y=-y, fill='tozeroy', line=dict(color='#818cf8', width=1), name='Stereo'))
        
        for sec in structure_map:
            start_x = (sec['start'] / total_dur) * 100
            fig.add_vline(x=start_x, line_width=1, line_dash="solid", line_color="rgba(255,255,255,0.2)")
            fig.add_annotation(x=start_x + 1, y=0.8, text=sec['label'], showarrow=False, xanchor="left", font=dict(color=sec.get('color', 'white'), size=10))

        fig.update_layout(
            height=200, margin=dict(l=0,r=0,t=20,b=0),
            paper_bgcolor="#0b0f19", plot_bgcolor="#0b0f19",
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', showticklabels=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', showticklabels=False, range=[-1.1, 1.1]),
            showlegend=False, hovermode="x unified"
        )
        
        st.markdown('<div class="viz-container">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # --- STYLE REMIX ---
        st.markdown('<div class="panel-title">🎛️ STYLE REMIXER</div>', unsafe_allow_html=True)
        st.markdown('<div class="remix-box">', unsafe_allow_html=True)
        
        rc1, rc2 = st.columns([1, 2])
        with rc1:
            intent_preset = st.selectbox(
                "Choose Vibe Modifier", 
                ["Original Analysis", "More Energetic", "Darker / Moody", "Cinematic / Epic", "80s Retro", "Acoustic / Stripped", "Lofi / Chill", "Heavy / Aggressive"]
            )
        with rc2:
            custom_intent = st.text_input("Or type custom intent...", placeholder="e.g. 'Cyberpunk chase
