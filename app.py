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

# --- 3. KNOWLEDGE BASE ---
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
    if not artist:
        return "https://images.unsplash.com/photo-1514525253440-b393452e8d26?w=1200"
    clean_artist = artist.split(',')[0].split('&')[0].split('feat')[0].strip()
    try:
        url = f"https://api.deezer.com/search/artist?q={clean_artist}"
        r = requests.get(url, timeout=5)
        data = r.json()
        if 'data' in data and data['data']:
            return data['data'][0]['picture_xl']
    except Exception:
        return "https://images.unsplash.com/photo-1514525253440-b393452e8d26?w=1200"
    return "https://images.unsplash.com/photo-1514525253440-b393452e8d26?w=1200"


# --- LIBROSA AUDIO ANALYSIS ENGINE ---
def extract_audio_features(file_path):
    """
    Examine the raw audio and extract:
    - Tempo (BPM)
    - Key
    - Timbre (brightness)
    - Energy
    - Simple vocal presence heuristic
    - Style hint based on harmonic/percussive balance
    """
    try:
        # Load audio (first 60 seconds for speed)
        y, sr = librosa.load(file_path, duration=60)

        # 1. TEMPO (BPM)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = round(float(tempo))

        # 2. KEY DETECTION (simple rough estimate)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_vals = np.sum(chroma, axis=1)
        pitches = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        key_idx = int(np.argmax(chroma_vals))
        key = pitches[key_idx]

        # 3. SPECTRAL CENTROID (Brightness/Timbre)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        avg_centroid = float(np.mean(spectral_centroids))
        if avg_centroid > 3000:
            brightness = "Bright / Aggressive"
        elif avg_centroid > 2000:
            brightness = "Balanced"
        else:
            brightness = "Dark / Mellow"

        # 4. RMS ENERGY (Loudness/Intensity)
        rms = librosa.feature.rms(y=y)[0]
        avg_energy = float(np.mean(rms))
        if avg_energy > 0.1:
            intensity = "High Energy"
        elif avg_energy > 0.05:
            intensity = "Moderate Energy"
        else:
            intensity = "Low / Chill"

        # 5. Harmonic / Percussive separation
        harmonic, percussive = librosa.effects.hpss(y)

        # 6. Simple vocal presence heuristic
        #    Look at harmonic content in the speech band (300–3400 Hz)
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

        # 7. Percussive vs harmonic energy hint
        perc_level = float(np.mean(np.abs(percussive)))
        harm_level = float(np.mean(np.abs(harmonic))) + 1e-9
        perc_ratio = perc_level / harm_level

        if perc_ratio > 1.2 and intensity != "Low / Chill":
            style_hint = "Band style with drums and rhythm section, possibly rock or pop with clear percussion."
        elif intensity == "Low / Chill" and brightness.startswith("Dark"):
            style_hint = "Lo-fi, ambient or chill ballad style, soft and relaxed."
        else:
            style_hint = "Modern production with a mix of electronic and acoustic elements."

        return {
            "success": True,
            "bpm": f"{bpm} BPM",
            "key": key,
            "timbre": brightness,
            "energy": intensity,
            "style_hint": style_hint,
            "vocals": vocals_flag,
            "voice_ratio": round(voice_ratio, 2)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def identify_song(file_path):
    shazam = Shazam()
    try:
        out = await shazam.recognize(file_path)
        if 'track' in out:
            track = out['track']
            return {
                "found": True,
                "source": "shazam",
                "title": track.get('title'),
                "artist": track.get('subtitle'),
                "album_art": track.get('images', {}).get('coverart'),
                "genre": track.get('genres', {}).get('primary', 'Electronic')
            }
        return {"found": False}
    except Exception as e:
        return {"found": False, "error": str(e)}


def analyze_gemini_json(song_data):
    """
    Use Gemini Flash 2.5 to convert either:
    - Librosa audio stats, or
    - Shazam metadata
    into structured JSON with mood, genre, instruments, vocal style, and a Suno-style prompt.
    """
    if not api_key:
        return None

    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
    except Exception:
        model = genai.GenerativeModel('gemini-1.5-flash')

    if song_data.get("source") == "librosa":
        # Unknown / custom track, rely on audio stats only
        prompt = f"""
You are an AI assistant specialized in music analysis and AI music prompting.

A user uploaded a completely unknown track, possibly created in Suno or another AI tool.
You only know these audio features:

- Tempo: {song_data.get('bpm')}
- Key: {song_data.get('key')}
- Timbre: {song_data.get('timbre')}
- Energy: {song_data.get('energy')}
- Style hint: {song_data.get('style_hint', '')}
- Vocal presence heuristic: {song_data.get('vocals', 'Unknown')} (voice band ratio {song_data.get('voice_ratio', 0)})

Use the vocal heuristic as a strong hint:
- If it says "Likely Vocals", assume there are clear vocals.
- If it says "Probably Instrumental / Background", assume the track can be treated as instrumental with no lead vocal.
- If "Unclear", you can choose either light vocals or instrumental.

Based on all this, infer:

1. A single-word primary mood (for example: "energetic", "melancholic", "dreamy", "cinematic").
2. A concise genre (for example: "indie rock", "acoustic ballad", "lofi hip hop", "EDM house", "rock with guitar and drums").
3. A short list of likely main instruments (for example: ["electric guitar", "drums", "bass", "synths"]).
4. Vocal type and style. If the heuristics suggest instrumental, set vocal_type to "instrumental" and vocal_style to "no lead vocal, instrumental track".
5. A compact Suno-style prompt (1–2 sentences) describing the style, tempo, key, mood, energy and instrumentation. This should be something the user can paste into Suno as a style reference.
6. 2–3 very short tips for recreating this vibe in AI music tools.

Return ONLY valid JSON in this exact structure:

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
        # Known or semi-known track via title/artist
        prompt = f"""
You are an AI assistant specialized in music analysis and AI music prompting.

Analyze the song "{song_data['title']}" by "{song_data['artist']}".
Based on typical information about this track (if known) or reasonable assumptions from the artist and title,
infer:

- A single-word mood.
- A concise genre.
- Tempo in BPM (string, for example "120 BPM", you can estimate if unknown).
- Key (you can estimate).
- Main instruments.
- Vocal type and vocal style.
- A short Suno-ready style prompt (1–2 sentences).
- 2–3 short tips for recreating the vibe with AI music tools.

Return ONLY valid JSON in this exact structure:

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
        # Fallback so the app still shows something useful even if JSON parsing fails
        return {
            "mood": "Unknown",
            "tempo": song_data.get("bpm", ""),
            "key": song_data.get("key", ""),
            "genre": song_data.get("genre", "Unknown"),
            "instruments": [],
            "vocal_type": "instrumental" if song_data.get("vocals", "").startswith("Probably Instrumental") else "Unknown",
            "vocal_style": "" if song_data.get("vocals", "").startswith("Probably Instrumental") else "",
            "suno_prompt": f"{song_data.get('genre', 'Unknown')} track at {song_data.get('bpm', '')} with {song_data.get('energy', '')} energy, suitable as a reference style.",
            "tips": []
        }


def format_lyrics_with_tags(raw_lyrics, song_analysis):
    if not api_key:
        return "Please set GEMINI_API_KEY in Streamlit secrets."
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
    except Exception:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
    prompt = f"""
Act as a Suno.ai meta-tagging expert.

CONTEXT:
- Genre: {song_analysis.get('genre', 'Pop')}
- Mood: {song_analysis.get('mood', 'General')}
- Official Tag Dictionary: {SUNO_TAGS}

TASK:
Insert structural tags like [Intro], [Verse], [Chorus], [Bridge], [Outro], etc, into the following lyrics
so they are ready for use in Suno or similar AI music generators.

Only return the tagged lyrics, no explanation.

LYRICS:
{raw_lyrics}
"""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {e}"


# --- 5. MAIN APPLICATION ---
def main():
    # Session state
    if 'song_data' not in st.session_state:
        st.session_state.song_data = None
        st.session_state.analysis = None
    if 'formatted_lyrics' not in st.session_state:
        st.session_state.formatted_lyrics = ""
    if 'uploaded_bytes' not in st.session_state:
        st.session_state.uploaded_bytes = None

    # API key notice
    if not api_key:
        st.warning("Gemini API key is not configured. Audio analysis will work, but AI prompts and lyric tagging will be limited.")

    # HEADER
    st.markdown("<h1 style='text-align:center; letter-spacing:-2px; margin-bottom:0;'>SUNOSONIC</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#555; font-size:0.8rem; letter-spacing:4px; margin-bottom:40px;'>AI AUDIO INTELLIGENCE</p>", unsafe_allow_html=True)

    # --- STATE 1: UPLOAD ---
    if not st.session_state.song_data:
        uploaded_file = st.file_uploader(" ", type=['mp3', 'wav', 'ogg'])

        if not uploaded_file:
            st.info("👆 Drop an audio file above to begin.")

        if uploaded_file:
            with st.spinner("🎧 Analyzing audio DNA..."):
                # Keep bytes for playback later
                st.session_state.uploaded_bytes = uploaded_file.getvalue()

                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                    tmp.write(st.session_state.uploaded_bytes)
                    tmp_path = tmp.name

                # 1. Librosa analysis (always, even if Shazam works)
                audio_stats = extract_audio_features(tmp_path)

                if not audio_stats.get("success"):
                    st.error("Audio file is corrupted or unreadable.")
                    os.remove(tmp_path)
                    return

                # 2. Try Shazam
                shazam_result = run_async(identify_song(tmp_path))

                if shazam_result.get("found"):
                    # Enrich Shazam data with Librosa stats
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
                    }
                else:
                    # Fallback to deep scan only (this is where Suno songs land)
                    st.toast("Metadata not found. Engaging deep audio scan...", icon="🧬")
                    result = {
                        "found": True,
                        "source": "librosa",
                        "title": "Unknown Track (Deep Scan)",
                        "artist": "Audio Fingerprint",
                        "album_art": "https://images.unsplash.com/photo-1511379938547-c1f69419868d?w=1200",
                        "genre": "Analyzing Signal...",
                        "bpm": audio_stats["bpm"],
                        "key": audio_stats["key"],
                        "timbre": audio_stats["timbre"],
                        "energy": audio_stats["energy"],
                        "style_hint": audio_stats["style_hint"],
                        "vocals": audio_stats["vocals"],
                        "voice_ratio": audio_stats["voice_ratio"],
                    }

                # 3. Fetch visuals
                if result["source"] == "shazam":
                    result["artist_bg"] = run_async(fetch_artist_image(result["artist"]))
                else:
                    result["artist_bg"] = "https://images.unsplash.com/photo-1478737270239-2f02b77ac6d5?w=1200"

                # 4. Run Gemini
                st.session_state.song_data = result
                st.session_state.analysis = analyze_gemini_json(result)
                os.remove(tmp_path)
                st.rerun()

    # --- STATE 2: DASHBOARD ---
    else:
        data = st.session_state.song_data
        ai = st.session_state.analysis or {}

        # HERO BANNER
        st.markdown(f"""
            <div class="hero-wrapper">
                <img src="{data['artist_bg']}" class="hero-bg">
                <div class="hero-overlay">
                    <div><span class="verified-badge">✓ {data['source'].upper()} ANALYSIS</span></div>
                    <div class="artist-title">{data['artist']}</div>
                    <div class="song-subtitle">{data['title']}</div>
                    <div class="meta-tags">
                        <span class="meta-pill">🎵 {ai.get('genre', data.get('genre', 'Unknown'))}</span>
                        <span class="meta-pill">⏱ {ai.get('tempo', data.get('bpm', '--'))}</span>
                        <span class="meta-pill">🎹 {ai.get('key', data.get('key', '--'))}</span>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Album art and audio
        if data.get('album_art') and 'http' in data['album_art']:
            st.image(data['album_art'], caption="Album art", use_container_width=True)

        if st.session_state.get("uploaded_bytes"):
            st.audio(st.session_state.uploaded_bytes, format="audio/mp3")

        st.markdown("<br>", unsafe_allow_html=True)

        # ANALYSIS GRID
        col1, col2 = st.columns(2)
        with col1:
            instruments_html = ''.join(
                [f'<span class="small-tag">{inst}</span>' for inst in ai.get('instruments', [])]
            )
            st.markdown(f"""
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
                    <div style="margin-top:10px; font-size:0.75rem; color:#94a3b8;">
                        BPM: {data.get('bpm', '--')} &nbsp;•&nbsp; Key: {data.get('key', '--')} &nbsp;•&nbsp; Timbre: {data.get('timbre', '--')}
                    </div>
                    <div style="margin-top:15px">{instruments_html}</div>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
                <div class="glass-panel glow-pink">
                    <div class="panel-header"><span style="color:#ec4899">🎙</span> VOCAL ARCHITECTURE</div>
                    <div class="stat-grid">
                        <div class="stat-box">
                            <div class="stat-label">TYPE</div>
                            <div class="stat-value">{ai.get('vocal_type', '-')}</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-label">STYLE</div>
                            <div class="stat-value">Modern</div>
                        </div>
                    </div>
                    <div style="margin-top:15px; font-size:0.9rem; color:#ccc; font-style:italic;">"{ai.get('vocal_style', '-')}"</div>
                </div>
            """, unsafe_allow_html=True)

        # VISUALIZER
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="glass-panel" style="border-top: 1px solid rgba(255,255,255,0.1);"><div class="panel-header">🎼 STRUCTURAL DYNAMICS & RMS (SIMULATED)</div></div>',
            unsafe_allow_html=True
        )
        spikiness = 2.0 if "High" in data.get('energy', '') else 0.5
        chart_data = pd.DataFrame(
            np.random.randn(80, 3) * spikiness,
            columns=['L', 'R', 'RMS']
        )
        st.area_chart(chart_data, height=120)

        st.markdown("<br>", unsafe_allow_html=True)

        # PROMPT & TIPS
        p_col, t_col = st.columns([1.5, 1])
        with p_col:
            st.markdown(
                f'<div class="glass-panel glow-purple"><div class="panel-header"><span style="color:#a855f7">🎹</span> SUNO AI STYLE PROMPT</div><div class="prompt-container">{ai.get("suno_prompt", "Prompt not available.")}</div></div>',
                unsafe_allow_html=True
            )

        with t_col:
            tips_list = ai.get('tips', [])
            tips_html = "".join(
                [f'<div class="tip-item"><div class="tip-num">{i+1}</div><div>{tip}</div></div>'
                 for i, tip in enumerate(tips_list)]
            )
            st.markdown(
                f'<div class="glass-panel"><div class="panel-header">💡 PRO TIPS</div>{tips_html or "No tips generated."}</div>',
                unsafe_allow_html=True
            )

        # LYRIC STUDIO
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="glass-panel glow-purple" style="border: 1px solid #4c1d95;">', unsafe_allow_html=True)
        st.markdown('<div class="panel-header">📝 LYRIC STUDIO <span class="small-tag" style="margin-left:10px; color:#a78bfa; border-color:#a78bfa;">AUTO TAGGER</span></div>', unsafe_allow_html=True)
        
        l_col1, l_col2 = st.columns(2)
        with l_col1:
            st.caption("Paste raw lyrics here")
            raw_input = st.text_area(
                "raw",
                height=300,
                placeholder="Type or paste lyrics...",
                label_visibility="collapsed",
                key="raw_lyrics_input"
            )
            
            with st.expander("📚 Suno meta tags reference"):
                for cat, tags in SUNO_TAGS.items():
                    st.markdown(f"**{cat}**")
                    st.markdown(" ".join([f"`{t}`" for t in tags]))

            if st.button("✨ Apply Suno meta tags", use_container_width=True):
                if raw_input:
                    with st.spinner("AI is structuring your lyrics based on the song style..."):
                        st.session_state.formatted_lyrics = format_lyrics_with_tags(raw_input, ai)
                else:
                    st.warning("Please paste some lyrics first.")
        
        with l_col2:
            st.caption("Formatted output")
            if st.session_state.formatted_lyrics:
                st.code(st.session_state.formatted_lyrics, language="markdown", line_numbers=False)
            else:
                st.markdown(
                    '<div class="lyric-output" style="color:#555; display:flex; align-items:center; justify-content:center;">Result will appear here...</div>',
                    unsafe_allow_html=True
                )
        st.markdown('</div>', unsafe_allow_html=True)

        # RESET
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("⬅ Analyze new track", use_container_width=True):
            st.session_state.song_data = None
            st.session_state.analysis = None
            st.session_state.formatted_lyrics = ""
            st.session_state.uploaded_bytes = None
            st.rerun()


if __name__ == "__main__":
    main()
