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

# -------------------------------------------------
# STREAMLIT CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Music Analyzer", layout="wide")

GLOBAL_CSS = """
<style>
/* Make the main container use the full width */
[data-testid="stAppViewContainer"] > .main {
    padding-left: 0;
    padding-right: 0;
}

[data-testid="stAppViewContainer"] .block-container {
    max-width: 1200px;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* Hero styles */
.hero-wrapper {
    width: 100%;
    margin: 1.5rem 0 1rem 0;
}

.hero-container {
    position: relative;
    width: 100%;
    height: 260px;
    overflow: hidden;
    border-radius: 16px;
}

.hero-bg {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;
    object-position: center;
    transform: scale(1.08);  /* slight zoom so it feels less boxed */
    filter: brightness(0.9);
}

.hero-gradient {
    position: absolute;
    inset: 0;
    background: linear-gradient(
        to right,
        rgba(0, 0, 0, 0.7),
        rgba(0, 0, 0, 0.35),
        rgba(0, 0, 0, 0.7)
    );
}

.hero-content {
    position: relative;
    z-index: 2;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: center;
    padding: 24px 32px;
    color: #ffffff;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
}

.hero-title {
    font-size: 28px;
    font-weight: 700;
    margin: 0 0 4px 0;
}

.hero-subtitle {
    font-size: 16px;
    opacity: 0.9;
    margin: 0 0 8px 0;
}

.hero-meta {
    font-size: 13px;
    opacity: 0.75;
}
</style>
"""
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def hero_banner(image_url: str, title: str = "", subtitle: str = "", meta: str = ""):
    st.markdown(
        f"""
        <div class="hero-wrapper">
          <div class="hero-container">
            <img src="{image_url}" class="hero-bg" />
            <div class="hero-gradient"></div>
            <div class="hero-content">
              <h1 class="hero-title">{title}</h1>
              <p class="hero-subtitle">{subtitle}</p>
              <p class="hero-meta">{meta}</p>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -------------------------------------------------
# SPOTIFY AUTH
# -------------------------------------------------
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


# -------------------------------------------------
# MAIN APP
# -------------------------------------------------
st.title("Music Analyzer")

uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"])

if uploaded_file:
    # Save audio temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    # Shazam detection
    async def detect_song(path):
        shazam = Shazam()
        out = await shazam.recognize_song(path)
        return out

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    shazam_result = loop.run_until_complete(detect_song(audio_path))

    track_title = ""
    artist_name = ""
    album_name = ""
    image_url = ""

    # Basic data from Shazam
    if "track" in shazam_result:
        track = shazam_result["track"]
        track_title = track.get("title", "")
        artist_name = track.get("subtitle", "")
        images = track.get("images", {})
        image_url = images.get("coverart", "")

    # Try Spotify lookup for better metadata
    token = get_spotify_token()
    if token and track_title and artist_name:
        query = f"{track_title} {artist_name}"
        url = f"https://api.spotify.com/v1/search?q={query}&type=track&limit=1"
        headers = {"Authorization": f"Bearer {token}"}
        r = requests.get(url, headers=headers)
        data = r.json()

        if data.get("tracks", {}).get("items"):
            item = data["tracks"]["items"][0]
            track_title = item["name"]
            artist_name = ", ".join(a["name"] for a in item["artists"])
            album_name = item["album"]["name"]
            album_images = item["album"]["images"]
            if album_images:
                image_url = album_images[0]["url"]

    # Hero banner if we have an image
    if image_url:
        meta_text = f"Album · {album_name}" if album_name else "Recognized track"
        hero_banner(
            image_url=image_url,
            title=track_title or "Unknown track",
            subtitle=artist_name or "",
            meta=meta_text,
        )
    else:
        st.subheader(track_title or "Unknown track")
        if artist_name:
            st.caption(artist_name)

    # Raw data in an expander so the UI does not feel broken
    with st.expander("Raw Shazam data"):
        st.json(shazam_result)
