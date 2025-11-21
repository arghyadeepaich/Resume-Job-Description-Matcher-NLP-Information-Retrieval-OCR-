import streamlit as st
import random

# --- 1. SETUP & DATA LOADING ---
try:
    with open('episode_links.txt', 'r') as f:
        # Read lines and remove empty ones
        episode_links = [link.strip() for link in f.readlines() if link.strip()]
except FileNotFoundError:
    st.error("Error: 'episode_links.txt' file not found. Please upload it.")
    episode_links = []

# --- 2. HELPER FUNCTIONS ---
def get_video_id(url):
    """Extracts video ID from both standard (v=) and short (youtu.be) URLs."""
    try:
        if "v=" in url:
            return url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in url:
            return url.split("youtu.be/")[1].split("?")[0]
    except IndexError:
        return None
    return None

def embed_random_episode():
    # Ensure we have an episode selected in session state
    if 'current_episode' not in st.session_state:
        if episode_links:
            st.session_state.current_episode = random.choice(episode_links)
        else:
            st.warning("No links found in episode_links.txt")
            return

    episode_url = st.session_state.current_episode
    video_id = get_video_id(episode_url)
    
    if not video_id:
        st.error(f"Could not play video. Invalid link format: {episode_url}")
        return
    
    embed_url = f"https://www.youtube.com/embed/{video_id}"
    
    # --- 3. DISPLAY THE VIDEO ---
    # We use HTML to make it responsive on mobile
    st.components.v1.html(f"""
        <style>
            .video-container {{
                position: relative;
                padding-bottom: 56.25%; /* 16:9 Aspect Ratio */
                height: 0;
                overflow: hidden;
            }}
            .video-container iframe {{
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                border: 0;
            }}
            .fallback-link {{
                display: block;
                margin-top: 10px;
                text-align: center;
                font-family: sans-serif;
                color: #d33;
                text-decoration: none;
                font-weight: bold;
            }}
        </style>
        
        <div class="video-container">
            <iframe src="{embed_url}" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
        </div>
        
        <a class="fallback-link" href="{episode_url}" target="_blank">
            Video not playing? Click here to watch on YouTube.
        </a>
    """, height=350)

# --- 4. APP LAYOUT ---
st.set_page_config(page_icon="👓", page_title="Retro TMKOC")
st.title("👓 Retro TMKOC Player")

# Button logic
if st.button("Play Random Episode"):
    if episode_links:
        # Pick a new random episode and save it to session state
        st.session_state.current_episode = random.choice(episode_links)
        # Rerun the app to update the view
        st.rerun()

# Always try to show the current episode if one is selected
if 'current_episode' in st.session_state:
    embed_random_episode()
