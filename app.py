import pickle
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import json # For saving/loading simple user data
import pandas as pd # Ensure pandas is imported for DataFrame operations
import numpy as np # Ensure numpy is imported for numerical operations

# Assuming mood_classifier.py is in the same directory and updated for sentiment
from mood_classifier import get_songs_by_vibe, load_music_data

# --- Spotify API Credentials ---
CLIENT_ID = "70a9fb89662f4dac8d07321b259eaad7"
CLIENT_SECRET = "4d6710460d764fbbb8d8753dc094d131"

# Initialize the Spotify client
client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# --- Helper function to get album cover URL ---
def get_song_album_cover_url(song_name, artist_name):
    search_query = f"track:{song_name} artist:{artist_name}"
    results = sp.search(q=search_query, type="track")

    if results and results["tracks"]["items"]:
        track = results["tracks"]["items"][0]
        # Check if album images exist before accessing
        if track["album"]["images"]:
            album_cover_url = track["album"]["images"][0]["url"]
            return album_cover_url
    return "https://i.postimg.cc/0QNxYz4V/social.png" # Fallback image

# --- CORE RECOMMENDATION LOGIC (with Explainable Recommendations) ---
def recommend(song, music_df, similarity_matrix):
    try:
        index = music_df[music_df['song'] == song].index[0]
    except IndexError:
        st.error(f"Song '{song}' not found in our database. Please select another.")
        return [], [], [] # Return empty lists if song not found

    distances = sorted(list(enumerate(similarity_matrix[index])), reverse=True, key=lambda x: x[1])
    
    recommended_music_names = []
    recommended_music_posters = []
    recommended_similarities = [] # New list for similarity scores

    # Get top 5 recommendations (excluding the input song itself)
    for i in distances[1:6]:
        song_data = music_df.iloc[i[0]]
        song_name = song_data['song']
        artist = song_data['artist']
        similarity_score = round(i[1] * 100, 2) # Convert similarity to percentage

        recommended_music_posters.append(get_song_album_cover_url(song_name, artist))
        recommended_music_names.append(song_name)
        recommended_similarities.append(similarity_score)
    
    return recommended_music_names, recommended_music_posters, recommended_similarities

# --- User Data Management (for Twin Listener) ---
USER_DATA_FILE = "user_likes.json"

def load_user_likes():
    try:
        with open(USER_DATA_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {} # Returns an empty dict if file doesn't exist or is invalid

def save_user_likes(user_likes_data):
    with open(USER_DATA_FILE, 'w') as f:
        json.dump(user_likes_data, f, indent=4)

def toggle_like_song(username, song_name, like=True):
    if username not in st.session_state.user_likes:
        st.session_state.user_likes[username] = []
    
    if like and song_name not in st.session_state.user_likes[username]:
        st.session_state.user_likes[username].append(song_name)
        st.session_state.user_likes[username] = list(set(st.session_state.user_likes[username])) # Ensure unique
    elif not like and song_name in st.session_state.user_likes[username]:
        st.session_state.user_likes[username].remove(song_name)
    save_user_likes(st.session_state.user_likes)

def find_music_twins(current_user, user_likes_data, min_overlap_percent=30): # Reduced for more matches
    if current_user not in user_likes_data or not user_likes_data[current_user]:
        return "You need to like some songs first to find twins!"

    current_user_liked = set(user_likes_data[current_user])
    if not current_user_liked:
        return "You need to like some songs first to find twins!"
        
    twins = {}

    for other_user, other_likes in user_likes_data.items():
        if other_user == current_user or not other_likes:
            continue
        
        other_user_liked = set(other_likes)
        
        common_songs = len(current_user_liked.intersection(other_user_liked))
        
        # Calculate overlap based on the smaller set of liked songs to be fair
        min_liked = min(len(current_user_liked), len(other_user_liked))
        
        if min_liked > 0: # Avoid division by zero
            overlap_percentage = (common_songs / min_liked) * 100
            if overlap_percentage >= min_overlap_percent:
                twins[other_user] = round(overlap_percentage, 2)
    
    # Sort twins by overlap percentage
    sorted_twins = sorted(twins.items(), key=lambda item: item[1], reverse=True)
    return sorted_twins

# --- Streamlit App Layout ---
st.header('Music Recommender System')

# --- Load Data ---
# music = pickle.load(open('df.pkl','rb')) # Use the function from mood_classifier.py
# If df.pkl is expected to have sentiment_score, use load_music_data
music = load_music_data()
similarity = pickle.load(open('similarity.pkl','rb'))

# --- User Profile (Simulated) ---
if 'user_likes' not in st.session_state:
    st.session_state.user_likes = load_user_likes()
if 'current_user' not in st.session_state:
    st.session_state.current_user = None

st.sidebar.subheader("👤 User Profile (Simulated)")
user_options = list(st.session_state.user_likes.keys())
selected_user_input = st.sidebar.selectbox("Select User:", user_options + ["New User"])

if selected_user_input == "New User":
    new_username = st.sidebar.text_input("Enter new username:")
    if st.sidebar.button("Create User") and new_username:
        if new_username not in st.session_state.user_likes:
            st.session_state.user_likes[new_username] = []
            save_user_likes(st.session_state.user_likes)
            st.session_state.current_user = new_username
            st.sidebar.success(f"User '{new_username}' created!")
            st.rerun()
 # Rerun to update selectbox
        else:
            st.sidebar.error("Username already exists!")
else:
    st.session_state.current_user = selected_user_input

if st.session_state.current_user:
    st.sidebar.write(f"Logged in as: **{st.session_state.current_user}**")
    st.sidebar.write("Your liked songs:")
    for liked_song in st.session_state.user_likes.get(st.session_state.current_user, []):
        st.sidebar.write(f"- {liked_song}")
else:
    st.sidebar.info("Please select or create a user to enable personalized features.")


# --- Original Recommendation Feature ---
st.subheader("🎵 Song Recommendations")
music_list = music['song'].values
selected_song = st.selectbox(
    "Type or select a song from the dropdown",
    music_list
)

if st.button('Show Recommendation'):
    recommended_music_names, recommended_music_posters, recommended_similarities = recommend(selected_song, music, similarity)
    
    if recommended_music_names: # Only display if recommendations were found
        cols = st.columns(5)
        for idx, (name, poster, score) in enumerate(zip(recommended_music_names, recommended_music_posters, recommended_similarities)):
            with cols[idx]:
                st.text(name)
                st.image(poster)
                st.markdown(f"**Similarity: {score}%**")
                if st.session_state.current_user:
                    is_liked = name in st.session_state.user_likes.get(st.session_state.current_user, [])
                    if st.button(f"💖 {'Liked!' if is_liked else 'Like'}", key=f"like_rec_{idx}_{name}", disabled=is_liked):
                        toggle_like_song(st.session_state.current_user, name, like=True)
                        st.experimental_rerun() # Rerun to update button state
                else:
                    st.caption("Log in to like songs.")

st.markdown("---") # Separator

# --- New: Vibe Match Recommender ---
st.subheader("🎵 Vibe Match Recommender (Sentiment-Based)")
vibe_mood_options = ["Happy", "Chill", "Hype", "Sad", "Romantic"]
selected_vibe = st.selectbox("Pick your current vibe:", vibe_mood_options)

if st.button("Get Vibe Match Recommendations"):
    # Ensure 'sentiment_score' column exists in your df.pkl for this to work effectively
    if 'sentiment_score' not in music.columns:
        st.warning("Sentiment analysis not yet performed on lyrics. Recommendations will be random. Please update 'Model Training.ipynb' to include sentiment analysis and save to 'df.pkl'.")
        vibe_recs_df = music.sample(min(5, len(music)))[['song', 'artist', 'album_cover_url']] # Fallback
    else:
        vibe_recs_df = get_songs_by_vibe(selected_vibe, music, num_songs=5)
    
    if not vibe_recs_df.empty:
        st.write(f"Songs matching your '{selected_vibe}' vibe:")
        cols = st.columns(5)
        for i, row in vibe_recs_df.iterrows():
            with cols[i % 5]: # Use modulo to cycle through columns
                st.text(row['song'])
                st.caption(row['artist'])
                if 'album_cover_url' in row and row['album_cover_url']:
                    st.image(row['album_cover_url'])
                else:
                    st.image("https://i.postimg.cc/0QNxYz4V/social.png")
                
                if st.session_state.current_user:
                    is_liked = row['song'] in st.session_state.user_likes.get(st.session_state.current_user, [])
                    if st.button(f"💖 {'Liked!' if is_liked else 'Like'}", key=f"like_vibe_{i}_{row['song']}", disabled=is_liked):
                        toggle_like_song(st.session_state.current_user, row['song'], like=True)
                        st.experimental_rerun()
                else:
                    st.caption("Log in to like songs.")
    else:
        st.info(f"No songs found matching the '{selected_vibe}' vibe based on current data.")


st.markdown("---")
# --- Find Music Twins ---
st.subheader("👯‍♂️ Find Your Music Twins!")
if st.session_state.current_user:
    if st.button("Find My Twins"):
        twins_results = find_music_twins(st.session_state.current_user, st.session_state.user_likes)
        if isinstance(twins_results, str):
            st.info(twins_results)
        elif twins_results:
            st.write("Here are your music twins:")
            for twin_user, percentage in twins_results:
                st.write(f"- **{twin_user}** ({percentage}% overlap in liked songs)")
                st.write(f"Common songs: {list(set(st.session_state.user_likes.get(st.session_state.current_user, [])).intersection(set(st.session_state.user_likes.get(twin_user, []))))}")
        else:
            st.info("No music twins found yet based on your liked songs. Try liking more songs!")
else:
    st.info("Please log in or create a user to find your music twins.")

st.markdown("---")
# --- Audio Feature Matching UI (Conceptual) ---
st.subheader("🎶 Audio Vibe Filter (Conceptual)")
st.info("This feature requires 'tempo', 'energy', 'danceability', 'valence' columns in your `df.pkl`. You'll need to extract these using Spotify API (or `librosa` for local files) and update `df.pkl` in your 'Model Training.ipynb' or a separate script.")

if all(col in music.columns for col in ['tempo', 'energy', 'danceability', 'valence']):
    desired_tempo = st.slider("Desired Tempo (BPM):", min_value=60.0, max_value=200.0, value=120.0, step=5.0)
    desired_energy = st.slider("Desired Energy:", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
    desired_danceability = st.slider("Desired Danceability:", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    desired_valence = st.slider("Desired Positivity (Valence):", min_value=0.0, max_value=1.0, value=0.6, step=0.05)

    if st.button("Find Songs by Audio Vibe"):
        # Simple filtering based on proximity
        music['temp_dist'] = abs(music['tempo'] - desired_tempo)
        music['energy_dist'] = abs(music['energy'] - desired_energy)
        music['dance_dist'] = abs(music['danceability'] - desired_danceability)
        music['valence_dist'] = abs(music['valence'] - desired_valence)
        
        # Sum of normalized distances (simple example, you might use a weighted sum)
        music['total_audio_dist'] = (music['temp_dist'] / 140) + music['energy_dist'] + music['dance_dist'] + music['valence_dist']
        
        audio_vibe_recs = music.sort_values(by='total_audio_dist').head(5)
        
        if not audio_vibe_recs.empty:
            st.write("Songs matching your audio vibe:")
            cols = st.columns(5)
            for i, row in audio_vibe_recs.iterrows():
                with cols[i % 5]:
                    st.text(row['song'])
                    st.caption(row['artist'])
                    st.write(f"BPM: {round(row['tempo'])}")
                    st.write(f"Energy: {round(row['energy'], 2)}")
                    if 'album_cover_url' in row and row['album_cover_url']:
                        st.image(row['album_cover_url'])
                    else:
                        st.image("https://i.postimg.cc/0QNxYz4V/social.png")
        else:
            st.info("No songs found matching this audio vibe.")
else:
    st.info("To enable 'Audio Vibe Filter', please integrate Spotify Audio Features (BPM, Energy, Danceability, Valence) into your `df.pkl`.")