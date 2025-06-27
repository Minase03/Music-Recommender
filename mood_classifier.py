import pandas as pd
import random

# Load song dataset
def load_music_data(df_path="df.pkl"):
    with open(df_path, "rb") as f:
        return pd.read_pickle(f)

# Simple genre/mood mapping — you can expand this
mood_to_keywords = {
    "Happy": ["Dance", "Pop", "Party", "Upbeat"],
    "Sad": ["Ballad", "Acoustic", "Slow", "Love"],
    "Energetic": ["Rock", "Rap", "Electronic"],
    "Relaxed": ["Chill", "Lo-Fi", "Indie", "Ambient"]
}

# Function to filter songs based on mood
def get_songs_by_mood(mood, df):
    keywords = mood_to_keywords.get(mood, [])
    # You can match genre, song name, or artist (based on available columns)
    filtered = df[df['song'].str.contains('|'.join(keywords), case=False, na=False) |
                  df['artist'].str.contains('|'.join(keywords), case=False, na=False)]
    
    # If nothing found, just return some random songs as fallback
    if filtered.empty:
        return df.sample(5)
    return filtered.sample(min(5, len(filtered)))
