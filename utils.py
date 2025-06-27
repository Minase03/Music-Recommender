import pandas as pd
import pickle
import numpy as np # Import numpy for random values

# Step 1: Sample music data with added lyrics and placeholder features
music_data = {
    'song': [
        'Shape of You',
        'Blinding Lights',
        'Dance Monkey',
        'Someone You Loved',
        'Rockstar',
        'Levitating',
        'Bad Guy',
        'Sunflower',
        'Perfect',
        'Old Town Road'
    ],
    'artist': [
        'Ed Sheeran',
        'The Weeknd',
        'Tones and I',
        'Lewis Capaldi',
        'Post Malone',
        'Dua Lipa',
        'Billie Eilish',
        'Post Malone & Swae Lee',
        'Ed Sheeran',
        'Lil Nas X'
    ],
    'lyrics': [
        'The club isn’t the best place to find a lover so the bar is where I go',
        'I said, ooh, I’m blinded by the lights, no, I can’t sleep until I feel your touch',
        'They say, oh my God, I see the way you shine, take your hands, my dear, and place them both in mine',
        'I’m going under and this time I fear there’s no one to save me, this all or nothing really got a way of driving me crazy',
        'I’ve been fuckin’ hoes and poppin’ pillies, man, I feel just like a rockstar, ah-ah, ah-ah',
        'You want me, I want you, baby, my sugarboo, I’m levitating',
        'White shirt now red, my bloody nose, sleepin’, you’re on your tippy toes',
        'Needless to say, I keep her in check, she was a bad-bad, nevertheless',
        'Baby, I’m dancing in the dark with you between my arms, barefoot on the grass',
        'I got the horses in the back, horse tack is attached, hat is matte black'
    ],
    # Placeholder for sentiment scores (-1.0 to 1.0)
    'sentiment_score': [
        0.7,   # Shape of You - Positive
        0.1,   # Blinding Lights - Neutral/Slightly Positive
        0.8,   # Dance Monkey - Positive
        -0.6,  # Someone You Loved - Negative
        -0.2,  # Rockstar - Slightly Negative/Neutral
        0.9,   # Levitating - Very Positive
        -0.4,  # Bad Guy - Negative/Neutral
        0.5,   # Sunflower - Positive
        0.95,  # Perfect - Very Positive
        0.2    # Old Town Road - Neutral/Slightly Positive
    ],
    # Placeholders for Spotify Audio Features (ranges 0.0-1.0, tempo in BPM)
    'tempo': [100.0, 171.0, 98.0, 96.0, 160.0, 103.0, 135.0, 90.0, 95.0, 136.0],
    'energy': [0.6, 0.8, 0.7, 0.4, 0.7, 0.9, 0.5, 0.6, 0.3, 0.7],
    'danceability': [0.8, 0.5, 0.8, 0.6, 0.6, 0.8, 0.7, 0.7, 0.5, 0.9],
    'valence': [0.9, 0.4, 0.7, 0.3, 0.4, 0.9, 0.3, 0.7, 0.8, 0.8]
}

# Step 2: Create DataFrame
music_df = pd.DataFrame(music_data)

# Step 3: Save to df.pkl
with open('df.pkl', 'wb') as file:
    pickle.dump(music_df, file)

print("✅ df.pkl has been created with enriched sample music data (including placeholder lyrics, sentiment, and audio features).")

# Step 4: Load df.pkl and display contents
with open('df.pkl', 'rb') as file:
    loaded_music_df = pickle.load(file)

print("\n🎵 Loaded music data from df.pkl:")
print(loaded_music_df.head()) # Display head to show new columns
print(f"\nDataFrame shape: {loaded_music_df.shape}")
print(f"Columns: {loaded_music_df.columns.tolist()}")