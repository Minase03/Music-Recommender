import pandas as pd
import pickle

# Step 1: Sample music data
music_data = {
    'song': [
        'Shape of You',
        'Blinding Lights',
        'Dance Monkey',
        'Someone You Loved',
        'Rockstar'
    ],
    'artist': [
        'Ed Sheeran',
        'The Weeknd',
        'Tones and I',
        'Lewis Capaldi',
        'Post Malone'
    ]
}

# Step 2: Create DataFrame
music_df = pd.DataFrame(music_data)

# Step 3: Save to df.pkl
with open('df.pkl', 'wb') as file:
    pickle.dump(music_df, file)

print("✅ df.pkl has been created with sample music data.")

# Step 4: Load df.pkl and display contents
with open('df.pkl', 'rb') as file:
    loaded_music_df = pickle.load(file)

print("\n🎵 Loaded music data from df.pkl:")
print(loaded_music_df)
