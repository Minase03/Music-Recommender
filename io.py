import pandas as pd
import pickle

# Sample data for music recommendations
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
    ],
    # Add more fields if needed, such as 'genre', 'year', etc.
}

# Create a DataFrame
music_df = pd.DataFrame(music_data)

# Save the DataFrame to a pickle file
with open('df.pkl', 'wb') as file:
    pickle.dump(music_df, file)

print("df.pkl has been created with sample music data.")
