import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Extended music data with images (album cover URLs)
music_data = {
    'song': [
        'Shape of You', 
        'Blinding Lights', 
        'Dance Monkey', 
        'Someone You Loved', 
        'Rockstar',
        'Levitating',  # New song added
        'Bad Guy',     # New song added
        'Sunflower'    # New song added
    ],
    'artist': [
        'Ed Sheeran', 
        'The Weeknd', 
        'Tones and I', 
        'Lewis Capaldi', 
        'Post Malone',
        'Dua Lipa',    # New artist added
        'Billie Eilish', # New artist added
        'Post Malone & Swae Lee' # New artist added
    ],
    'lyrics': [
        'The club isn’t the best place to find a lover', 
        'I said, ooh, I’m blinded by the lights', 
        'They say, oh my God, I see the way you shine', 
        'I’m going under and this time I fear there’s no one to save me', 
        'I’ve been fuckin’ hoes and poppin’ pillies, man, I feel just like a rockstar',
        'You want me, I want you, baby', # New lyrics added
        'White shirt now red, my bloody nose', # New lyrics added
        'Then I fell in love, she’s so sweet, oh' # New lyrics added
    ],
    'image_url': [
        "https://i.scdn.co/image/ab67616d0000b273ba5db46f4b838ef6027e6f96",  # Ed Sheeran - Shape of You
        "https://i.scdn.co/image/ab67616d0000b2738863bc11d2aa12b54f5aeb36",  # The Weeknd - Blinding Lights
        "https://i.scdn.co/image/ab67616d0000b27373c672ea1f2a26b96e8b0596",  # Tones and I - Dance Monkey
        "https://i.scdn.co/image/ab67616d0000b273fc2101e6889d6ce9025f85f2",  # Lewis Capaldi - Someone You Loved
        "https://i.scdn.co/image/ab67616d0000b273b1c4b76e23414c9f20242268",  # Post Malone - Rockstar
        "https://i.scdn.co/image/ab67616d0000b2737a8362ab7b3a30db0b8c2e59",  # Dua Lipa - Levitating
        "https://i.scdn.co/image/ab67616d0000b27332b41e96714c12a91a6d5282",  # Billie Eilish - Bad Guy
        "https://i.scdn.co/image/ab67616d0000b27362f5739e3eb2e385a396cfd3"   # Post Malone & Swae Lee - Sunflower
    ]
}

# Create a DataFrame
music_df = pd.DataFrame(music_data)

# Initialize TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')

# Fit and transform the lyrics into a matrix
tfidf_matrix = tfidf.fit_transform(music_df['lyrics'])

# Compute cosine similarity matrix
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Save the similarity matrix to a pickle file
with open('similarity.pkl', 'wb') as file:
    pickle.dump(similarity_matrix, file)

# Save the music DataFrame to a pickle file (df.pkl)
with open('df.pkl', 'wb') as file:
    pickle.dump(music_df, file)

print("similarity.pkl and df.pkl have been successfully created.")
