import pickle

# Load the df.pkl file
with open('df.pkl', 'rb') as file:
    music = pickle.load(file)

# Display the content to verify
print(music)
