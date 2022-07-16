import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pickle

EEG_PLAYLIST_ID = '563Ye1ZDYdECqG5TLlEWSZ'
META_PATH = 'metadata.pkl'

scope = "user-read-playback-state,user-modify-playback-state"
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope))

# Get EEG Playlist details
res = sp.playlist(EEG_PLAYLIST_ID)
track_items = res['tracks']['items']
track_ids = [item['track']['id'] for item in track_items]

# Get Track audio features
track_analyses = {track_id: sp.audio_analysis(
    track_id) for track_id in track_ids}

# Get Track Analysis Features (useful for sections of a song)
track_features = sp.audio_features(track_ids)
track_features = {
    track_feature['id']: track_feature for track_feature in track_features}

# Create metadata dictionaries
track_metadata = {
    'track_analyses': track_analyses,
    'track_features': track_features
}

# Save metadata
with open(META_PATH, 'wb') as f:
    pickle.dump(track_metadata, f)
print("Metadata saved")
