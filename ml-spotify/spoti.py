import time
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from pprint import pprint

scope = "user-read-playback-state,user-modify-playback-state"

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope))

while True:
    current_track = sp.current_user_playing_track()
    is_playing = current_track['is_playing']
    progress_ms = current_track['progress_ms']
    tid = current_track['item']['id']
    name = current_track['item']['name']
    duration_ms = current_track['item']['duration_ms']

    if not is_playing:
        sp.start_playback()

    audio_analysis = sp.audio_features(tid)

    print(name, tid)
    pprint(audio_analysis)

    sleep_time = (duration_ms - progress_ms) / 1000

    print('sleeping for', sleep_time, 'seconds')
    time.sleep(sleep_time)
