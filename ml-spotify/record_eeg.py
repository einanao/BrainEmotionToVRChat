import argparse
import time
import pickle
import random

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels

import spotipy
from spotipy.oauth2 import SpotifyOAuth

META_PATH = 'metadata.pkl'
SAVE_PATH = 'recording.pkl'
COOLDOWN = 5


def tryFunc(func, val):
    try:
        return func(val)
    except:
        return None


def loudest_sections(sections, n):
    sections = sorted(sections, key=lambda section: -section['loudness'])[:n]
    return sections


def main():
    # find the minumum section size of a track for the eeg playlist
    # find the top n loudest sections per song based on the min section size
    with open(META_PATH, 'rb') as f:
        track_metadata = pickle.load(f)

    track_features = track_metadata['track_features']
    track_analyses = track_metadata['track_analyses']

    minumum_section_size = min([len(value['sections'])
                                for value in track_analyses.values()])
    section_idxs = list(range(minumum_section_size))

    track_emotions = {tid: (tfeat['valence'], tfeat['energy'])
                      for tid, tfeat in track_features.items()}
    track_sections = {tid: loudest_sections(value['sections'], minumum_section_size)
                      for tid, value in track_analyses.items()}
    track_ids = list(track_sections.keys())

    BoardShim.enable_board_logger()

    ### Paramater Setting ###
    parser = argparse.ArgumentParser()
    # use docs to check which parameters are required for specific board, e.g. for Cyton - set serial port
    parser.add_argument('--timeout', type=int, help='timeout for device discovery or connection', required=False,
                        default=0)
    parser.add_argument('--ip-port', type=int,
                        help='ip port', required=False, default=0)
    parser.add_argument('--ip-protocol', type=int, help='ip protocol, check IpProtocolType enum', required=False,
                        default=0)
    parser.add_argument('--ip-address', type=str,
                        help='ip address', required=False, default='')
    parser.add_argument('--serial-port', type=str,
                        help='serial port', required=False, default='')
    parser.add_argument('--mac-address', type=str,
                        help='mac address', required=False, default='')
    parser.add_argument('--other-info', type=str,
                        help='other info', required=False, default='')
    parser.add_argument('--streamer-params', type=str,
                        help='streamer params', required=False, default='')
    parser.add_argument('--serial-number', type=str,
                        help='serial number', required=False, default='')
    parser.add_argument('--file', type=str, help='file',
                        required=False, default='')

    parser.add_argument('--board-id', type=int, help='board id, check docs to get a list of supported boards',
                        required=True)
    parser.add_argument('--spotify-device-id', type=str, help='spotify device id, needed to remote control your spotify client',
                        required=True)

    args = parser.parse_args()

    params = BrainFlowInputParams()
    params.ip_port = args.ip_port
    params.serial_port = args.serial_port
    params.mac_address = args.mac_address
    params.other_info = args.other_info
    params.serial_number = args.serial_number
    params.ip_address = args.ip_address
    params.ip_protocol = args.ip_protocol
    params.timeout = args.timeout
    params.file = args.file

    ### EEG board setup ###
    board = BoardShim(args.board_id, params)
    master_board_id = board.get_board_id()
    sampling_rate = tryFunc(BoardShim.get_sampling_rate, master_board_id)
    board.prepare_session()

    ### Spotify Setup ###
    scope = "user-read-playback-state,user-modify-playback-state"
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope))

    def pause_playback():
        if sp.current_playback()['is_playing']:
            sp.pause_playback()

    ### Pickle Setup ###
    record_data = []

    try:
        BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'Intializing')
        board.start_stream(sampling_rate * 30 * 60, args.streamer_params)
        time.sleep(5)

        BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'Main Loop Started')
        for section_idx in section_idxs:
            random.shuffle(track_ids)
            for track_id in track_ids:
                track_emotion = track_emotions[track_id]
                track_section = track_sections[track_id][section_idx]

                start = track_section['start']
                duration = track_section['duration']

                log_msg = "Recording brainwaves for song {} section {} with duration {:.3f}s".format(
                    track_id, section_idx, duration)
                BoardShim.log_message(LogLevels.LEVEL_INFO.value, log_msg)

                sp.start_playback(uris=['spotify:track:'+track_id],
                                  position_ms=start*1000,
                                  device_id=args.spotify_device_id)

                time.sleep(duration)
                pause_playback()

                BoardShim.log_message(
                    LogLevels.LEVEL_INFO.value, "Playback stopped. Appending data to list")
                board_data = board.get_current_board_data(
                    int(sampling_rate * duration + 0.5))
                entry = {
                    'emotion': track_emotion,
                    'board_data': board_data,
                    'sampling_rate': sampling_rate
                }
                record_data.append(entry)

                BoardShim.log_message(LogLevels.LEVEL_INFO.value, "Cooldown")
                time.sleep(COOLDOWN)

        BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'Writing to disk')
        with open(SAVE_PATH, 'wb') as f:
            pickle.dump(record_data, f)
    except KeyboardInterrupt:
        None
    finally:
        BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'Shutting down')
        ### Cleanup ###
        pause_playback()
        board.stop_stream()
        board.release_session()


if __name__ == "__main__":
    main()
