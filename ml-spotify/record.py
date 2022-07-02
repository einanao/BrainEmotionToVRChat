import argparse
import time
import pickle
import os.path as osp
import numpy as np
import pprint

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels
from brainflow.data_filter import DataFilter, DetrendOperations

import spotipy
from spotipy.oauth2 import SpotifyOAuth


def tryFunc(func, val):
    try:
        return func(val)
    except:
        return None


def main():
    BoardShim.enable_board_logger()
    DataFilter.enable_data_logger()

    ### Uncomment this to see debug messages ###
    # BoardShim.set_log_level(LogLevels.LEVEL_DEBUG.value)

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
    parser.add_argument('--board-id', type=int, help='board id, check docs to get a list of supported boards',
                        required=True)
    parser.add_argument('--file', type=str, help='file',
                        required=False, default='')
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
    eeg_channels = tryFunc(BoardShim.get_eeg_channels, master_board_id)
    sampling_rate = tryFunc(BoardShim.get_sampling_rate, master_board_id)
    board.prepare_session()

    ### EEG Streaming Params ###
    eeg_window_size = 2
    update_speed = 0.001  # 4Hz update rate for VRChat OSC

    ### Spotify Setup ###
    scope = "user-read-playback-state,user-modify-playback-state"
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope))

    ### Pickle Setup ###
    SAVE_PATH = 'recording.pkl'
    save_dict = {}
    if osp.isfile(SAVE_PATH):
        with open(SAVE_PATH, 'rb') as f:
            save_dict = pickle.load(f)

    try:
        BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'Intializing')
        board.start_stream(450000, args.streamer_params)
        time.sleep(5)

        BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'Main Loop Started')
        while True:

            ### Displaying current data spread ###
            rows = [entry['audio_analysis'][0] for entry in save_dict.values()]
            audio_emotions = [(row['valence'], row['energy']) for row in rows]
            valences, energies = zip(*audio_emotions)
            avg_valence, std_valence = np.average(valences), np.std(valences)
            avg_energy, std_energy = np.average(energies), np.std(energies)
            spread_dict = {
                'avg_valence': avg_valence,
                'std_valence': std_valence,
                'avg_energy': avg_energy,
                'std_energy': std_energy,
                'data_length': len(audio_emotions)
            }
            data_spread_msg = '\nCurrent Data Spread:\n' + \
                pprint.pformat(spread_dict)
            BoardShim.log_message(LogLevels.LEVEL_INFO.value, data_spread_msg)

            BoardShim.log_message(LogLevels.LEVEL_INFO.value,
                                  'Waiting for Spotify to start')
            duration = float("inf")
            while True:
                current_track = sp.current_user_playing_track()
                if current_track:
                    progress_ms = current_track['progress_ms']
                    tid = current_track['item']['id']
                    name = current_track['item']['name']
                    duration_ms = current_track['item']['duration_ms']
                    duration = (duration_ms - progress_ms) / 1000
                    if current_track['is_playing'] and duration > 0:
                        break
                else:
                    time.sleep(1)

            log_msg = 'Getting audio features for song {}'.format(name)
            BoardShim.log_message(LogLevels.LEVEL_INFO.value, log_msg)
            audio_analysis = sp.audio_features(tid)

            start_time = time.time()
            end_time = start_time + duration

            log_msg = "Recording brainwaves for song {} with duration {:.3f}s".format(
                name, duration)
            BoardShim.log_message(LogLevels.LEVEL_INFO.value, log_msg)
            sample_avgs, sample_stds = [], []
            while time.time() < end_time:
                BoardShim.log_message(
                    LogLevels.LEVEL_DEBUG.value, "Getting Board Data")
                data = board.get_current_board_data(
                    eeg_window_size * sampling_rate)

                BoardShim.log_message(
                    LogLevels.LEVEL_DEBUG.value, "Calculating Power Bands")
                for eeg_channel in eeg_channels:
                    DataFilter.detrend(data[eeg_channel],
                                       DetrendOperations.LINEAR)
                feature_vector_avg, feature_vector_std = DataFilter.get_avg_band_powers(
                    data, eeg_channels, sampling_rate, True)

                BoardShim.log_message(
                    LogLevels.LEVEL_DEBUG.value, "Add to sample list")
                sample_avgs.append(feature_vector_avg)
                sample_stds.append(feature_vector_std)

                BoardShim.log_message(LogLevels.LEVEL_DEBUG.value, "Sleeping")
                time.sleep(update_speed)

            if len(sample_avgs) > 0:
                BoardShim.log_message(
                    LogLevels.LEVEL_INFO.value, 'Saving session')
                feature_avgs = np.array(sample_avgs)
                feature_avgs = np.array(sample_stds)
                row = {
                    'tid': tid,
                    'name': name,
                    'audio_analysis': audio_analysis,
                    'feature_avgs': feature_avgs,
                    'feature_stds': feature_avgs
                }
                save_dict[start_time] = row

    except KeyboardInterrupt:
        BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'Writing to disk')
        with open(SAVE_PATH, 'wb') as f:
            pickle.dump(save_dict, f)

        BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'Shutting down')
    finally:
        ### Cleanup ###
        board.stop_stream()
        board.release_session()


if __name__ == "__main__":
    main()
