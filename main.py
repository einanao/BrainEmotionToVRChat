import argparse
import time
import enum
import math
import numpy as np
import os

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, DetrendOperations, FilterTypes
from brainflow.ml_model import MLModel, BrainFlowMetrics, BrainFlowClassifiers, BrainFlowModelParams

from pythonosc.udp_client import SimpleUDPClient
from scipy.signal import find_peaks


START_FREQ = 2.0
END_FREQ = 45.0


class BAND_POWERS(enum.IntEnum):
    Gamma = 4
    Beta = 3
    Alpha = 2
    Theta = 1
    Delta = 0


class OSC_Path:
    Relax = '/avatar/parameters/osc_relax_avg'
    Focus = '/avatar/parameters/osc_focus_avg'
    Battery = '/avatar/parameters/osc_battery_lvl'
    ConnectionStatus = '/avatar/parameters/osc_is_connected'


def tanh_normalize(data, scale, offset):
    return np.tanh(scale * (data + offset))


def smooth(current_value, target_value, weight):
    current_value = (1.0 - weight) * current_value + weight * target_value
    return current_value


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

    ### OSC Setup ###
    ip = "127.0.0.1"
    send_port = 9000
    osc_client = SimpleUDPClient(ip, send_port)

    ### Biosensor board setup ###
    board = BoardShim(args.board_id, params)
    master_board_id = board.get_board_id()
    eeg_channels = tryFunc(BoardShim.get_eeg_channels, master_board_id)
    sampling_rate = tryFunc(BoardShim.get_sampling_rate, master_board_id)
    battery_channel = tryFunc(BoardShim.get_battery_channel, master_board_id)
    time_channel = tryFunc(BoardShim.get_timestamp_channel, master_board_id)
    board.prepare_session()

    ### EEG Band Calculation Params ###
    eeg_window_size = 2

    # normalize ratios between -1 and 1.
    # ONNX model range is [0,1] centered around 0.5. Tune scale to taste
    normalize_offset = -0.5
    normalize_scale = 1.3

    # Smoothing params
    detrend_eeg = True

    ### Streaming Params ###
    update_speed = 1 / 4  # 4Hz update rate for VRChat OSC
    ring_buffer_size = eeg_window_size * sampling_rate
    startup_time = 5
    board_timeout = 5

    ### ONNX Setup ###
    model_params = BrainFlowModelParams(BrainFlowMetrics.USER_DEFINED.value,
                                        BrainFlowClassifiers.ONNX_CLASSIFIER.value)
    model_params.file = os.path.join(os.getcwd(), 'spotify_emotion.onnx')
    model = MLModel(model_params)
    model.prepare()

    try:
        BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'Intializing')
        board.start_stream(ring_buffer_size, args.streamer_params)
        time.sleep(startup_time)

        BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'Main Loop Started')
        while True:
            BoardShim.log_message(
                LogLevels.LEVEL_DEBUG.value, "Getting Board Data")
            data = board.get_current_board_data(
                eeg_window_size * sampling_rate)

            BoardShim.log_message(LogLevels.LEVEL_DEBUG.value, "Timeout Check")
            time_data = data[time_channel]
            last_sample_time = time_data[-1]
            current_time = time.time()
            if current_time - last_sample_time > board_timeout:
                raise TimeoutError("Biosensor board timed out")

            battery_level = None if not battery_channel else data[battery_channel][-1]
            if battery_level:
                BoardShim.log_message(
                    LogLevels.LEVEL_DEBUG.value, "Battery: {}".format(battery_level))

            ### START ONNX SECTION ###
            BoardShim.log_message(
                LogLevels.LEVEL_DEBUG.value, "Calculating Power Bands")
            if detrend_eeg:
                for eeg_channel in eeg_channels:
                    DataFilter.detrend(data[eeg_channel],
                                       DetrendOperations.LINEAR)
            bands = DataFilter.get_avg_band_powers(
                data, eeg_channels, sampling_rate, True)
            feature_vector, _ = bands

            prediction = model.predict(feature_vector)
            prediction = tanh_normalize(
                prediction, 2 * normalize_scale, normalize_offset)
            current_focus = prediction[0]
            current_relax = prediction[1]
            BoardShim.log_message(LogLevels.LEVEL_DEBUG.value, "Arousal: {:.3f}\tValence: {:.3f}".format(
                current_focus, current_relax))
            ### END ONNX SECTION ###

            BoardShim.log_message(LogLevels.LEVEL_DEBUG.value, "Sending")
            osc_client.send_message(OSC_Path.Focus, current_focus)
            osc_client.send_message(OSC_Path.Relax, current_relax)
            osc_client.send_message(OSC_Path.ConnectionStatus, True)

            BoardShim.log_message(LogLevels.LEVEL_DEBUG.value, "Sleeping")
            time.sleep(update_speed)

    except KeyboardInterrupt:
        BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'Shutting down')
    except TimeoutError:
        BoardShim.log_message(LogLevels.LEVEL_INFO.value,
                              'Biosensor board timed out')
    finally:
        osc_client.send_message(OSC_Path.ConnectionStatus, False)
        ### Cleanup ###
        model.release()
        board.stop_stream()
        board.release_session()


if __name__ == "__main__":
    main()
