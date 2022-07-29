import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import random
import numpy as np

from model import Model
import matplotlib.pyplot as plt

from brainflow.board_shim import BoardShim
from brainflow.data_filter import DataFilter, DetrendOperations

SAVE_PATH = 'recording.pkl'
BOARD_ID = 7
WINDOW_SECOND = 2
START_SEC = 0

# Values from range of frequencies found in get_avg_band_powers
START_FREQ = 2.0
END_FREQ = 45.0

with open(SAVE_PATH, 'rb') as f:
    record_data = pickle.load(f)
sampling_rate = record_data[0]['sampling_rate']
window = WINDOW_SECOND * sampling_rate

eeg_channels = BoardShim.get_eeg_channels(BOARD_ID)
sampling_rate = BoardShim.get_sampling_rate(BOARD_ID)


### Create Data ###

# Clean eeg signals
for entry in record_data:
    data = entry['board_data']
    for eeg_channel in eeg_channels:
        # DataFilter.perform_bandpass(data[eeg_channel], sampling_rate, START_FREQ, END_FREQ, 4,
        #                             FilterTypes.BUTTERWORTH.value, 0)
        DataFilter.detrend(data[eeg_channel],
                           DetrendOperations.LINEAR)

minimum_sample_size = min(
    [len(entry['board_data'][eeg_channels[0]]) for entry in record_data])


# Batch creation
start_idx = START_SEC * sampling_rate
end_idx = minimum_sample_size - window


def get_feature_vector(data):
    feature_vector, _ = DataFilter.get_avg_band_powers(
        data, eeg_channels, sampling_rate, True)
    return feature_vector


test_entries = []
for entry in record_data:
    data = entry['board_data']
    data_windows = [np.array([data_row[i: i + window] for data_row in data])
                    for i in range(start_idx, end_idx)]
    target_emotion = entry['emotion']
    target_emotions = [target_emotion] * len(data_windows)
    test_entries.extend(list(zip(target_emotions, data_windows)))

# test_entries = random.sample(test_entries, 10000)
random.shuffle(test_entries)


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


bsize = 100
test_batches = batch(test_entries, bsize)
batches_count = int(len(test_entries)/bsize)

# create model
model = Model()
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_list = []

test_batches = list(test_batches)
target_var = np.var([list(zip(*test_batch))[0][0] for test_batch in test_batches])

for i, test_batch in enumerate(test_batches):

    # get batch and format for pytorch
    target_emotions, data_windows = zip(*test_batch)
    feature_vectors = [get_feature_vector(
        data_window) for data_window in data_windows]

    targets = torch.Tensor(target_emotions).view(len(target_emotions), 1, 2)
    inputs = torch.Tensor(feature_vectors).view(len(feature_vectors), 1, 5)

    # Training
    model.zero_grad()
    outputs = model(inputs)
    loss = loss_function(outputs, targets)
    loss.backward()
    optimizer.step()

    print(i, batches_count, loss.item() / target_var, target_emotions[0], outputs[0])
    loss_list.append(loss.item())

    #if loss.item() < 0.05:
    #    break

# Save as ONNX model
dummy_input = inputs[0]
torch.onnx.export(model, dummy_input, "spotify_emotion.onnx")

plt.plot(loss_list)
plt.show()
