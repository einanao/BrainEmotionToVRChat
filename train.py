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

X = [] # inputs
Y = [] # targets
for entry in record_data:
    data = entry['board_data']
    data_windows = [np.array([data_row[i: i + window] for data_row in data])
                    for i in range(start_idx, end_idx)]
    target_emotion = entry['emotion']
    target_emotions = [target_emotion] * len(data_windows)
    feature_vectors = [get_feature_vector(
        data_window) for data_window in data_windows]

    X.extend(feature_vectors)
    Y.extend(target_emotions)
X = np.array(X)
Y = np.array(Y)
print('dataset shape: ', X.shape, Y.shape)

n = X.shape[0]
idxes = np.arange(0, n, 1)
np.random.shuffle(idxes)
X = X[idxes]
Y = Y[idxes]

with open('dataset.pkl', 'wb') as f:
  pickle.dump((X, Y), f, pickle.HIGHEST_PROTOCOL)

def converged(val_losses, ftol=1e-6, min_iters=2, eps=1e-9):
  return len(val_losses) >= max(2, min_iters) and (
      val_losses[-1] == np.nan or abs(val_losses[-1] - val_losses[-2]) /
      (eps + abs(val_losses[-2])) < ftol)

batch_size = 32 # training batch size
val_freq = 1000 # after every val_freq gradient steps, compute validation loss
train_frac = 0.9 # fraction of dataset to allocate to training set (rest is allocated to validation set)
n_epochs = 100 # number of passes through the training set

# create model
model = Model()
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4) # :P. see "adam is safe" https://karpathy.github.io/2019/04/25/recipe/

train_losses = []
val_losses = []
val_loss_steps = []
target_var = np.var(Y)
n_train_idxes = int(train_frac * n)
train_idxes = idxes[:n_train_idxes]
val_idxes = idxes[n_train_idxes:]
n_batches = int(np.ceil(n_train_idxes / batch_size))

def eval_loss(batch_idxes, train=True):
    # get batch and format for pytorch
    X_batch = X[batch_idxes]
    Y_batch = Y[batch_idxes]

    targets = torch.Tensor(Y_batch).view(len(Y_batch), 1, 2)
    inputs = torch.Tensor(X_batch).view(len(X_batch), 1, 5)

    # Training
    model.zero_grad()
    outputs = model(inputs)
    loss = loss_function(outputs, targets)
    loss.backward()
    if train:
      optimizer.step()

    return loss.item() / target_var

last_epoch = False
val_loss = None
for i in range(n_epochs):
    j = 0
    while j < n_train_idxes:
        train_batch_idxes = train_idxes[j:j+batch_size]
        train_loss = eval_loss(train_batch_idxes, train=True)
        train_losses.append(train_loss)
        if len(train_losses) % val_freq == 0:
            val_loss = eval_loss(val_idxes, train=False)
            val_losses.append(val_loss)
            val_loss_steps.append(len(train_losses))
        print(i, n_epochs, j // batch_size, n_batches, train_loss, val_loss)
        if converged(val_losses):
            last_epoch = True
            break
        j += batch_size
    if last_epoch:
        break

# Save as ONNX model
dummy_input = torch.Tensor(X[:1]).view(1, 1, 5)[0]
torch.onnx.export(model, dummy_input, "spotify_emotion.onnx")

plt.xlabel('gradient steps')
plt.ylabel('loss')
plt.plot(train_losses, label='training set')
plt.plot(val_loss_steps, val_losses, label='validation set')
plt.yscale('log')
plt.legend(loc='best')
plt.show()
