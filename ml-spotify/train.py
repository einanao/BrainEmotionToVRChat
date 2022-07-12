import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import random

from pprint import pprint
from model import Model
import matplotlib.pyplot as plt

SAVE_PATH = 'recording.pkl'
SAMPLING_RATE = 250
NUM_LAYERS = 2

window = 2 * SAMPLING_RATE

save_dict = {}
with open(SAVE_PATH, 'rb') as f:
    save_dict = pickle.load(f)
entries = [entry for entry in save_dict.values() if len(
    entry['feature_avgs']) > window]
select_sample_count = 5000
print('select_sample_count', select_sample_count)

model = Model()
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_list = []

for i in range(select_sample_count):
    entry = random.choice(entries)

    # Getting target tensor
    audio_analysis = entry['audio_analysis'][0]
    target = torch.Tensor(
        [audio_analysis['valence'], audio_analysis['energy']])
    target = target.view(1, -1)

    # Getting input tensor
    input_tensor = random.choice(entry['feature_avgs'])
    input_tensor = torch.Tensor(input_tensor).view(1, -1)

    # training time
    model.zero_grad()
    output = model(input_tensor)

    loss = loss_function(output, target)
    loss.backward()
    optimizer.step()

    print(i, output, target, loss.item())
    loss_list.append(loss.item())
    if loss.item() < .005 and i > 500:
        break

plt.plot(loss_list)
plt.show()

torch.onnx.export(model, input_tensor, "spotify_emotion.onnx")

# for entry in entries:
#     print(len(entry['feature_avgs']))
# print(len(entries))
