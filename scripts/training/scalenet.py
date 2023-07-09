#! /usr/bin/env python3

from jjf.models.scalegan import EntropyMatrixSampler, ScaleLSTM
from jjf.gradus.system.entropy import EntropyMatrix
from jjf.gradus.system.western import WesternSystem
from jjf.gradus.pitch import Pitch
import os
from keras.utils import to_categorical
import numpy as np
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

system = WesternSystem()
entropy = EntropyMatrix(system)
sampler = EntropyMatrixSampler(entropy, max_input_samples=10000, max_sample_sequence=2)
## nn = ScaleNet()
nn = ScaleLSTM()
print("Sampling...")
X, y = sampler.sample()
print(X.shape)
X = X.reshape((10000, 2, 1))
y = np.array([[x] for x in y])
# X = tf.expand_dims(X, axis=0)
# y = to_categorical(y, num_classes=128)
# y = tf.expand_dims(y, axis=0)
print("Training...")
nn.model.fit(X, y, epochs=10, batch_size=100)
yhat = nn.model.predict(X[:])
# for i in range(10):
#     print(
#         f"Incoming pitch set: {entropy.indices_to_pitches(X[:][i].reshape((2,)))}, Predicted pitch: {entropy.indices_to_pitches([int(yhat[i])])}"
#     )
for i in range(10):
    print(
        f"Incoming pitch set: {entropy.indices_to_pitches(X[:][i].reshape((2,)))}, Predicted pitch: {entropy.indices_to_pitches([int(round(yhat[i][0]))])}, Training pitch: {entropy.indices_to_pitches([int(y[i][0])])}"
    )

# print(
#    f"Incoming pitch set: {entropy.indices_to_pitches(X[:][0].reshape((2,)))}, Predicted pitch: {entropy.indices_to_pitches([int(yhat[0])])}, Training pitch: {entropy.indices_to_pitches([int(y[0][0])])}"
# )

print("trying C major")
yhat = nn.model.predict(
    [
        [
            entropy.pitches_to_indices([system.pitches["C4"]]),
            entropy.pitches_to_indices([system.pitches["G4"]]),
        ]
    ]
)
print("Predicted pitch: ", entropy.indices_to_pitches([int(yhat[0])]))
