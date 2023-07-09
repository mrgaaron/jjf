import copy
from ..gradus.system.entropy import EntropyMatrix
import numpy as np
from scipy.stats import norm
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import SGD
import random


class EntropyMatrixSampler:
    def __init__(
        self,
        entropy_matrix: EntropyMatrix,
        max_input_samples=1_000_000,
        max_sample_sequence=3,
    ):
        self.entropy_matrix = entropy_matrix
        self.pitches = self.entropy_matrix.system.pitches
        self.max_sample_sequence = max_sample_sequence
        self.max_input_samples = max_input_samples

    def _sample_pitch_space(self):
        for _ in range(self.max_input_samples):
            pitches = []
            num_pitches = copy.copy(self.max_sample_sequence)
            while num_pitches > 0:
                if not pitches:
                    pitch = np.random.choice(list(self.entropy_matrix.system.pitches))
                    pitches.append(pitch)
                    num_pitches -= 1
                    continue

                last_idx = self.entropy_matrix.pitches_to_indices([pitches[-1]])[0]
                p = norm.pdf(
                    range(len(self.entropy_matrix.system.pitches)),
                    loc=last_idx,
                    scale=8,
                )
                p /= np.sum(p)

                pitch = np.random.choice(list(self.entropy_matrix.system.pitches), p=p)
                if pitch not in pitches:
                    pitches.append(pitch)
                    num_pitches -= 1
            yield pitches
        # for i, pitch in enumerate(self.entropy_matrix.system.pitches):
        #     for j, pitch2 in enumerate(self.entropy_matrix.system.pitches):
        #         if pitch.ratio(pitch2) in [5 / 4, 4 / 3, 3 / 2, 5 / 3, 2 / 1]:
        #             yield [pitch, pitch2]

    def sample(self):
        X = []
        y = []
        for inp in self._sample_pitch_space():
            x = np.zeros(self.max_sample_sequence)
            entropies = []
            for candidate in self.entropy_matrix.system.pitches:
                features = list(inp) + [candidate]
                entropies.append(self.entropy_matrix.total_pairwise_entropy(features))

            x[: len(inp)] = self.entropy_matrix.pitches_to_indices(inp)
            X.append(x)
            iter = 0
            while True:
                candidate = np.argsort(entropies)[iter]
                if candidate not in x and random.random() < 0.8:
                    y.append(candidate)
                    break
                iter += 1
        return np.array(X), np.array(y)


class ScaleNet:
    def __init__(self):
        self.model = self.define_nn()

    def define_nn(self):
        model = Sequential()
        # model.add(
        #    LSTM(8, input_shape=(3, 1), return_sequences=False, name="lstm_layer")
        # )
        model.add(
            Dense(
                8,
                activation="linear",
                name="dense_hidden_layer",
                kernel_initializer="normal",
            )
        )
        model.add(
            Dense(
                2,
                activation="linear",
                name="dense_hidden_layer2",
                kernel_initializer="normal",
            )
        )
        model.add(
            Dense(
                128,
                activation="softmax",
                name="output_layer",
            )
        )
        model.compile(
            loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"]
        )
        return model


class ScaleLSTM:
    def __init__(self):
        self.model = self.define_nn()

    def define_nn(self):
        model = Sequential()
        model.add(
            LSTM(16, input_shape=(2, 1), return_sequences=False, name="lstm_layer")
        )
        # model.add(
        #     LSTM(8, input_shape=(2, 1), return_sequences=False, name="lstm_layer2")
        # )
        model.add(
            Dense(
                8,
                activation="linear",
                name="dense_hidden_layer",
                kernel_initializer="normal",
            )
        )
        model.add(
            Dense(
                4,
                activation="linear",
                name="dense_hidden_layer2",
                kernel_initializer="normal",
            )
        )
        model.add(
            Dense(
                1,
                activation="linear",
                kernel_initializer="normal",
                name="output_layer",
            )
        )
        opt = SGD(learning_rate=0.01, momentum=0.9)
        model.compile(
            loss="mse",
            optimizer="adam",
            metrics=["accuracy", "mse"],
        )
        return model
