from ..gradus.system.entropy import EntropyMatrix
import numpy as np
from numpy.random import randn
from numpy.random import randint
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten, BatchNormalization
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from matplotlib import pyplot


class EntropyMatrixSampler:
    def __init__(
        self,
        entropy_matrix: EntropyMatrix,
        max_input_samples=1_000_000,
        max_sample_sequence=5,
    ):
        self.entropy_matrix = entropy_matrix.matrix
        self.pitches = self.entropy_matrix.system.pitches
        self.max_sample_sequence = max_sample_sequence
        self.max_input_samples = max_input_samples

    def _sample_pitch_space(self):
        for _ in range(self.max_input_samples):
            base_arr = np.zeros(self.max_sample_sequence)
            for i in range(1, np.random.randint(self.max_sample_sequence)):
                base_arr[i] = randint(0, len(self.pitches))
            yield base_arr

    def sample(self):
        for inp in self._sample_pitch_space():
            yield self.entropy_matrix[inp, :]


class Discriminator:
    def __init__(self):
        self.model = self.define_discriminator()

    def define_discriminator(in_shape=(128, 128, 1)):
        model = Sequential()
        model.add(
            Conv2D(16, (3, 3), strides=(2, 2), padding="same", input_shape=in_shape)
        )
        model.add(Dropout(0.5))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))
        model.add(Conv2D(16, (3, 3), strides=(2, 2), padding="same"))
        model.add(Dropout(0.5))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(1, activation="sigmoid"))
        opt = Adam(lr=0.00002, beta_1=0.5)
        model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
        return model


class Generator:
    def __init__(self):
        self.model = self.define_generator()

    def define_generator(latent_dim=10):
        model = Sequential()
        n_nodes = 128 * 53 * 53
        model.add(Dense(n_nodes, input_dim=latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((53, 53, 128)))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(64, (4, 4), strides=(2, 2), padding="same"))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())
        model.add(Conv2D(1, (7, 7), padding="same", activation="sigmoid"))
        return model


class ScaleGAN:
    def __init__(self):
        self.discriminator = Discriminator()
        self.generator = Generator()
        self.model = self.define_gan(self.generator, self.discriminator)

    def define_gan(self, generator, discriminator):
        discriminator.trainable = False
        model = Sequential()
        model.add(generator)
        model.add(discriminator)
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss="binary_crossentropy", optimizer=opt)
        return model
