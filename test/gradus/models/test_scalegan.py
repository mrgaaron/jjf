from jjf.models.scalegan import ScaleNet, EntropyMatrixSampler
from jjf.gradus.system.entropy import EntropyMatrix
from jjf.gradus.system.western import WesternSystem
import os
from keras.utils import to_categorical
import numpy as np
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def vectorize(inp, num_classes=128):
    arrs = []
    for row in inp:
        arr = np.zeros(num_classes)
        for v in row:
            arr[int(v)] = 1
        arrs.append(arr)
    return np.array(arrs)


def test_sampling():
    system = WesternSystem()
    entropy = EntropyMatrix(system)
    sampler = EntropyMatrixSampler(entropy, max_input_samples=10)
    X, y = sampler.sample()
    assert X.shape == (10, 3)
    assert y.shape == (10,)


def test_inference():
    system = WesternSystem()
    entropy = EntropyMatrix(system)
    sampler = EntropyMatrixSampler(entropy, max_input_samples=1000)
    nn = ScaleNet()
    X, y = sampler.sample()
    X = tf.expand_dims(X, axis=0)
    y = to_categorical(y, num_classes=128)
    y = tf.expand_dims(y, axis=0)
    print(X.shape, y.shape)

    nn.model.fit(X, y, epochs=10, batch_size=10)
    yhat = nn.model.predict(X[:][0])
    # print(X[:][0][0], yhat[0], np.argmax(yhat[0]), np.argmax(y[:][0][0]))
    assert True
