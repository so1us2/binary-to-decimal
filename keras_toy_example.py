import numpy as np
from keras.layers import Input, Dense, Activation
from keras.models import Model

from bin2dec_utils import *

# A toy example to use a neural network to fit the y = x^2 function.

X_test = np.array([[1],[2],[3],[4],[5]]) / 5
Y_test = np.array([[1],[4],[9],[16],[25]]) / 25

model_input = Input(shape=(1,))
X = Dense(10)(model_input)
X = Dense(10)(X)
X = Dense(10)(X)
X = Dense(1)(X)

model = Model(inputs=model_input, outputs=X)
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])

model.fit(X_test, Y_test, epochs=500, batch_size=5)