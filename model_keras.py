from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Embedding, merge, Activation
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.advanced_activations import PReLU, ELU
from keras.layers.normalization import BatchNormalization
from utils import *

###############################################################################
# Keras
def create_keras_model(num_teams, num_features):
    print('Creating keras model with %d teams and %d features.' % (num_teams, num_features))
    model = Sequential()
    model.add(Dense(12, input_dim=num_features, init='uniform'))
    model.add(BatchNormalization())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer="sgd")
    return model


#
#
def estimate_keras(model, X, y, X_ncaa, w=None):
    index_count = len(game_keys)
    model.fit(X.values[:, index_count:], y, batch_size=1024, sample_weight=w,
        nb_epoch = epoch_count, validation_split=0.1, verbose=2)
    y_fit = model.predict(X_ncaa.values[:, index_count:], batch_size=1024, verbose=2)[:, 0]
    return y_fit


