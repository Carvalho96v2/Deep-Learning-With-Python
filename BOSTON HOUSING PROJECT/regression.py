# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 17:23:19 2018

@author: JDeCarvalho
"""
#Imports
from keras.datasets import boston_housing
from keras import models, layers
import numpy as np

#Fetching data
(train_data, train_targets),(test_data, test_targets) = boston_housing.load_data()

#Exploring the datasets
train_data[0]
train_data.shape
train_data.ndim

train_targets[0]
train_targets.shape
train_targets.ndim
train_targets.mean()
train_targets.max()
train_targets.min()

#Data normalisation
mean = train_data.mean(axis=0)
train_data -= mean
standard_deviation = train_data.std(axis=0)
train_data /= standard_deviation

test_data -=mean
test_data /= standard_deviation

#Building the network
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, 
                           activation='relu',
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))  #No activation - linear layer
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

#K-fold validation
k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []
all_mae_histories = [] 
for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i+1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    
    partial_train_data = np.concatenate(
            [train_data[:i * num_val_samples],
             train_data[(i+1) * num_val_samples:]],
             axis=0)
    
    partial_train_targets = np.concatenate(
            [train_targets[:i * num_val_samples],
             train_targets[(i+1) * num_val_samples:]],
             axis=0)

    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets,
              epochs=num_epochs, batch_size=1,
              verbose=0)
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)
    