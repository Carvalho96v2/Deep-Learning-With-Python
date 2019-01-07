# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 13:20:25 2018

@author: JDeCarvalho
"""
#Imports
from keras.datasets import imdb
import numpy as np
from keras import models, layers
import matplotlib.pyplot as plt

#Loading the data
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

#Decoding reviews
word_index = imdb.get_word_index()
reverse_word_index = dict(
        [(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[5]])

#Data preprocessing
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] =1
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

#Building the network
model = models.Sequential()
model.add(layers.Dense(32, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(8, activation='tanh'))
model.add(layers.Dense(1, activation='sigmoid'))

#Training
model.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['accuracy'])

#Setting the validation set
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

#Fitting
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=4,
                    batch_size=512,
                    validation_data=(x_val, y_val))

#Plotting the training and validation loss
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1) 

plt.plot(epochs, loss_values, 'bo', label='Training Loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss.pdf', bbox_inches='tight')
plt.show()

#Plotting the training and validation accuracy
plt.clf() #clearing the plot
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc_values, 'bo', label='Training Accuracy') #bo = blue dots
plt.plot(epochs, val_acc_values, 'b', label='TValidation Accuracy') #b = blue line
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy.pdf', bbox_inches='tight')
plt.show()

#Evaluating the results
results = model.evaluate(x_test, y_test)
print(results)


