# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 20:24:12 2018

@author: JDeCarvalho
"""
#Imports
from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
import matplotlib.pyplot as plt

#Loading the MNIST dataset from KERAS
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#Data pre-processing
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255

#Preparing the labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#Network Architecture
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))

#Compilation 
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

#Training the network
network.fit(train_images, train_labels, epochs=5, batch_size=128)

#Testing accuracy on test set
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc: ', test_acc)

#Visualising the data - displaying a digit
digit = train_images[4] #in this case, we'll be displaying the 4th digit
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()


#Testing on my own image
import matplotlib.image as img
image = img.imread('three.png')
print(image.shape)
image =image.reshape(3,28*28)
plt.imshow(image, cmap=plt.cm.binary)







