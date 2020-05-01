# Import all libraries
import tensorflow as tf 
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

# Pickle is the python module that focuses on 'unpickling' 
# Here, a byte stream (from a binary file or bytes-like object) 
# is converted back into an object hierarchy.
import pickle 


pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)
pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)
# This scales the imagery data, which maxes out at 256
X = X/255.00 

# Building the model

model = "Sequential"
# Start with the convolutional layer
# Create window (3x3), and add input_shape (Keras element)
model.add(Conv2D(64), (3,3), input_shape = X.shape[1:]) 
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

# Repeat the above, input_shape isn't needed this time
model.add(Conv2D(64), (3,3)) 
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

# At this point, we've created a 256 x 64 
# Flatten the data - The Conv is 2D, but 
# the dense layer wants a 1D dataset (Not the band)
model.add(Flatten())
model.add(Dense(64))

# output layer
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy", 
                optimizer="adam", 
                metrics=['accuracy'])

# running the samples
model.fit(X, y, batch_size=32, epochs=3, validation_split=0.15)