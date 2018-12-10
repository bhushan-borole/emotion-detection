import tensorflow as tf
import pandas as pd
import keras
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Activation, Dropout, Flatten

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.backend.tensorflow_backend import set_session
import numpy as np
'''
Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
for this error the below 2 statements are included
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# CPU-GPU configuration
config = tf.ConfigProto(device_count = {'GPU': 0 , 'CPU': 56} ) #max: 1 gpu, 56 cpu
session = tf.Session(config=config) 
keras.backend.set_session(session)

# variables
num_expressions = 7 # angry, disgust, fear, happy, sad, surprise, neutral
batch_size = 256
epochs = 5


def train():
	x_train, y_train, x_test, y_test = [], [], [], []
	df = pd.read_csv('dataset/fer2013.csv')
	for i, row in df.iterrows():

		emotion, image, usage = row['emotion'], row['pixels'], row['usage']
		pixels = np.array(image.split(' '), 'float32')
		emotion = keras.utils.to_categorical(emotion, num_expressions)

		if 'Training' in usage:
			y_train.append(emotion)
			x_train.append(pixels)
		if 'PublicTest' in usage:
			y_test.append(emotion)
			x_test.append(pixels)

			

	print('Length: ', len(x_train))
	# data transformation to train and test sets
	x_train = np.array(x_train, 'float32')
	y_train = np.array(y_train, 'float32')
	x_test = np.array(x_test, 'float32')
	y_test = np.array(y_test, 'float32')

	x_train /= 255 #normalize inputs between [0, 1]
	x_test /= 255

	x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
	x_train = x_train.astype('float32')
	x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)
	x_test = x_test.astype('float32')

	print('Train Sample: {}'.format(x_train.shape[0]))
	print('Test Sample: {}'.format(x_test.shape[0]))

	# constructing the cnn structure
	model = Sequential()

	# 1st convolution layer
	'''
	relu : (Rectified Linear Unit) Activation Function
	https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6 
	'''
	model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48, 48, 1)))
	model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

	# 2nd Convolution Layer
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

	#3rd convolution layer
	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

	model.add(Flatten())

	# fully connected neural network
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.2))

	model.add(Dense(num_expressions, activation='softmax'))

	# batch process
	gen = ImageDataGenerator()
	train_generator = gen.flow(x_train, y_train, batch_size=batch_size)

	model.compile(
			loss='categorical_crossentropy',
			optimizer=keras.optimizers.Adam(),
			metrics=['accuracy']
		)
	model.fit_generator(train_generator, steps_per_epoch=batch_size, epochs=epochs)

	train_score = model.evaluate(x_train, y_train, verbose=0)
	print('Train loss:', train_score[0])
	print('Train accuracy:', 100*train_score[1])
	 
	test_score = model.evaluate(x_test, y_test, verbose=0)
	print('Test loss:', test_score[0])
	print('Test accuracy:', 100*test_score[1])

	model.save('my_model.hdf5')


if __name__ == '__main__':
	train()
