import keras
from keras.layers import Input, Flatten
from keras.layers.core import Dropout, Activation, Dense
from keras.models import Model
from modules.vgg_modules import VGGModules
from keras import backend as K

from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

from keras.optimizers import SGD

import numpy as np

class VGGNetMNIST:

	def __init__(self):
		self.input_shape = (28, 28, 1)
		self.num_classes = 10
		self.model = self.build_model()

	def build_model(self):
		inputs_shape = self.input_shape
		inputs = Input(shape=inputs_shape) #28x28

		x = VGGModules.convModule2(inputs, filters=64, kernel_size=(3, 3)) #14x14
		x = VGGModules.convModule2(x, filters=128, kernel_size=(3, 3)) # 7x7
		x = VGGModules.convModule3(x, filters=256, kernel_size=(3, 3)) # 3x3
		x = VGGModules.convModule3(x, filters=512, kernel_size=(3, 3))

		x = Flatten()(x)
		x = Dense(units=4096,
			kernel_regularizer=keras.regularizers.l2(0.0005))(x)
		x = Activation('relu')(x)

		x = Dense(units=4096,
			kernel_regularizer=keras.regularizers.l2(0.0005))(x)
		x = Activation('relu')(x)

		x = Dense(units=self.num_classes)(x)
		x = Activation('softmax')(x)

		model = Model(inputs=inputs, outputs=x, name='MNIST VGGNet')
		return model

	def train(self, learning_rate=1e-3, epochs=100, batch_size=128, summary=False):
		lr_drop=20
		lr_decay = 1e-6
		# Download the dataset
		print("[INFO]: Downloading the dataset")
		(trainX, trainY), (testX, testY) = mnist.load_data()

		# Normalise the dataset
		trainX = trainX.astype(np.float32)
		testX = testX.astype(np.float32)

		trainX = trainX / 255.0
		testX = testX / 255.0

		trainX = np.expand_dims(trainX, axis=-1)
		testX = np.expand_dims(testX, axis=-1)

		trainY = keras.utils.to_categorical(trainY, self.num_classes)
		testY = keras.utils.to_categorical(testY, self.num_classes)

		print("[INFO]: Compiling model")
		optimizer = SGD(lr=learning_rate, momentum=0.9, nesterov=True)
		self.model.compile(optimizer=optimizer,
			loss='categorical_crossentropy',
			metrics=['accuracy'])

		if summary:
			print("[INFO]: ========= MODEL SUMMARY ========")
			self.model.summary()
		
		filepath=r"MNISTVGGNet-weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
		callbacks = [ModelCheckpoint(filepath,
			monitor='val_accuracy',
			save_best_only=True,
			mode='max')]

		print("[INFO]: Training model")
		history = self.model.fit(trainX, trainY,
			epochs=epochs,
			validation_data=(testX, testY),
			callbacks=callbacks)

		return history



