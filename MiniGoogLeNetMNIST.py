import keras
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers import concatenate
from keras import backend as K

from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist

import numpy as np
import os

from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

from modules.minigooglenet_modules import MiniGoogLeNetModules

class MiniGoogLeNetMNIST:
	def __init__(self):
		self.input_shape = (28, 28, 1)
		self.num_classes = 10
		self.model = self.build_model()

	def build_model(self):

		inputs = Input(shape=self.input_shape)

		x = MiniGoogLeNetModules.ConvolutionalModule(inputs, num_filters=96, kernel_size=(3, 3), strides=1)

		x = MiniGoogLeNetModules.InceptionModule(x, filters1x1=32, filters3x3=32)
		x = MiniGoogLeNetModules.InceptionModule(x, filters1x1=32, filters3x3=48)
		x = MiniGoogLeNetModules.DownsamplingModule(x, num_filters=80)

		x = MiniGoogLeNetModules.InceptionModule(x, filters1x1=112, filters3x3=48)
		x = MiniGoogLeNetModules.InceptionModule(x, filters1x1=96, filters3x3=64)
		x = MiniGoogLeNetModules.InceptionModule(x, filters1x1=80, filters3x3=80)
		x = MiniGoogLeNetModules.InceptionModule(x, filters1x1=48, filters3x3=96)
		x = MiniGoogLeNetModules.DownsamplingModule(x, num_filters=96)

		x = MiniGoogLeNetModules.InceptionModule(x, filters1x1=176, filters3x3=160)
		x = MiniGoogLeNetModules.InceptionModule(x, filters1x1=176, filters3x3=160)
		x = AveragePooling2D(pool_size=(6, 6))(x)

		x = Dropout(rate=0.6)(x)
		x = Flatten()(x)
		x = Dense(units=self.num_classes)(x)
		x = Activation('softmax')(x)
		model = Model(inputs=inputs, outputs=x, name='Mini GoogLeNet - MNIST')
		return model

	def train(self, init_lr=1e-3, epochs=15, batch_size=100, summary=False):
		INIT_LR=init_lr

		print("[INFO]: Downloading dataset")
		(trainX, trainY), (testX, testY) = mnist.load_data()
		trainX = trainX.astype(np.float32)
		testX = testX.astype(np.float32)

		trainX = np.expand_dims(trainX, axis=-1)
		testX = np.expand_dims(testX, axis=-1)

		trainY = keras.utils.to_categorical(trainY, self.num_classes)
		testY = keras.utils.to_categorical(testY, self.num_classes)

		datagen = ImageDataGenerator(rescale=1.0/255.0)

		print("[INFO]: Compiling model")
		optimizer = SGD(lr=INIT_LR, momentum=0.9)
		self.model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)

		if summary:
			print("[INFO]: ==================MODEL SUMMARY==================")
			self.model.summary()

		def lr_scheduler(epoch):
			maxEpochs = epochs
			baseLR = INIT_LR
			power = 1.0

			alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power

			return alpha

		#======================================================
		#Callback setup
		filepath=r"MNISTMiniGoogLeNet-weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
		callbacks = [LearningRateScheduler(lr_scheduler),
					ModelCheckpoint(filepath, monitor='val_accuracy', save_best_only=True, mode='max')]
		#======================================================

		print("[INFO]: Training model")
		history = self.model.fit(trainX, trainY,
			validation_data=(testX, testY),	
			epochs=epochs,
			verbose=1,
			workers=2,
			callbacks=callbacks)
		return history