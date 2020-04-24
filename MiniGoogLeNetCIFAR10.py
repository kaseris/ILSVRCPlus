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
from keras.datasets import cifar10

import numpy as np
import os

from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

from modules.minigooglenet_modules import MiniGoogLeNetModules

class MiniGoogLeNetCIFAR:

	def __init__(self):
		self.input_shape = (32, 32, 3)
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
		x = AveragePooling2D(pool_size=(7, 7))(x)

		x = Flatten()(x)
		x = Dense(units=self.num_classes)(x)
		x = Activation('softmax')(x)
		model = Model(inputs=inputs, outputs=x, name='Mini GoogLeNet - CIFAR10')
		return model

	def train(self, init_lr=5e-3, epochs=200, batch_size=64):
		INIT_LR = init_lr

		print("[INFO]: Downloading the dataset")

		(trainX, trainY), (testX, testY) = cifar10.load_data()
		trainX = trainX.astype(np.float32)
		testX = testX.astype(np.float32)

		mean = np.mean(trainX, axis=0)
		trainX -= mean
		testX -= mean

		trainY = keras.utils.to_categorical(trainY, self.num_classes)
		testY = keras.utils.to_categorical(testY, self.num_classes)

		datagen = ImageDataGenerator(width_shift_range = 0.1,
			height_shift_range = 0.1,
			horizontal_flip = True,
			fill_mode = "nearest")

		def lr_scheduler(epoch):
			maxEpochs = epochs
			baseLR = INIT_LR
			power = 1.0

			alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power

			return alpha

		filepath=r"weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
		callbacks = [LearningRateScheduler(lr_scheduler),
					ModelCheckpoint(filepath, monitor='val_accuracy', save_best_only=True, mode='max')]

		print("[INFO]: Compiling model")
		optimizer = SGD(lr=INIT_LR, momentum=0.9)
		self.model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)

		print("[INFO]: Training Model")
		history=self.model.fit_generator(datagen.flow(trainX, trainY, batch_size),
			validation_data=(testX, testY),
			steps_per_epoch=len(trainX)//batch_size,
			epochs=epochs,
			callbacks=callbacks,
			verbose=1,
			workers=2)
		return history




