import keras
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.core import Dense, Dropout, Activation
from keras import backend as K
from keras.regularizers import l2
from keras.layers import Input, Flatten
from keras.models import Model

from modules.googlenet_modules import GoogLeNetModules
from modules.common.custom_layers import LRN

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler

from keras.optimizers import SGD

import numpy as np

class GoogLeNetCIFAR:
	def __init__(self):
		self.input_shape = (32, 32, 3)
		self.num_classes = 10
		self.model = self.build_model()

	def build_model(self):
		inputs = Input(shape=self.input_shape)

		#===========================
		x = Conv2D(filters=64,
			kernel_size=(7, 7),
			strides=2,
			padding='same',
			kernel_regularizer=l2(0.0001))(inputs)
		x = Activation('relu')(x)
		x = MaxPooling2D(pool_size=(3, 3),
			strides=2)(x)
		x = LRN()(x)
		#===========================
		x = GoogLeNetModules.Inception(x, filters=[64, (96, 128), (16, 32), 32])
		x = GoogLeNetModules.Inception(x, filters=[128, (128, 192), (32, 96), 64])
		#===========================
		x = MaxPooling2D(pool_size=(3, 3),
			strides=2,
			padding='same')(x)
		x = GoogLeNetModules.Inception(x, filters=[192, (96, 208), (16, 48), 64])
		x = GoogLeNetModules.Inception(x, filters=[160, (112, 224), (24, 64), 64])
		x = GoogLeNetModules.Inception(x, filters=[128, (128, 256), (24, 64), 64])
		x = GoogLeNetModules.Inception(x, filters=[112, (144, 288), (32, 64), 64])
		x = GoogLeNetModules.Inception(x, filters=[256, (160, 320), (32, 128), 128])
		#===========================
		x = MaxPooling2D(pool_size=(3, 3),
			strides=2,
			padding='same')(x)
		x = GoogLeNetModules.Inception(x, filters=[256, (160, 320), (32, 128), 128])
		x = GoogLeNetModules.Inception(x, filters=[384, (192, 384), (48, 128), 128])
		# Smaller maxpooling
		x = AveragePooling2D(pool_size=(2, 2),
			strides=1,
			padding='valid')(x)

		x = Dropout(rate=0.4)(x)
		x = Flatten()(x)
		x = Dense(units=self.num_classes)(x)
		x = Activation('softmax')(x)

		model = Model(inputs=inputs, outputs=x, name="GoogLeNet_CIFAR-10")
		return model

	def train(self, learning_rate=0.1, epochs=150, batch_size=256, summary=False):
		lr_drop=20
		lr_decay = 1e-6
		# Download the dataset
		print("[INFO]: Downloading the dataset")
		(trainX, trainY), (testX, testY) = cifar10.load_data()

		# Normalise the dataset
		trainX = trainX.astype(np.float32)
		testX = testX.astype(np.float32)

		mean = np.mean(trainX, axis=(0, 1, 2, 3))
		std = np.std(trainX, axis=(0, 1, 2, 3))
		trainX = (trainX - mean) / (std + 1e-7)
		testX = (testX - mean) / (std + 1e-7)

		trainY = keras.utils.to_categorical(trainY, 10)
		testY = keras.utils.to_categorical(testY, 10)

		datagen = ImageDataGenerator(featurewise_center=False,
			samplewise_center=False,
			featurewise_std_normalization=False,
			samplewise_std_normalization=False,
			zca_whitening=False,
			rotation_range=15,
			width_shift_range=0.1,
			height_shift_range=0.1,
			horizontal_flip=True,
			vertical_flip=False)
		datagen.fit(trainX)

		def lr_scheduler(epoch):
			return learning_rate * (0.5 ** (epoch // lr_drop))
		callbacks = [LearningRateScheduler(lr_scheduler)]

		print("[INFO]: Compiling model")
		optimizer = SGD(lr=learning_rate, momentum=0.9, nesterov=True)
		self.model.compile(optimizer=optimizer,
			loss='categorical_crossentropy',
			metrics=['accuracy'])

		if summary:
			print("[INFO]: ========= MODEL SUMMARY  =========")
			self.model.summary()

		print("[INFO]: Training model")
		history = self.model.fit_generator(datagen.flow(trainX, trainY, batch_size=batch_size),
			steps_per_epoch=trainX.shape[0] // batch_size,
			epochs=epochs,
			validation_data=(testX, testY),
			callbacks=callbacks)

		self.model.save('CIFAR10_GoogLeNet.h5')
		return history

