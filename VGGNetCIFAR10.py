import keras
from keras.layers import Input, Flatten
from keras.layers.core import Dropout, Activation, Dense
from keras.models import Model
from modules.vgg_modules import VGG_v2Modules
from keras import backend as K

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler

from keras.optimizers import SGD

import numpy as np

class VGGNetCIFAR:

	def __init__(self):
		self.input_shape = (32, 32, 3)
		self.num_classes = 10
		self.model = self.build_model()

	def build_model(self):
		inputs_shape = self.input_shape
		inputs = Input(shape=inputs_shape)

		x = VGG_v2Modules.convModule(inputs, filters=64, kernel_size=(3, 3), dropout_rate=0.3)
		x = VGG_v2Modules.convModule(x, filters=128, kernel_size=(3, 3), dropout_rate=0.4)
		x = VGG_v2Modules.convModule3(x, filters=256, kernel_size=(3, 3), dropout_rate=0.4)
		x = VGG_v2Modules.convModule3(x, filters=512, kernel_size=(3, 3), dropout_rate=0.4)
		x = VGG_v2Modules.convModule3(x, filters=512, kernel_size=(3, 3), dropout_rate=0.4)

		x = Dropout(rate=0.5)(x)
		x = Flatten()(x)

		x = VGG_v2Modules.fcModule(x, 512, 0.5)

		x = Dense(units=self.num_classes)(x)
		x = Activation('softmax')(x)

		model = Model(inputs=inputs, outputs=x, name='CIFAR10_VGGNet')
		return model

	def train(self, learning_rate=1e-2, epochs=100, batch_size=128, summary=False):
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
		#callbacks = [LearningRateScheduler(lr_scheduler)]

		print("[INFO]: Compiling model")
		optimizer = SGD(lr=learning_rate, momentum=0.9, nesterov=True)
		self.model.compile(optimizer=optimizer,
			loss='categorical_crossentropy',
			metrics=['accuracy'])

		if summary:
			print("[INFO]: ========= MODEL SUMMARY  =========")
			self.model.summary()

		filepath=r"CIFAR-VGGNet-weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
		callbacks = [ModelCheckpoint(filepath,
			monitor='val_accuracy',
			save_best_only=True,
			mode='max'), LearningRateScheduler(lr_scheduler)]

		print("[INFO]: Training model")
		history = self.model.fit_generator(datagen.flow(trainX, trainY, batch_size=batch_size),
			steps_per_epoch=trainX.shape[0] // batch_size,
			epochs=epochs,
			validation_data=(testX, testY),
			callbacks=callbacks)

		self.model.save('CIFAR10_VGGNet.h5')
		return history



