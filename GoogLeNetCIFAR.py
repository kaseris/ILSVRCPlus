import keras
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.regularizers import l2
from keras.layers import Input, Flatten
from keras.models import Model

from modules.googlenet_modules import GoogLeNetModules
from modules.common.custom_layers import LRN

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

from keras.optimizers import SGD, Adam

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
			kernel_regularizer=l2(0.0001))(inputs) # 16x16
		x = Activation('relu')(x)
		x = MaxPooling2D(pool_size=(3, 3),
			strides=2,
			padding='same')(x) # 8x8
		x = BatchNormalization()(x)

		# Stage 2
		x = Conv2D(filters=64,
			kernel_size=(1, 1),
			strides=1,
			padding='same',
			kernel_regularizer=l2(0.0001))(x)
		x = Activation('relu')(x)
		x = Conv2D(filters=192,
			kernel_size=(3, 3),
			strides=1,
			padding='same',
			kernel_regularizer=l2(1e-4))(x)
		x = BatchNormalization()(x)
		x = Activation('relu')(x)
		x = MaxPooling2D(pool_size=(3, 3),
			strides=2,
			padding='same')(x) #4x4
		#===========================
		x = GoogLeNetModules.Inception(x, filters=[64, (96, 128), (16, 32), 32]) # 3a
		x = GoogLeNetModules.Inception(x, filters=[128, (128, 192), (32, 96), 64]) # 3b
		#===========================
		x = MaxPooling2D(pool_size=(3, 3),
			strides=2,
			padding='same')(x) #2x2
		x = GoogLeNetModules.Inception(x, filters=[192, (96, 208), (16, 48), 64]) #4a
		aux1 = GoogLeNetModules.Auxillary(x, name='aux1')
		x = GoogLeNetModules.Inception(x, filters=[160, (112, 224), (24, 64), 64]) #4b
		x = GoogLeNetModules.Inception(x, filters=[128, (128, 256), (24, 64), 64]) #4c
		x = GoogLeNetModules.Inception(x, filters=[112, (144, 288), (32, 64), 64]) # 4d
		aux2 = GoogLeNetModules.Auxillary(x, name='aux2')
		x = GoogLeNetModules.Inception(x, filters=[256, (160, 320), (32, 128), 128]) # 4e
		#===========================
		x = MaxPooling2D(pool_size=(2, 2),
			strides=1,
			padding='same')(x)
		x = GoogLeNetModules.Inception(x, filters=[256, (160, 320), (32, 128), 128]) # 5a
		x = GoogLeNetModules.Inception(x, filters=[384, (192, 384), (48, 128), 128]) # 5b
		# Smaller avgpooling
		x = AveragePooling2D(pool_size=(2, 2),
			strides=1,
			padding='valid')(x)

		x = Dropout(rate=0.4)(x)
		x = Flatten()(x)
		#==================================
		# Addition
		# x = Dense(units=1024,
		# 	kernel_regularizer=l2(0.0005))(x)
		# x = Dropout(rate=0.5)(x)
		#==================================
		x = Dense(units=self.num_classes)(x)
		main = Activation('softmax', name='main')(x)

		model = Model(inputs=inputs, outputs=[main, aux1, aux2], name="GoogLeNet_CIFAR-10")

		return model

	def train(self, learning_rate=1e-3, epochs=200, batch_size=256, summary=False):
		lr_drop=20
		lr_decay = 1e-6
		# Download the dataset
		print("[INFO]: Downloading the dataset")
		(trainX, trainY), (testX, testY) = cifar10.load_data()

		trainX = trainX.astype(np.float32)
		testX = testX.astype(np.float32)

		trainY = keras.utils.to_categorical(trainY, 10)
		testY = keras.utils.to_categorical(testY, 10)

		train_datagen = ImageDataGenerator(rescale=1. / 255,
			shear_range=0.2,
			zoom_range=0.2,
			horizontal_flip=True,
			validation_split=0.2,
			rotation_range=15)

		train_datagen.fit(trainX)

		def multi_gen(datagen, batch_size, subset):
			gen = train_datagen.flow(trainX, trainY, batch_size,
				subset=subset)

			while True:
				gnext = gen.next()
				yield gnext[0], [gnext[1], gnext[1], gnext[1]]

		train_data_generator = multi_gen(train_datagen, batch_size, 'training')
		val_data_generator = multi_gen(train_datagen, batch_size, 'validation')
		train_samples = int(0.8 * trainX.shape[0])
		val_samples = int(0.2 * trainX.shape[0])

		# mean = np.mean(trainX, axis=0)
		
		# trainX = (trainX - mean)
		# testX = (testX - mean)

		# datagen = ImageDataGenerator(width_shift_range = 0.1,
		# 	height_shift_range = 0.1,
		# 	horizontal_flip = True,
		# 	fill_mode = "nearest")
		# datagen.fit(trainX)

		def lr_scheduler(epoch):
			if epoch < 150:
				return 1e-3
			elif epoch >= 150 or epoch <250:
				return 5e-4
			else:
				return 1e-4

		filepath=r"GoogLeNet-weights-improvement-{epoch:02d}-{val_main_accuracy:.2f}.hdf5"
		callbacks = [LearningRateScheduler(lr_scheduler),
		ModelCheckpoint(filepath, monitor='val_main_accuracy', save_best_only=True, mode='max')]

		print("[INFO]: Compiling model")
		optimizer = SGD(lr=lr_scheduler(0), momentum=0.9, nesterov=True)
		#optimizer = Adam(learning_rate=lr_scheduler(0))
		self.model.compile(optimizer=optimizer,
			loss='categorical_crossentropy',
			loss_weights = {'main': 1.0, 'aux1': 0.3, 'aux2': 0.3},
			metrics=['accuracy'])

		if summary:
			print("[INFO]: ========= MODEL SUMMARY  =========")
			self.model.summary()

		print("[INFO]: Training model")
		history = self.model.fit_generator(train_data_generator,
			epochs=epochs,
			steps_per_epoch=train_samples//batch_size,
			validation_data=val_data_generator,
			validation_steps=val_samples//batch_size,
			callbacks=callbacks)

		self.model.save('CIFAR10_GoogLeNet.h5')
		return history

