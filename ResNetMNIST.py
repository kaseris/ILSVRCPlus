import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10

from modules.resnet_modules import ResNetModules

import numpy as np
import os

class ResNetMNIST:

	def __init__(self, version=1, n=8):
		self.version = version
		self.input_shape = (28, 28, 1)
		self.num_classes = 10
		self.depth = n
		self.model = self.build_model()
		

	def build_model(self):
		if self.version == 1:
			return self.resnet_v1()
		else:
			return self.resnet_v2()

	def resnet_v1(self):
		if (self.depth - 2) % 6 != 0:
			raise ValueError('Depth should be 6n+2 (20, 26, 32, ..)')

		num_filters = 16
		num_res_blocks = int((self.depth - 2) / 6)
		inputs = Input(shape=self.input_shape)
		x = ResNetModules.resnet_layer(input_tensor=inputs)

		for stack in range(3):
			for res_block in range(num_res_blocks):
				strides = 1
				if stack > 0 and res_block == 0:
					strides = 2
				y = ResNetModules.resnet_layer(input_tensor=x, num_filters=num_filters, strides=strides)
				y = ResNetModules.resnet_layer(input_tensor=y, num_filters=num_filters, activation=None)
				if stack > 0 and res_block==0:
					x = ResNetModules.resnet_layer(input_tensor=x, num_filters=num_filters, kernel_size=1, strides=strides, activation=None, batch_normalization=False)
				x = keras.layers.add([x, y])
				x = Activation('relu')(x)
			num_filters *= 2

		x = AveragePooling2D(pool_size=7)(x)
		y = Flatten()(x)
		outputs = Dense(units=self.num_classes,
			activation='softmax',
			kernel_initializer='he_normal')(y)

		model = Model(inputs=inputs, outputs=outputs, name='ResNet v1 - MNIST')
		return model


	def resnet_v2(self):
		if (self.depth - 2) % 9 != 0:
			raise ValueError('Depth should be 9n+2 (56, 92, 101, ..)')

		num_filters_in = 16
		num_res_blocks = int((self.depth - 2) / 9)

		inputs = Input(shape=self.input_shape)
		x = ResNetModules.resnet_layer(input_tensor=inputs, num_filters=num_filters_in, conv_first=True)

		for stage in range(3):
			for res_block in range(num_res_blocks):
				activation = 'relu'
				batch_normalization = True
				strides = 1
				if stage==0:
					num_filters_out = num_filters_in * 4
					if res_block == 0:
						activation = None
						batch_normalization = False

				else:
					num_filters_out = num_filters_in * 2
					if res_block == 0:
						strides = 2

				y = ResNetModules.resnet_layer(input_tensor=x, num_filters=num_filters_in, strides=strides, kernel_size=1, activation=activation, batch_normalization=batch_normalization, conv_first=False)
				y = ResNetModules.resnet_layer(input_tensor=y, num_filters=num_filters_in, conv_first=False)
				y = ResNetModules.resnet_layer(input_tensor=y, num_filters=num_filters_out, kernel_size=1, conv_first=False)

				if res_block == 0:
					x = ResNetModules.resnet_layer(input_tensor=x, num_filters=num_filters_out, kernel_size=1, strides=strides, activation=None, batch_normalization=False)
				x = keras.layers.add([x, y])

			num_filters_in = num_filters_out

		x = BatchNormalization()(x)
		x = Activation('relu')(x)
		x = AveragePooling2D(pool_size=7)(x)
		y = Flatten()(x)
		outputs = Dense(units=self.num_classes, activation='softmax', kernel_initializer='he_normal')(y)

		model = Model(inputs=inputs, outputs=outputs, name='ResNet v2 - MNIST')
		return model

	def train(self, epochs=15, learning_rate=1e-3, batch_size=32, summary=False):
		self.model.compile(optimizer='adam',
			lr=lr_scheduler,
			metrics=['accuracy'])

		if summary:
			self.model.summary()

