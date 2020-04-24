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

import numpy as np
import os

class ResNetModules:

	@staticmethod
	def resnet_layer(input_tensor, num_filters=16, kernel_size=3, strides=1, activation='relu', batch_normalization=True, conv_first=True):
		conv_tensor = Conv2D(filters=num_filters,
		kernel_size=kernel_size,
		strides=strides,
		padding='same',
		kernel_initializer='he_normal',
		kernel_regularizer=l2(1e-4))
		x = input_tensor
		if conv_first:
			x = conv_tensor(x)
			if batch_normalization:
				x = BatchNormalization()(x)
			if activation is not None:
				x = Activation(activation)(x)
		else:
			if batch_normalization:
				x = BatchNormalization()(x)
			if activation is not None:
				x = Activation(activation)(x)
			x = conv(x)
	
		return x
	
