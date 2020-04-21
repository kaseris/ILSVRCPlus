import keras
from keras.models import Model
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Flatten
from keras import backend as K
from keras.regularizers import l2

class VGG_v2Modules:

	@staticmethod
	def convModule(x, filters, kernel_size, dropout_rate):
		'''
		Creates a convolutional layer consisted of two Convolutional modules
		with ReLU activation in the following pattern:
		Conv2D->ReLU->BN->Dropout->Conv2D->ReLU->BN->MaxPool
		'''
		# 1st Convolution
		x = Conv2D(filters=filters,
			kernel_size=kernel_size,
			padding='same',
			kernel_regularizer=l2(0.0005))(x)
		x = Activation('relu')(x)
		x = BatchNormalization()(x)
		x = Dropout(rate=dropout_rate)(x)

		# 2nd Convolution
		x = Conv2D(filters=filters,
			kernel_size=kernel_size,
			padding='same',
			kernel_regularizer=l2(0.0005))(x)
		x = Activation('relu')(x)
		x = BatchNormalization()(x)

		x = MaxPooling2D(pool_size=(2, 2))(x)
		return x

	@staticmethod
	def convModule3(x, filters, kernel_size, dropout_rate):
		'''
		Creates a convolutional layer consisted of three Convolutional modules
		with ReLU activation in the following pattern:
		Conv2D->ReLU->BN->Dropout->Conv2D->ReLU->BN->Dropout->
		->Conv2D->ReLU->BN->MaxPool
		'''

		# 1st Convolution
		x = Conv2D(filters=filters,
			kernel_size=kernel_size,
			padding='same',
			kernel_regularizer=l2(0.0005))(x)
		x = Activation('relu')(x)
		x = BatchNormalization()(x)
		x = Dropout(rate=dropout_rate)(x)

		# 2nd Convolution
		x = Conv2D(filters=filters,
			kernel_size=kernel_size,
			padding='same',
			kernel_regularizer=l2(0.0005))(x)
		x = Activation('relu')(x)
		x = BatchNormalization()(x)
		x = Dropout(rate=dropout_rate)(x)

		# 3rd Convolution
		x = Conv2D(filters=filters,
			kernel_size=kernel_size,
			padding='same',
			kernel_regularizer=l2(0.0005))(x)
		x = Activation('relu')(x)
		x = BatchNormalization()(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)

		return x

	@staticmethod
	def fcModule(x, units, dropout_rate):
		x = Dense(units=units,
			kernel_regularizer=l2(0.0005))(x)
		x = Activation('relu')(x)
		x = BatchNormalization()(x)
		x = Dropout(rate=dropout_rate)(x)
		return x

class VGGModules:

	@staticmethod
	def convModule(x, filters, kernel_size):
		'''
		Creates a layer of the pattern, as suggested in the original publication:
		Conv2D -> MaxPool2D

		The convolution modules are followed by a ReLU activation function.
		'''
		x = Conv2D(filters=filters,
			kernel_size=kernel_size,
			padding='same',
			kernel_regularizer=l2(0.0005))(x)
		x = Activation('relu')(x)

		x = MaxPooling2D(pool_size=(2, 2))(x)
		return x

	@staticmethod
	def convModule2(x, filters, kernel_size):
		'''
		Creates a layer of the pattern, as suggested in the original publication:
		Conv2D -> Conv2D -> MaxPool2D

		The convolution modules are followed by a ReLU activation function.
		'''
		x = Conv2D(filters=filters,
			kernel_size=kernel_size,
			kernel_regularizer=l2(0.0005),
			padding='same')(x)
		x = Activation('relu')(x)

		x = Conv2D(filters=filters,
			kernel_size=kernel_size,
			kernel_regularizer=l2(0.0005),
			padding='same')(x)
		x = Activation('relu')(x)

		x = MaxPooling2D(pool_size=(2, 2))(x)
		return x

	@staticmethod
	def convModule3(x, filters, kernel_size):
		'''
		Creates a layer of the pattern, as suggested in the original publication:
		Conv2D -> Conv2D -> Conv2D -> MaxPool2D

		The convolution modules are followed by a ReLU activation function.
		'''
		x = Conv2D(filters=filters,
			kernel_size=kernel_size,
			padding='same',
			kernel_regularizer=l2(0.0005))(x)

		x = Activation('relu')(x)

		x = Conv2D(filters=filters,
			kernel_size=kernel_size,
			padding='same',
			kernel_regularizer=l2(0.0005))(x)

		x = Activation('relu')(x)

		x = Conv2D(filters=filters,
			kernel_size=kernel_size,
			padding='same',
			kernel_regularizer=l2(0.0005))(x)

		x = Activation('relu')(x)

		x = MaxPooling2D(pool_size=(2, 2))(x)
		return x