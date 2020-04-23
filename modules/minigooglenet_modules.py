import keras
from keras.models import Model
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Flatten, Concatenate
from keras import backend as K
from keras.regularizers import l2
from keras.layers import concatenate

class MiniGoogLeNetModules:

	@staticmethod
	def ConvolutionalModule(x, num_filters, kernel_size, axis=-1, strides=1, padding='same'):
		x = Conv2D(filters=num_filters,
			kernel_size=kernel_size,
			strides=strides,
			padding=padding,
			kernel_regularizer=l2(1e-4))(x)
		x = BatchNormalization(axis=axis)(x)
		x = Activation('relu')(x)
		return x

	@staticmethod
	def InceptionModule(x, filters1x1, filters3x3, axis=-1):
		conv1x1 = MiniGoogLeNetModules.ConvolutionalModule(x, num_filters=filters1x1, kernel_size=(1, 1), strides=1, axis=-1)
		conv3x3 = MiniGoogLeNetModules.ConvolutionalModule(x, num_filters=filters1x1, kernel_size=(3, 3), strides=1, axis=-1)
		x = concatenate([conv1x1, conv3x3], axis=-1)
		return x

	@staticmethod
	def DownsamplingModule(x, num_filters):
		conv3x3 = MiniGoogLeNetModules.ConvolutionalModule(x, num_filters=num_filters, kernel_size=(3,3), strides=2, padding='valid')
		pool = MaxPooling2D(pool_size=(3, 3), strides=2)(x)
		x = concatenate([conv3x3, pool], axis=-1)
		return x