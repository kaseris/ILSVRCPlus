import keras
from keras.models import Model
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Flatten, Concatenate
from keras import backend as K
from keras.regularizers import l2

class GoogLeNetModules:

	@staticmethod
	def Inception(input_tensor, filters, padding='same'):
		'''
		Returns an Inception layer, as proposed in the original paper.
		'''
		
		# 1st path - Projection 1x1 Conv
		path1 = Conv2D(filters=filters[0],
			kernel_size=(1, 1),
			padding=padding,
			kernel_regularizer=l2(0.0001),
			strides=1)(input_tensor)
		path1 = Activation('relu')(path1)

		# 2nd path 1x1 -> 3x3
		path2 = Conv2D(filters=filters[1][0],
			kernel_size=(1, 1),
			padding=padding,
			kernel_regularizer=l2(0.0001),
			strides=1)(input_tensor)
		path2 = Activation('relu')(path2)
		path2 = Conv2D(filters=filters[1][1],
			kernel_size=(3, 3),
			padding=padding,
			kernel_regularizer=l2(0.0001),
			strides=1)(path2)
		path2 = Activation('relu')(path2)

		# 3rd path 1x1 -> 5x5
		path3 = Conv2D(filters=filters[2][0],
			kernel_size=(1, 1),
			padding=padding,
			kernel_regularizer=l2(0.0001),
			strides=1)(input_tensor)
		path3 = Activation('relu')(path3)
		path3 = Conv2D(filters=filters[2][1],
			kernel_size=(5, 5),
			padding=padding,
			kernel_regularizer=l2(0.0001),
			strides=1)(path3)
		path3 = Activation('relu')(path3)

		# 4th path MaxPooling -> Conv2D
		path4 = MaxPooling2D(pool_size=(3, 3),
			padding=padding,
			strides=1)(input_tensor)
		path4 = Conv2D(filters=filters[3],
			kernel_size=(1, 1),
			padding=padding,
			kernel_regularizer=l2(0.0001),
			strides=1)(path4)
		return Concatenate(axis=-1)([path1, path2, path3, path4])

	@staticmethod
	def Auxillary(x, num_classes=10):
		x = AveragePooling2D(pool_size=(5, 5),
			padding='valid',
			strides=3)(x)
		x = Conv2D(filters=128,
			kernel_size=(1, 1),
			padding='same',
			kernel_regularizer=l2(0.0001))(x)
		x = Activation('relu')(x)

		x = Flatten()(x)

		x = Dense(units=1024,
			kernel_regularizer=l2(0.0001))(x)
		x = Activation('relu')(x)
		x = Dropout(rate=0.7)(x)
		x = Dense(units=num_classes,
			kernel_regularizer=l2(0.0001))(x)
		x = Activation('softmax')(x)
		return x



