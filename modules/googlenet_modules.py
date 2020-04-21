import keras
from keras.models import Model
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Flatten, Concatenate
from keras import backend as K
from keras.regularizers import l2

class GoogLeNetModules:

	@staticmethod
	def Inception(input_tensor, filters, kernel_size, padding='same', dropout_rate):
		'''
		Returns an Inception layer, as proposed in the original paper.
		'''
		
		# 1st path - Projection 1x1 Conv
		path1 = Conv2D(filters=filters[0],
			kernel_size=(1, 1),
			padding=padding,
			kernel_regularizer=l2(0.0005))(input_tensor)
		path1 = Activation('relu')(path1)

		# 2nd path 1x1 -> 3x3
		path2 = Conv2D(filters=filters[1][0],
			kernel_size=(1, 1),
			padding=padding,
			kernel_regularizer=l2(0.0005))(input_tensor)
		path2 = Activation('relu')(path2)
		path2 = Conv2D(filters=filters[1][1],
			kernel_size=(3, 3),
			padding=padding,
			kernel_regularizer=l2(0.0005))(path2)
		path2 = Activation('relu')(path2)

		# 3rd path 1x1 -> 5x5
		path3 = Conv2D(filters=filters[2][0],
			kernel_size=(1, 1),
			padding=padding,
			kernel_regularizer=l2(0.0005))(input_tensor)
		path3 = Activation('relu')(path3)
		path3 = Conv2D(filters=filters[2][1],
			kernel_size=(5, 5),
			padding=padding,
			kernel_regularizer=l2(0.0005))(path3)
		path3 = Activation('relu')(path3)

		# 4th path MaxPooling -> Conv2D
		path4 = MaxPooling2D(pool_size=(3, 3),
			padding=padding)(input_tensor)
		path4 = Conv2D(filters=filters[3],
			kernel_size=(1, 1),
			padding=padding,
			kernel_regularizer=l2(0.0005))(path4)
		return Concatenate(axis=-1)([path1 path2 path3 path4])


