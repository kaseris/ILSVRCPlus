from keras.layers import Input, Flatten
from keras.layers.core import Dropout, Activation, Dense
from keras.models import Model
from modules.vgg_modules import VGGModules
from keras import backend as K

class VGGNetMNIST:

	def __init__(self):
		self.input_shape = (28, 28, 1)
		self.num_classes = 10
		self.model = self.build_model()

	def build_model(self):
		inputs_shape = self.input_shape
		inputs = Input(shape=inputs_shape)

		x = VGGModules.convModule(inputs, filters=64, kernel_size=(3, 3), dropout_rate=0.3)
		x = VGGModules.convModule(x, filters=128, kernel_size=(3, 3), dropout_rate=0.3)

		x = Dropout(rate=0.5)(x)
		x = Flatten()(x)

		x = VGGModules.fcModule(x, 512, 0.5)
		x = Dense(units=self.num_classes)(x)
		x = Activation('softmax')(x)

		model = Model(inputs=inputs, outputs=x, name='CIFAR10_VGGNet')
		return model
