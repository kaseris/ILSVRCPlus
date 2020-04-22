import keras
from keras.layers import Layer
from keras import backend as K
try:
	from keras import initializations
except:
	from keras import initializers as initializations

class LRN(Layer):

	def __init__(self, alpha=0.0001, k=1, beta=0.75, n=5, **kwargs):
		self.alpha = alpha
		self.k = k
		self.beta = beta
		self.n = n
		super(LRN, self).__init__(**kwargs)

	def get_output(self, train):

		X = self.get_input(train)
		b, ch, r, c = K.shape(X)
		half_n = self.n // 2
		input_sqr = K.square(X)
		extra_channels = K.zeros((b, ch + 2 * half_n, r, c))
		input_sqr = K.concatenate([extra_channels[:, :half_n, :, :],
									input_sqr,
									extra_channels[:, half_n + ch:, :, :]],
								   axis=1)
		scale = self.k
		for i in range(self.n):
			scale += self.alpha * input_sqr[:, i:i + ch, :, :]
		scale = scale ** self.beta
		return X / scale

	def get_config(self):
		config = {"alpha": self.alpha,
				  "k": self.k,
				  "beta": self.beta,
				  "n": self.n}
		base_config = super(LRN, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))