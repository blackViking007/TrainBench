from abc import ABCMeta, abstractmethod

class Autoencoder(object, metaclass = ABCMeta):
	"""Interface that classifiers must implement"""
	@abstractmethod
	def get(self, images, train_phase = False, l2_penalty = 0.0):
		"""Define the model with its inputs.
		Use this function to define the model in training and when exporting the model
		in the protobuf format

		Args:
			images: model input
			train_phase: set it to True when defining the model during training
			l2_penalty: float value, weight decay (l2) penalty

		Returns:
			is_training_: tf.bool placeholder enable/disable training ops at runtime
			predictions: the model output
		"""

	@abstractmethod
	def loss(self, predictions, real_values):
		"""Return the loss operation between predictions and real_values
		Args:
			predictions: predicted values
			labels: real_values

		Returns:
			Loss tensor of type float
		"""