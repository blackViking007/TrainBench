from abc import ABCMeta, abstractmethod

class Classifier(object, metaclass = ABCMeta):
	"""Classifier is the interface that classifiers must implement"""

	@abstractmethod
	def get(self, images, num_classes, train_phase = False, l2_penalty = 0.0):
		"""Define the model with its inputs.
		Use this function to define the model in training and when exporting the model
		in the protobuf format.

		Args:
			images: model input
			num_classes: number of classes to predict
			train_phase: set it to True when defining the model during train
			l2_penalty: float value, weight decay (l2) penalty

		Returns:
			is_training_: tf.bool placeholder enable/disable training ops at run time
			logits: the model output
		"""

	@abstractmethod
	def loss(self, logits, labels):
		"""Returns the loss operation between logits and labels
		Args:
			logits: logits from inference()
			labels: Labels from distorted_inputs or inputs(). 1-D tensor
					of shape [batch_size]

		Returns:
			Loss tensor of type float"""