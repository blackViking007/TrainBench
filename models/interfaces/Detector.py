from abc import ABCMeta, abstractmethod

class Detector(object, metaclass = ABCMeta):
	"""Detector is the interface that detectors must implement"""

	@abstractmethod
	def get(self, images, num_classes, train_phase = False, l2_penalty = 0.0):
		"""Define the model with its inputs
		Use this function to define the model in training and when exporting the model
		in the protobuf format

		Args:
			images: model input, tensor with batch_size elements
			num_classes: number of classes to predict
			train_pahse: set it to True when defining the model during train
			l2_penalty: float value, weight decay (l2) penalty

		Returns:
			is_training: tf.bool placeholder enable/disable training ops at run time
			logits: the unscaled prediction for a class specific detector
			bboxes: the predicted coordinates for every detected object in the input image
					this must have the same number of rows of logits
		"""

	@abstractmethod
	def loss(self, label_relations, bboxes_relations):
		"""Return the loss operation
		Args:
			label_relations: a tuple with 2 elements, usually the pair (labels, logits),
							each one a tensor of batch_size elements
			bboxes_realtions: a tuple with 2 elements, usually the pair (coordinates, bboxes)
							where coordinates are the ground truth coordinates and 
							bboxes, the predicted ones

		Returns:
			Loss tensor of type float,
		"""