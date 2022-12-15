import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

from keras.metrics import Precision, Recall, AUC
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

from tensorflow.keras.applications import VGG16, VGG19, ResNet50, DenseNet121, EfficientNetB0, InceptionV3, Xception

def build_SOTA(model, input_net, num_classes, learning_rate):

	if model == 'VGG16':
		model = VGG16(include_top=True, weights=None, input_shape=input_net, classes=num_classes, classifier_activation='softmax')

		model.compile(
			loss=CategoricalCrossentropy(),
			optimizer=Adam(learning_rate=learning_rate),
			metrics=["acc", Precision(name="prec"), Recall(name="rec"), AUC(name="auc")])

	elif model == 'VGG19':
		model = VGG19(include_top=True, weights=None, input_shape=input_net, classes=num_classes, classifier_activation='softmax')

		model.compile(
			loss=CategoricalCrossentropy(),
			optimizer=Adam(learning_rate=learning_rate),
			metrics=["acc", Precision(name="prec"), Recall(name="rec"), AUC(name="auc")])

	elif model == 'ResNet50':
		model = ResNet50(include_top=True, weights=None, input_shape=input_net, classes=num_classes, classifier_activation='softmax')

		model.compile(
			loss=CategoricalCrossentropy(),
			optimizer=Adam(learning_rate=learning_rate),
			metrics=["acc", Precision(name="prec"), Recall(name="rec"), AUC(name="auc")])

	elif model == 'EfficientNetB0':
		model = EfficientNetB0(include_top=True, weights=None, input_shape=input_net, classes=num_classes, classifier_activation='softmax')

		model.compile(
			loss=CategoricalCrossentropy(),
			optimizer=Adam(learning_rate=learning_rate),
			metrics=["acc", Precision(name="prec"), Recall(name="rec"), AUC(name="auc")])

	elif model == 'InceptionV3':
		model = InceptionV3(include_top=True, weights=None, input_shape=input_net, classes=num_classes, classifier_activation='softmax')

		model.compile(
			loss=CategoricalCrossentropy(),
			optimizer=Adam(learning_rate=learning_rate),
			metrics=["acc", Precision(name="prec"), Recall(name="rec"), AUC(name="auc")])

	elif model == 'Xception':
		model = Xception(include_top=True, weights=None, input_shape=input_net, classes=num_classes, classifier_activation='softmax')

		model.compile(
			loss=CategoricalCrossentropy(),
			optimizer=Adam(learning_rate=learning_rate),
			metrics=["acc", Precision(name="prec"), Recall(name="rec"), AUC(name="auc")])

	elif model == 'DenseNet121':
		model = DenseNet121(include_top=True, weights=None, input_shape=input_net, classes=num_classes, classifier_activation='softmax')

		model.compile(
			loss=CategoricalCrossentropy(),
			optimizer=Adam(learning_rate=learning_rate),
			metrics=["acc", Precision(name="prec"), Recall(name="rec"), AUC(name="auc")])


	return model