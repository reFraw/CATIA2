import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.models import Sequential
from keras.layers import Rescaling, Conv2D, LeakyReLU, AveragePooling2D, MaxPooling2D, Flatten, Dropout, Dense, BatchNormalization
from keras.metrics import Precision, Recall, AUC
from tensorflow.keras.optimizers import RMSprop, Adam

def build_RAVNET(input_net, num_classes, learning_rate):

	model = Sequential([
		Conv2D(32, (3, 3), activation='relu', input_shape=input_net),
		AveragePooling2D((2, 2)),
		BatchNormalization(),
		Conv2D(64, (3, 3), activation='relu'),
		AveragePooling2D((2, 2)),
		BatchNormalization(),
		Conv2D(128, (3, 3), activation='relu'),
		AveragePooling2D((2, 2)),
		BatchNormalization(),
		Conv2D(256, (3, 3), activation='relu'),
		AveragePooling2D((2, 2)),
		BatchNormalization(),
		Conv2D(512, (3, 3), activation='relu'),
		MaxPooling2D((2, 2)),
		BatchNormalization(),
		Flatten(),
		Dropout(0.5),
		Dense(256, activation='relu'),
		Dropout(0.5),
		Dense(128, activation='relu'),
		Dropout(0.5),
		Dense(num_classes, activation='softmax')], name='RAVNET')

	model.compile(
		loss='categorical_crossentropy',
		optimizer=Adam(learning_rate=learning_rate),
		metrics=['acc', Precision(name='prec'), Recall(name='rec'), AUC(name='auc')])

	return model