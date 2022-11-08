import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.models import Sequential
from keras.layers import Rescaling, Conv2D, LeakyReLU, AveragePooling2D, MaxPool2D, Flatten, Dropout, Dense, BatchNormalization
from keras.metrics import Precision, Recall, AUC
from tensorflow.keras.optimizers import RMSprop, Adam

def build_ALEXNET(input_net, num_classes, learning_rate):

	model = Sequential([
		Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=input_net),
		BatchNormalization(),
		MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
		Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same"),
		BatchNormalization(),
		MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
		Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
		BatchNormalization(),
		Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
		BatchNormalization(),
		Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
		BatchNormalization(),
		MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
		Flatten(),
		Dense(4096, activation='relu'),
		Dropout(0.5),
		Dense(4096, activation='relu'),
		Dropout(0.5),
		Dense(num_classes, activation='softmax')], name='ALEXNET')

	model.compile(
		loss='categorical_crossentropy',
		optimizer=Adam(learning_rate=learning_rate),
		metrics=['acc', Precision(name='prec'), Recall(name='rec'), AUC(name='auc')])

	return model