import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.models import Sequential
from keras.layers import Rescaling, Conv2D, LeakyReLU, AveragePooling2D, MaxPooling2D, Flatten, Dropout, Dense, Activation
from keras.metrics import Precision, Recall, AUC
from tensorflow.keras.optimizers import RMSprop, Adam

def build_LENET(input_net, num_classes, learning_rate):

	model = Sequential([
		Conv2D(6, 5, activation='tanh', input_shape=input_net),
		AveragePooling2D(2),
		Activation('sigmoid'),
		Conv2D(120, 5, activation='tanh'),
		Flatten(),
		Dense(84, activation='tanh'),
		Dense(num_classes, activation='softmax')],name='LE_NET')

	model.compile(
		loss='categorical_crossentropy',
		optimizer=Adam(learning_rate=learning_rate),
		metrics=['acc', Precision(name='prec'), Recall(name='rec'), AUC(name='auc')])

	return model