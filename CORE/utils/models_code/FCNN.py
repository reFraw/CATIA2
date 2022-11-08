import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.models import Sequential
from keras.layers import Rescaling, Conv2D, LeakyReLU, AveragePooling2D, MaxPooling2D, Flatten, Dropout, Dense
from keras.metrics import Precision, Recall, AUC
from tensorflow.keras.optimizers import RMSprop, Adam

def build_FCNN(input_net, num_classes, learning_rate):

	model = Sequential([
		Rescaling(scale=1./255, input_shape=input_net, name='Rescaling'),
		Conv2D(64, (3, 3), name='Conv_1'),
		LeakyReLU(alpha=0.2, name='Leaky_ReLU_1'),
		AveragePooling2D((2, 2), name='AveragePooling2D_1'),
		Conv2D(128, (3, 3), name='Conv_2'),
		LeakyReLU(alpha=0.2, name='Leaky_ReLU_2'),
		AveragePooling2D((2, 2), name='AveragePooling2D_2'),
		Conv2D(128, (3, 3), activation='relu', name='Conv_3_with_ReLU'),
		MaxPooling2D((2, 2), name='MaxPooling2D'),
		Flatten(name='Flatten'),
		Dropout(0.5, name='Dropout_1'),
		Dense(256, activation='relu', name='Dense_with_ReLU_1'),
		Dropout(0.5, name='Dropout_2'),
		Dense(64, activation='relu', name='Dense_with_ReLU_2'),
		Dense(num_classes, activation='softmax', name='Dense_with_Softmax')], name='FCNN')

	model.compile(
		loss='categorical_crossentropy',
		optimizer=RMSprop(learning_rate=learning_rate),
		metrics=['acc', Precision(name='prec'), Recall(name='rec'), AUC(name='auc')])

	return model