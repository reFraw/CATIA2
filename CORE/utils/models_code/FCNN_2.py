import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.layers import Input, Rescaling, Conv2D, LeakyReLU, AveragePooling2D, MaxPooling2D, Flatten, Dropout, Dense, concatenate
from keras.metrics import Precision, Recall, AUC
from keras.models import Model
from tensorflow.keras.optimizers import RMSprop

def build_FCNN2(input_net, num_classes, learning_rate):

	visible = Input(shape=input_net)

	# Branch - 1
	y1 = Conv2D(32, (3, 3)) (visible)
	y1 = LeakyReLU(alpha=0.2) (y1)
	y1 = AveragePooling2D((2, 2)) (y1)
	y1 = Conv2D(64, (3, 3)) (y1)
	y1 = LeakyReLU(alpha=0.2) (y1)
	y1 = AveragePooling2D((2, 2)) (y1)
	y1 = Conv2D(64, (3, 3), activation='relu') (y1)
	y1 = MaxPooling2D((2, 2)) (y1)

	flat1 = Flatten() (y1)

	# Branch - 2
	y2 = Conv2D(32, (7, 7)) (visible)
	y2 = LeakyReLU(alpha=0.2) (y2)
	y2 = AveragePooling2D((2, 2)) (y2)
	y2 = Conv2D(64, (7, 7)) (y2)
	y2 = LeakyReLU(alpha=0.2) (y2)
	y2 = AveragePooling2D((2, 2)) (y2)
	y2 = Conv2D(64, (7, 7), activation='relu') (y2)
	y2 = MaxPooling2D((2, 2)) (y2)

	flat2 = Flatten() (y2)

	# Final classificator
	merged = concatenate([flat1, flat2])
	hidden = Dense(256, activation='relu') (merged)
	hidden = Dropout(0.5) (hidden)
	hidden = Dense(64, activation='relu') (hidden)
	hidden = Dropout(0.5) (hidden)

	output = Dense(num_classes, activation='softmax') (hidden)

	# Compiling
	model = Model(inputs=visible, outputs=output, name='FCNN-2')

	model.compile(
		loss='categorical_crossentropy',
		optimizer=RMSprop(learning_rate=learning_rate),
		metrics=['acc', Precision(name='prec'), Recall(name='rec'), AUC(name='auc')])

	return model
