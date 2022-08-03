from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.optimizers import Adam

def FAB_CONVNET(input_net, num_classes, learning_rate):

	model = models.Sequential([
		layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_net),
		layers.AveragePooling2D((2, 2)),
		layers.Conv2D(64, (3, 3), activation='elu'),
		layers.MaxPooling2D((2, 2)),
		layers.Conv2D(128, (3, 3), activation='relu'),
		layers.AveragePooling2D((2, 2)),
		layers.Conv2D(256, (3, 3), activation='elu'),
		layers.AveragePooling2D((2, 2)),
		layers.Flatten(),
		layers.Dropout(0.3),
		layers.Dense(64, activation='relu'),
		layers.Dropout(0.3),
		layers.Dense(num_classes, activation='softmax')], name='FAB_CONVNET')

	model.compile(
		loss='categorical_crossentropy',
		optimizer=Adam(learning_rate),
		metrics=['acc', Precision(name='prec'), Recall(name='rec'), AUC(name='auc')])

	return model