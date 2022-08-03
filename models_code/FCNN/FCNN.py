from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.optimizers import Adam

def FCNN(input_net, num_classes, learning_rate):

	model = models.Sequential([
		layers.Conv2D(64, (3, 3), input_shape=input_net),
		layers.LeakyReLU(alpha=0.2),
		layers.AveragePooling2D((2, 2)),
		layers.Conv2D(128, (3, 3)),
		layers.LeakyReLU(alpha=0.2),
		layers.AveragePooling2D((2, 2)),
		layers.Conv2D(128, (3, 3), activation='relu'),
		layers.MaxPooling2D((2, 2)),
		layers.Flatten(),
		layers.Dropout(0.5),
		layers.Dense(256, activation='relu'),
		layers.Dropout(0.5),
		layers.Dense(64, activation='relu'),
		layers.Dense(num_classes, activation='softmax')], name='FCNN')

	model.compile(
		loss='categorical_crossentropy',
		optimizer=Adam(learning_rate),
		metrics = ['acc', Precision(name='prec'), Recall(name='rec'), AUC(name='auc')])

	return model