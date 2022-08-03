import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow

from tensorflow.keras.utils import image_dataset_from_directory
from .main_var import main_v

def workflow(mode, model=None):

	global main_v

	if main_v['mode'] == 'test':

		input_model = tensorflow.keras.models.load_model(main_v['input_model'])

		print('\n ------- TEST DATA -------')

		test_data = image_dataset_from_directory(
			main_v['test_path'],
			labels='inferred',
			label_mode='categorical',
			color_mode=main_v['color_mode'],
			batch_size=main_v['batch_size'],
			image_size=main_v['image_size'],
			seed=0,
			shuffle=True)

		print('\n\n>>> Starting testing phase.\n')

		test_scores = input_model.evaluate(test_data)
		print('\n')

	elif main_v['mode'] == 'train-val':

		print('\n ------- TRAIN DATA -------')

		train_data = image_dataset_from_directory(
			main_v['train_path'],
			labels='inferred',
			label_mode='categorical',
			color_mode=main_v['color_mode'],
			batch_size=main_v['batch_size'],
			image_size=main_v['image_size'],
			validation_split=main_v['val_split'],
			subset='training',
			seed=0,
			shuffle=True)

		print('\n ------- VALID DATA -------')

		val_data = image_dataset_from_directory(
			main_v['train_path'],
			labels='inferred',
			label_mode='categorical',
			color_mode=main_v['color_mode'],
			batch_size=main_v['batch_size'],
			image_size=main_v['image_size'],
			validation_split=main_v['val_split'],
			subset='validation',
			seed=0,
			shuffle=True)

		print('\n\n>>> Starting training-valid phase.\n')

		history = model.fit(
			train_data,
			epochs=main_v['epochs'],
			validation_data=val_data)

		if main_v['output_model'] is not None:
			model.save(main_v['output_model'])

	elif main_v['mode'] == 'train-test':

		print('\n ------- TRAIN DATA -------')

		train_data = image_dataset_from_directory(
			main_v['train_path'],
			labels='inferred',
			label_mode='categorical',
			color_mode=main_v['color_mode'],
			batch_size=main_v['batch_size'],
			image_size=main_v['image_size'],
			seed=0,
			shuffle=True)

		print('\n ------- TEST DATA -------')

		test_data = image_dataset_from_directory(
			main_v['test_path'],
			labels='inferred',
			label_mode='categorical',
			color_mode=main_v['color_mode'],
			batch_size=main_v['batch_size'],
			image_size=main_v['image_size'],
			seed=0,
			shuffle=True)

		print('\n\n>>> Starting training phase.\n')

		history = model.fit(
			train_data,
			epochs=main_v['epochs'])

		print('\n\n>>> Starting testing phase.\n')

		test_scores = model.evaluate(test_data)

		if main_v['output_model'] is not None:
			model.save(main_v['output_model'])

