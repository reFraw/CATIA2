import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow
from tensorflow.keras.utils import image_dataset_from_directory as img_data
from datetime import datetime

from .variables import main_variables as main_v
from .variables import path
from .colors import Bcolors

from .models_code.FCNN import build_FCNN
from .models_code.FCNN_2 import build_FCNN2
from .models_code.FAB_CONVNET import build_FABCONVNET
from .models_code.RAVNET import build_RAVNET
from .models_code.SCNN import build_SCNN
from .models_code.ALEXNET import build_ALEXNET
from .models_code.LE_NET import build_LENET
from .common_functions import clear_screen, show_menu
from .results import save_report, save_graph

def check_start():

	train_check = main_v['dataset'] != None and main_v['architecture'] != None
	test_check = main_v['dataset'] != None

	if main_v['mode'] == 'test' and test_check:
		return True

	elif (main_v['mode'] == 'train-val' or main_v['mode'] == 'train-test') and train_check:
		return True

	else:
		return False


def startNN():

	try:

		date = datetime.now()
		date = date.strftime('_%d%m%Y-%H%M%S')

		if main_v['mode'] != 'test':
			if main_v['architecture'] == 'FCNN':
				model = build_FCNN(main_v['input_net'], main_v['num_classes'], main_v['learning_rate'])
			elif main_v['architecture'] == 'FCNN-2':
				model = build_FCNN2(main_v['input_net'], main_v['num_classes'], main_v['learning_rate'])
			elif main_v['architecture'] == 'FAB_CONVNET':
				model = build_FABCONVNET(main_v['input_net'], main_v['num_classes'], main_v['learning_rate'])
			elif main_v['architecture'] == 'RAVNET':
				model = build_RAVNET(main_v['input_net'], main_v['num_classes'], main_v['learning_rate'])
			elif main_v['architecture'] == 'SCNN':
				model = build_SCNN(main_v['input_net'], main_v['num_classes'], main_v['learning_rate'])
			elif main_v['architecture'] == 'ALEXNET':
				model = build_ALEXNET(main_v['input_net'], main_v['num_classes'], main_v['learning_rate'])
			elif main_v['architecture'] == 'LE_NET':
				model = build_LENET(main_v['input_net'], main_v['num_classes'], main_v['learning_rate'])

			if main_v['output_model_name'] != None:
				output_name = main_v['output_model_name'] + '_m' + main_v['architecture'] + '_i' + str(main_v['input_net'][0]) + 'x' + str(main_v['channels'])
				output_path = os.path.join(path['models_saved'], output_name)
				main_v['output_model_path'] = output_path

		# ==============================================================================================================================================

		if main_v['mode'] == 'test':
			print('\n[*] Loading model {}'.format(main_v['input_model_name']))
			loaded_model = tensorflow.keras.models.load_model(main_v['input_model_path'])

			test_path = os.path.join(path['dataset'], main_v['dataset'], 'test')

			print('[*] Checking for testing data\n')
			print('[TESTING DATA]')
			test_data = img_data(
				test_path,
				labels='inferred',
				label_mode='categorical',
				color_mode=main_v['color_mode'],
				batch_size = 1,
				image_size=main_v['image_size'],
				interpolation='area')

			print('\n[*] Starting execution at {}\n'.format(datetime.now()))

			test_results = loaded_model.evaluate(test_data, return_dict=True)

			save_report(test_results=test_results, time=date)

			waiter = input('\n>>> Press ENTER to continue')
			for item in main_v:
				main_v[item] = None

			clear_screen()
			show_menu()

		# ==============================================================================================================================================

		if main_v['mode'] == 'train-val':

			train_path = os.path.join(path['dataset'], main_v['dataset'], 'training')

			print('\n[*] Checking for training and validation data\n')
			print('[TRAIN DATA]')
			train_data = img_data(
				train_path,
				labels='inferred',
				label_mode='categorical',
				color_mode=main_v['color_mode'],
				batch_size=main_v['batch_size'],
				interpolation='area',
				image_size=main_v['image_size'],
				validation_split=main_v['val_split'],
				subset='training',
				seed=42,
				shuffle=True)

			print('\n[VALIDATION DATA]')
			val_data = img_data(
				train_path,
				labels='inferred',
				label_mode='categorical',
				color_mode=main_v['color_mode'],
				batch_size=main_v['batch_size'],
				interpolation='area',
				image_size=main_v['image_size'],
				validation_split=main_v['val_split'],
				subset='validation',
				seed=42,
				shuffle=True)


			print('\n[*] Building model')
			print('[*] Starting execution at {}\n'.format(datetime.now()))

			history = model.fit(
				train_data,
				validation_data = val_data,
				validation_freq = 1,
				epochs = main_v['epochs'])

			if main_v['output_model_name'] != None:
				print('\n[*] Saving model {}'.format(output_name))
				model.save(main_v['output_model_path'])

			save_report(history, date)
			save_graph(history, date)

			waiter = input('\n>>> Press ENTER to continue')
			for item in main_v:
				main_v[item] = None

			clear_screen()
			show_menu()

		# ==============================================================================================================================================
		
		elif main_v['mode'] == 'train-test':

			train_path = os.path.join(path['dataset'], main_v['dataset'], 'training')
			test_path = os.path.join(path['dataset'], main_v['dataset'], 'test')

			print('\n[*] Checking for training and testing data\n')
			print('[TRAIN DATA]')
			train_data = img_data(
				train_path,
				labels='inferred',
				label_mode='categorical',
				color_mode=main_v['color_mode'],
				batch_size=main_v['batch_size'],
				interpolation='area',
				image_size=main_v['image_size'],
				seed=42,
				shuffle=True)

			print('\n[TESTING DATA]')
			test_data = img_data(
				test_path,
				labels='inferred',
				label_mode='categorical',
				color_mode=main_v['color_mode'],
				batch_size = 1,
				image_size=main_v['image_size'],
				interpolation='area')

			print('\n[*] Building model')
			print('[*] Starting execution at {}\n'.format(datetime.now()))

			history = model.fit(
				train_data,
				epochs = main_v['epochs'])

			if main_v['output_model_name'] != None:
				print('\n[*] Saving model {}'.format(output_name))
				model.save(main_v['output_model_path'])

			print('\n[*] Starting testing phase\n')
			test_results = model.evaluate(test_data)

			save_report(history, date, test_results)
			save_graph(history, date)

			waiter = input('\n>>> Press ENTER to continue')
			for item in main_v:
				main_v[item] = None

			clear_screen()
			show_menu()

	except Exception as e:
		print('\n>>> Error occurred :')
		print(Bcolors.FAIL)
		print(e)
		print(Bcolors.ENDC)