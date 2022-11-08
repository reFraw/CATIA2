from .variables import path
from .variables import main_variables as main_v
from .variables import multi_variables as multi_v
from .colors import Bcolors
from .header import Headers

import platform
import os

def clear_screen():
	sys = platform.system()
	if sys == 'Windows':
		os.system('cls')
	else:
		os.system('clear')


def create_dirs():
	if not os.path.exists(path['dataset']):
		os.makedirs(path['dataset'])

	if not os.path.exists(path['models_saved']):
		os.makedirs(path['models_saved'])

	if not os.path.exists(path['plot']):
		os.makedirs(path['plot'])

	if not os.path.exists(path['report']):
		os.makedirs(path['report'])


def show_menu():
	print(Bcolors.HEADER)
	print(Headers.LOGO)
	print(Bcolors.ENDC)
	print(Bcolors.OKCYAN)
	print('     ========================= SELECT AN OPTION =========================\n')
	print('     [01] Mode Selection\t\t\t[P] Check Parameters')
	print('     [02] Dataset Selection\t\t\t[D] Open DATASETS Folder')
	print('     [03] Architecture Selection\t\t[R] Open RESULTS Folder')
	print('     [04] Start NN\t\t\t\t[M] Open MODELS_SAVED Folder\n')

	print('     [CC] Clear parameters\t\t\t[AA] Check Architecture')
	print('     [EE] CLose Program\t\t\t\t[OO] Save model\n')
	print(Bcolors.ENDC)


def check_parameters():

	global main_v

	if main_v['input_model_name'] != None:
		index_name = main_v['input_model_name'].index('_m')
		input_name = main_v['input_model_name'][:index_name]
		
		index_img = main_v['input_model_name'].index('_i')
		image_size = main_v['input_model_name'][index_img + 2:]
		index_x = image_size.index('x')
		image_size = (int(image_size[:index_x]), int(image_size[:index_x]))
		channels = main_v['input_model_name'][-1]
		if channels == '1':
			main_v['color_mode'] = 'grayscale'
		elif channels == '3':
			main_v['color_mode'] = 'rgb'
		elif channels == '4':
			main_v['color_mode'] = 'rgba'

	if main_v['mode'] is None:
		print('\n>>> Set mode first')

	elif main_v['mode'] == 'train-val':
		print(Bcolors.OKGREEN)
		print('\n# ============ PARAMETERS ============ #\n')
		print('Mode : {}'.format(main_v['mode']))
		print('Dataset : {}'.format(main_v['dataset']))
		print('Number of classes : {}'.format(main_v['num_classes']))
		print('Validation split : {}'.format(main_v['val_split']))
		print('Epochs : {}'.format(main_v['epochs']))
		print('Image size : {}'.format(main_v['image_size']))
		print('Color mode (Channels) : {} ({})'.format(main_v['color_mode'], main_v['channels']))
		print('Architecture : {}'.format(main_v['architecture']))
		print('Learning rate : {}'.format(main_v['learning_rate']))
		print('Batch size : {}'.format(main_v['batch_size']))
		print('Output model : {}'.format(main_v['output_model_name']))
		print(Bcolors.ENDC)

	elif main_v['mode'] == 'train-test':
		print(Bcolors.OKGREEN)
		print('\n# ============ PARAMETERS ============ #\n')
		print('Mode : {}'.format(main_v['mode']))
		print('Dataset : {}'.format(main_v['dataset']))
		print('Number of classes : {}'.format(main_v['num_classes']))
		print('Epochs : {}'.format(main_v['epochs']))
		print('Image size : {}'.format(main_v['image_size']))
		print('Color mode (Channels) : {} ({})'.format(main_v['color_mode'], main_v['channels']))
		print('Architecture : {}'.format(main_v['architecture']))
		print('Learning rate : {}'.format(main_v['learning_rate']))
		print('Batch size : {}'.format(main_v['batch_size']))
		print('Output model : {}'.format(main_v['output_model_name']))
		print(Bcolors.ENDC)

	elif main_v['mode'] == 'test':
		print(Bcolors.OKGREEN)
		print('\n# ============ PARAMETERS ============ #\n')
		print('Mode : {}'.format(main_v['mode']))
		print('Input model : {}'.format(input_name))
		print('Input model path : {}'.format(main_v['input_model_path']))
		print('Image size : {}'.format(image_size))
		print('Color mode (Channels) : {} ({})'.format(main_v['color_mode'], channels))
		print('Dataset : {}'.format(main_v['dataset']))
		print('Number of classes : {}'.format(main_v['num_classes']))
		print(Bcolors.ENDC)


def clear_parameters():

	for parameter in main_v:
		main_v[parameter] = None

	print(Bcolors.OKGREEN + '[*] Parameters cleared' + Bcolors.ENDC)


