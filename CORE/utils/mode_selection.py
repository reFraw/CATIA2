import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow

from .header import Headers
from .common_functions import clear_screen
from .common_functions import show_menu
from .colors import Bcolors
from .variables import main_variables as main_v
from .variables import path

def mode_selection():

	global main_v
	global path

	clear_screen()

	print(Bcolors.HEADER + Headers.MODE_HEADER + Bcolors.ENDC)
	print(Bcolors.OKCYAN)
	print('''
>>> Select one of the available modes\n
  [1] Training w/ Validation
  [2] Training w/ Testing
  [3] Testing
  [4] GradCAM - Gradient-Weighted Class Activation Mapping\n''')
	print(Bcolors.ENDC)

	inp =''

	while inp != '1' and inp != '2' and inp != '3' and inp != '4':

		inp = input(Headers.INPUT_CHASER)

		if inp == '1':
			if main_v['mode'] == 'test':
				main_v['color_mode'] = None

			main_v['mode'] = 'train-val'
			main_v['input_model_name'] = None

			val_split = 1.0
			while val_split <= 0.0 or val_split >= 1.0:
				try:
					val_split = float(input('\n>>> Insert validation split. It must be a number between 0 and 1 (extremes excluded).' + Headers.INPUT_CHASER))
					if val_split <= 0.0 or val_split >= 1.0:
						print('\n>>> Invalid input')
					else:
						main_v['val_split'] = val_split
				except:
					print('\n>>> Invalid input')

		elif inp == '2':
			if main_v['mode'] == 'test':
				main_v['color_mode'] = None
			main_v['mode'] = 'train-test'
			main_v['input_model_name'] = None

		elif inp == '3' or inp == '4':

			if len(os.listdir(path['models_saved'])) == 0:
				print('\n>>> No models available')

			else:
				chosen_model = ''
				model_available = [model for model in os.listdir(path['models_saved'])]
				if inp == '3':
					main_v['mode'] = 'test'
				else:
					main_v['mode'] = 'gradcam'
				print('\n>>> Select one of the available models:\n')
				for model in model_available:
					print(Bcolors.OKGREEN + '  ' + model + Bcolors.ENDC)
				chosen_model = input(Headers.INPUT_CHASER)
				while True:	
					if chosen_model not in model_available:
						print('>>> Invalid input')
						chosen_model = input(Headers.INPUT_CHASER)
					else:
						main_v['input_model_name'] = chosen_model
						main_v['input_model_path'] = os.path.join(path['models_saved'], chosen_model)

						index_name = main_v['input_model_name'].index('_m')
						input_name = main_v['input_model_name'][:index_name]
		
						index_img = main_v['input_model_name'].index('_i')
						image_size = main_v['input_model_name'][index_img + 2:]
						index_x = image_size.index('x')
						image_size = (int(image_size[:index_x]), int(image_size[:index_x]))
						main_v['image_size'] = image_size
						channels = main_v['input_model_name'][-1]
						main_v['channels'] = int(channels)
						if channels == '1':
							main_v['color_mode'] = 'grayscale'
						elif channels == '3':
							main_v['color_mode'] = 'rgb'
						elif channels == '4':
							main_v['color_mode'] = 'rgba'

						break

		else:
			print('\n>>> Invalid input')



	waiter = input('\n>>> Press ENTER to continue')

	clear_screen()
	show_menu()
