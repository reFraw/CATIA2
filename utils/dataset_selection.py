from .variables import path
from .variables import main_variables as main_v
from .variables import multi_variables as multi_v
from .colors import Bcolors
from .header import Headers
from .common_functions import clear_screen
from .common_functions import show_menu

import os
import re

def dataset_selection():

	if len(os.listdir(path['dataset'])) == 0:
		print('\n>>> No dataset available')

	else:

		if main_v['mode'] == 'test' or main_v['mode'] == 'multi-train':
			clear_screen()
			print(Bcolors.HEADER)
			print(Headers.DATASET_HEADER)
			print(Bcolors.ENDC)
			dataset_available = [dataset for dataset in os.listdir(path['dataset'])]
			print('\n>>> Select one of the available dataset')
			print(Bcolors.OKGREEN)
			for item in dataset_available:
				print('  ' + item)
			print(Bcolors.ENDC)
			dataset_choice = ''
			while dataset_choice not in dataset_available:
				dataset_choice = input(Headers.INPUT_CHASER)
				if dataset_choice not in dataset_available:
					print('\n>>> Invalid input')
				else:
					main_v['dataset'] = dataset_choice
					main_v['num_classes'] = len(os.listdir(os.path.join(path['dataset'], dataset_choice, 'training')))
			waiter = input('\n>>> Press ENTER to continue')

			clear_screen()
			show_menu()

		elif main_v['mode'] == 'train-test' or main_v['mode'] == 'train-val':
			clear_screen()
			print(Bcolors.HEADER)
			print(Headers.DATASET_HEADER)
			print(Bcolors.ENDC)
			dataset_available = [dataset for dataset in os.listdir(path['dataset'])]
			print('\n>>> Select one of the available dataset')
			print(Bcolors.OKGREEN)
			for item in dataset_available:
				print('  ' + item)
			print(Bcolors.ENDC)
			dataset_choice = ''
			while dataset_choice not in dataset_available:

				dataset_choice = input(Headers.INPUT_CHASER)
				if dataset_choice not in dataset_available:
					print('\n>>> Invalid input')
				else:
					main_v['dataset'] = dataset_choice
					main_v['num_classes'] = len(os.listdir(os.path.join(path['dataset'], dataset_choice, 'training')))

			img_size = ''
			print('\n>>> Insert image size in [WIDTH]x[CHANNELS] format. (ex. 128x1)')
			while True:
				img_size = input(Headers.INPUT_CHASER)
				if re.match('^[0-9]+x1$', img_size) or re.match('^[0-9]+x[3-4]{1}$', img_size):
					image_wh = img_size.split('x')[0]
					main_v['image_size'] = (int(image_wh), int(image_wh))
					main_v['channels'] = int(img_size.split('x')[1])
					main_v['input_net'] = (int(image_wh), int(image_wh), main_v['channels'])
					if main_v['channels'] == 1:
						main_v['color_mode'] = 'grayscale'
					elif main_v['channels'] == 3:
						main_v['color_mode'] = 'rgb'
					else:
						main_v['color_mode'] = 'rgba'
					break
				else:
					print('\n>>> Invalid input')

			batch_size = -1
			while batch_size <= 0:
				try:
					batch_size = int(input('\n>>> Insert batch size\n' + Headers.INPUT_CHASER))
					if batch_size <= 0:
						print('\n>>> Invalid input')
					else:
						main_v['batch_size'] = batch_size
				except:
					print('\n>>> Invalid input')

			waiter = input('\n>>> Press ENTER to continue')

			clear_screen()
			show_menu()

		else:
			print('\n>>> Please select mode first')
