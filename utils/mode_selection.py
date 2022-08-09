import os
import platform

from .colors import bcolors
from .main_var import main_v

def check_system():

	sys = platform.system()

	return sys

def select_mode(current_path):

	sys = check_system()
	if sys == 'Windows':
		clear = 'cls'
	elif sys == 'Linux':
		clear = 'clear'

	global main_v

	header = """\n               _______ _______ ______  _______                     
              (_______|_______|______)(_______)                    
               _  _  _ _     _ _     _ _____                       
              | ||_|| | |   | | |   | |  ___)                      
              | |   | | |___| | |__/ /| |_____                     
              |_|   |_|\_____/|_____/ |_______)                    
                                                                   
  ______ _______ _       _______ _______ _______ _ _______ _______ 
 / _____|_______|_)     (_______|_______|_______) (_______|_______)
( (____  _____   _       _____   _          _   | |_     _ _     _ 
 \____ \|  ___) | |     |  ___) | |        | |  | | |   | | |   | |
 _____) ) |_____| |_____| |_____| |_____   | |  | | |___| | |   | |
(______/|_______)_______)_______)\______)  |_|  |_|\_____/|_|   |_|
                                                                   \n\n"""

	os.system(clear)
	print(bcolors.OKCYAN + header + bcolors.ENDC)

	mode = input('>>> Insert mode between [TRAIN-VAL], [train-test], [test].\n<<< ')

	if mode == '' or mode.lower() == 'train-val':
		main_v['mode'] = 'train-val'

		try:
			main_v['val_split'] = float(input('\n>>> Enter the validation split. Press ENTER to set default value 0.2.\n<<< '))
		except:
			main_v['val_split'] = 0.2

		if main_v['val_split'] <= 0 or main_v['val_split'] >= 1:
			print('\n>>> Invalid input. Set default value.')

			main_v['val_split'] = 0.2


	elif mode.lower() == 'train-test':
		main_v['mode'] = 'train-test'


	elif mode.lower() == 'test':
		main_v['mode'] = 'test'

		if len(os.listdir(current_path)) == 0:
			print('\n>>> No model available.')

		else:
			print('\n>>> Select one of the available models:\n')

			for item in os.listdir(current_path):
				print(bcolors.OKGREEN + '\t' + item + bcolors.ENDC)

			main_v['input_model_name'] = input('\n<<< ')

			try:
				index = main_v['input_model_name'].index('_i')
				input_size = main_v['input_model_name'][index + 2 :]
				image_size = int(input_size.split('x')[0])
				main_v['channels'] = int(input_size.split('x')[1])

				if main_v['channels'] == 1:
					main_v['color_mode'] = 'grayscale'
				elif main_v['channels'] == 3:
					main_v['color_mode'] = 'rgb'
				elif main_v['channels'] == 4:
					main_v['color_mode'] = 'rgba'

				main_v['batch_size'] = 32
				main_v['input_net'] = (image_size, image_size, main_v['channels'])
				main_v['image_size'] = (image_size, image_size)

				input_model = os.path.join(current_path, main_v['input_model_name'])
				main_v['input_model'] = input_model

			except:
				print('\n>>> Wrong model name.')

	else:
		print('\n>>> Invalid input. Set default value.')
		main_v['mode'] = 'train-val'

		try:
			main_v['val_split'] = float(input('\n>>> Enter the validation split. Press ENTER to set default value 0.2.\n<<< '))
		except:
			main_v['val_split'] = 0.2

		if main_v['val_split'] <= 0 or main_v['val_split'] >= 1:
			print('\n>>> Invalid input. Set default value.')
			main_v['val_split'] = 0.2

	waiter = input('\n>>> Press ENTER to continue...')

	os.system(clear)