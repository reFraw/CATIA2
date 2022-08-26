from .main_var import main_v 
from .colors import bcolors

import os
import platform

def check_system():

	sys = platform.system()

	return sys

def select_dataset(dataset_path, mode):

	sys = check_system()
	if sys == 'Windows':
		clear = 'cls'
	elif sys == 'Linux':
		clear = 'clear'	

	global main_v

	header = """\n     ______  _______ _______ _______  ______ _______ _______       
    (______)(_______|_______|_______)/ _____|_______|_______)      
     _     _ _______    _    _______( (____  _____      _          
    | |   | |  ___  |  | |  |  ___  |\____ \|  ___)    | |         
    | |__/ /| |   | |  | |  | |   | |_____) ) |_____   | |         
    |_____/ |_|   |_|  |_|  |_|   |_(______/|_______)  |_|         
                                                                   
  ______ _______ _       _______ _______ _______ _ _______ _______ 
 / _____|_______|_)     (_______|_______|_______) (_______|_______)
( (____  _____   _       _____   _          _   | |_     _ _     _ 
 \____ \|  ___) | |     |  ___) | |        | |  | | |   | | |   | |
 _____) ) |_____| |_____| |_____| |_____   | |  | | |___| | |   | |
(______/|_______)_______)_______)\______)  |_|  |_|\_____/|_|   |_|
                                                                   \n\n"""

	if len(os.listdir(dataset_path)) == 0:
		print('\n>>> No dataset available')
		os.system(clear)
		return False

	else:

		os.system(clear)
		print(bcolors.OKCYAN + header + bcolors.ENDC)

		print("\n>>> If you haven't chosen the mode yet, press ENTER and come back later.")
		print('\n>>> Select one of the availabe dataset:\n')

		for item in os.listdir(dataset_path):
			print(bcolors.OKGREEN + '\t' + item + bcolors.ENDC)

		data_name = input('\n<<< ')

		if data_name == '':
			pass
			os.system(clear)

		else:

			main_v['dataset_name'] = data_name

			data_path = os.path.join(dataset_path, data_name)
			train_path = os.path.join(data_path, 'training')
			test_path = os.path.join(data_path, 'test')
			try:
				main_v['num_classes'] = len(os.listdir(train_path))
			except:
				print('\n>>> This dataset has no classes. Please check the folder.')
				main_v['dataset_name'] = None
				waiter = input('\n>>> Press ENTER to continue...')
				os.system('clear')
				return True

			main_v['train_path'] = train_path
			main_v['test_path'] = test_path

			if mode == 'train-test' or mode == 'train-val':

				input_size = input('\n>>> Insert input size in the format SIZExCHANNELS.\n<<< ')

				try:
					image_size = int(input_size.split('x')[0])
					main_v['channels'] = int(input_size.split('x')[1])

					if main_v['channels'] == 1:
						main_v['color_mode'] = 'grayscale'
					elif main_v['channels'] == 3:
						main_v['color_mode'] = 'rgb'
					elif main_v['channels'] == 4:
						main_v['color_mode'] = 'rgba'

					main_v['input_net'] = (image_size, image_size, main_v['channels'])
					main_v['image_size'] = (image_size, image_size)

				except:
					print('\n>>> Invalid input format. Set default value 100x1.')
					main_v['input_net'] = (100, 100, 1)
					main_v['image_size'] = (100, 100)
					main_v['color_mode'] = 'grayscale'

				main_v['batch_size'] = int(input('\n>>> Insert the batch size\n<<< '))

				waiter = input('\n>>> Press ENTER to continue...')	

			os.system(clear)