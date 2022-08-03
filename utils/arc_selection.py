from .main_var import main_v 
from .colors import bcolors

import os

def select_arc(archi_path, mode):

	global main_v

	if mode == 'test':
		print('\n>>> Function not enabled in test mode.')

	else:
		print('\n>>> Select one of the available architecture: \n')

		for item in os.listdir(archi_path):
			if item.startswith('__'):
				pass
			else:
				print(bcolors.OKGREEN + '\t' + item + bcolors.ENDC)

		main_v['architecture'] = input('\n<<< ')
		main_v['learning_rate'] = float(input('\n>>> Set the learning rate\n<<< '))