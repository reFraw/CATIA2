from .main_var import main_v 
from .colors import bcolors
import time

import os

def select_arc(archi_path, mode):

	global main_v

	header = """\n _______ ______ _______ _     _ _ _______ _______ _______ _______ _     _ ______  _______ 
(_______|_____ (_______|_)   (_) (_______|_______|_______|_______|_)   (_|_____ \(_______)
 _______ _____) )       _______| |   _    _____   _          _    _     _ _____) )_____   
|  ___  |  __  / |     |  ___  | |  | |  |  ___) | |        | |  | |   | |  __  /|  ___)  
| |   | | |  \ \ |_____| |   | | |  | |  | |_____| |_____   | |  | |___| | |  \ \| |_____ 
|_|   |_|_|   |_\______)_|   |_|_|  |_|  |_______)\______)  |_|   \_____/|_|   |_|_______)
                                                                                          
          ______ _______ _       _______ _______ _______ _ _______ _______                
         / _____|_______|_)     (_______|_______|_______) (_______|_______)               
        ( (____  _____   _       _____   _          _   | |_     _ _     _                
         \____ \|  ___) | |     |  ___) | |        | |  | | |   | | |   | |               
         _____) ) |_____| |_____| |_____| |_____   | |  | | |___| | |   | |               
        (______/|_______)_______)_______)\______)  |_|  |_|\_____/|_|   |_|               
                                                                                          \n\n"""


	if mode == 'test':
		print('\n>>> Function not enabled in test mode.')
		return False

	else:

		os.system('clear')
		print(bcolors.OKCYAN + header + bcolors.ENDC)

		print('\n>>> Select one of the available architecture: \n')

		for item in os.listdir(archi_path):
			if item.startswith('__'):
				pass
			else:
				print(bcolors.OKGREEN + '\t' + item + bcolors.ENDC)

		main_v['architecture'] = input('\n<<< ')

		try:
			main_v['learning_rate'] = float(input('\n>>> Set the learning rate\n<<< '))
		except:
			print('\n>>> Invalid iput value. Set default value 0.01')
			main_v['learning_rate'] = 0.01

		try:
			main_v['epochs'] = int(input('\n>>> Insert epochs for training\n<<< '))
		except:
			print('\n>>> Invalid iput value. Set default value 10')
			main_v['epochs'] = 10

		waiter = input('\n>>> Press ENTER to continue...')

		os.system('clear')

		return True