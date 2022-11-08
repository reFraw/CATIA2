#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import re
import tensorflow

from utils.header import Headers
from utils.colors import Bcolors
from utils.variables import main_variables as main_v
from utils.variables import multi_variables as multi_v
from utils.variables import path
from utils.common_functions import clear_screen
from utils.common_functions import create_dirs
from utils.common_functions import show_menu
from utils.common_functions import clear_parameters
from utils.common_functions import check_parameters
from utils.common_functions import in_wsl
from utils.mode_selection import mode_selection
from utils.dataset_selection import dataset_selection
from utils.arc_selection import architecture_selection
from utils.check_arc import check_architecture
from utils.start import check_start
from utils.start import startNN

if __name__ == '__main__':
	
	clear_screen()

	ROOT = os.path.abspath(__file__)
	FILENAME = os.path.basename(__file__)
	ROOT = ROOT.replace(FILENAME, '')

	path['dataset'] = os.path.join(ROOT, 'DATASETS')
	path['models_saved'] = os.path.join(ROOT, 'models_saved')
	path['results'] = os.path.join(ROOT, 'results')
	path['plot'] = os.path.join(ROOT, 'results', 'plot')
	path['report'] = os.path.join(ROOT, 'results', 'report')

	create_dirs()
	show_menu()
	
	onWSL = in_wsl()

	while True:
		chaser = input(Headers.INPUT_CHASER)

		if chaser.lower() == 'ee':
			waiter = input('\n>>> Press any key to close the program...')
			clear_screen()
			print(quit())

		elif chaser.lower() == 'oo':
			if main_v['mode'] == 'test':
				print('\n>>> Function not enabled')
			elif main_v['mode'] is None:
				print('\n>>> Set mode first')
			else:
				print('\n>>> Insert output model name')
				main_v['output_model_name'] = input(Headers.INPUT_CHASER)

		elif chaser == '1' or chaser == '01':
			mode_selection()

		elif chaser == '2' or chaser == '02':
			dataset_selection()

		elif chaser == '3' or chaser == '03':
			architecture_selection()

		elif chaser == '4' or chaser == '04':
			if check_start():
				startNN()
			else:
				print('\n>>> Please check parameters')

		elif chaser == 'cc':
			clear_parameters()

		elif chaser.lower() == 'aa':
			check_architecture()

		elif chaser.lower() == 'p':
			check_parameters()

		elif chaser.lower() == 'd':
			os.chdir(path['dataset'])
			if onWSL:
				os.system('explorer.exe .')
			else:
				os.system('open .')
			os.chdir(ROOT)

		elif chaser.lower() == 'm':
			os.chdir(path['models_saved'])
			if onWSL:
				os.system('explorer.exe .')
			else:
				os.system('open .')
			os.chdir(ROOT)

		elif chaser.lower() == 'r':
			os.chdir(path['results'])
			if onWSL:
				os.system('explorer.exe .')
			else:
				os.system('open .')
			os.chdir(ROOT)

		else:
			print('>>> Invalid input')
