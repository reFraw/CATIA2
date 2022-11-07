import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow

from .header import Headers
from .common_functions import clear_screen
from .common_functions import show_menu
from .colors import Bcolors
from .variables import main_variables as main_v
from .variables import path
from .variables import available_arc

def architecture_selection():

	global main_v

	if main_v['mode'] == 'test':
		print('\n>>> Function not enabled')

	elif main_v['mode'] is None:
		print('\n>>> Select mode first')

	elif main_v['mode'] == 'multi-train':
		clear_screen()
		print(Bcolors.HEADER + Headers.MODEL_HEADER + Bcolors.ENDC)
		print(Bcolors.OKCYAN + '\n>>> Chose one of the available model:\n' + Bcolors.ENDC)
		indexes = []
		for index, model in enumerate(available_arc, start=1):
			indexes.append(index)
			print(Bcolors.OKGREEN + '  [{}] '.format(index) + model + Bcolors.ENDC)

		model_choice = ''
		while model_choice not in indexes:
			try:
				model_choice = int(input(Headers.INPUT_CHASER))
				if model_choice not in indexes:
					print('\n>>> Invalid input')
			except:
				print('\n>>> Invalid input')

		if model_choice == 1:
			main_v['architecture'] = 'FCNN'
		elif model_choice == 2:
			main_v['architecture'] = 'FAB_CONVNET'
		elif model_choice == 3:
			main_v['architecture'] = 'RAVNET'
		elif model_choice == 4:
			main_v['architecture'] = 'SCNN'
		elif model_choice == 5:
			main_v['architecture'] = 'ALEXNET'
		elif model_choice == 6:
			main_v['architecture'] = 'LE_NET'

		waiter = input('\nPress ENTER to continue')

		clear_screen()
		show_menu()

	else:
		clear_screen()
		print(Bcolors.HEADER + Headers.MODEL_HEADER + Bcolors.ENDC)
		print(Bcolors.OKCYAN + '\n>>> Chose one of the available model:\n' + Bcolors.ENDC)
		indexes = []
		for index, model in enumerate(available_arc, start=1):
			indexes.append(index)
			print(Bcolors.OKGREEN + '  [{}] '.format(index) + model + Bcolors.ENDC)

		model_choice = ''
		while model_choice not in indexes:
			try:
				model_choice = int(input(Headers.INPUT_CHASER))
				if model_choice not in indexes:
					print('\n>>> Invalid input')
			except:
				print('\n>>> Invalid input')

		if model_choice == 1:
			main_v['architecture'] = 'FCNN'
		elif model_choice == 2:
			main_v['architecture'] = 'FAB_CONVNET'
		elif model_choice == 3:
			main_v['architecture'] = 'RAVNET'
		elif model_choice == 4:
			main_v['architecture'] = 'SCNN'
		elif model_choice == 5:
			main_v['architecture'] = 'ALEXNET'
		elif model_choice == 6:
			main_v['architecture'] = 'LE_NET'
		if model_choice == 7:
			main_v['architecture'] = 'FCNN-2'

		lr_choice = -0.1
		print('\n>>> Insert learning rate')
		while lr_choice <= 0:
			try:
				lr_choice = float(input(Headers.INPUT_CHASER))
				if lr_choice <= 0:
					print('\n>>> Invalid input')
				else:
					main_v['learning_rate'] = lr_choice
			except:
				print('\n>>> Invalid input')

		epochs = -1
		print('\n>>> Insert epochs number')
		while epochs <= 0:
			try:
				epochs = int(input(Headers.INPUT_CHASER))
				if epochs <= 0:
					print('\n>>> Invalid input')
				else:
					main_v['epochs'] = epochs
			except:
				print('\n>>> Invalid input')

		waiter = input('\nPress ENTER to continue')

		clear_screen()
		show_menu()