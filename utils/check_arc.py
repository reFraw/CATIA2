import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from .variables import available_arc
from .colors import Bcolors
from .header import Headers
from .models_code.FCNN import build_FCNN
from .models_code.FCNN_2 import build_FCNN2
from .models_code.FAB_CONVNET import build_FABCONVNET
from .models_code.RAVNET import build_RAVNET
from .models_code.SCNN import build_SCNN
from .models_code.ALEXNET import build_ALEXNET
from .models_code.LE_NET import build_LENET
from .common_functions import clear_screen, show_menu

from time import sleep

def check_architecture():

	clear_screen()

	print(Bcolors.HEADER + Headers.MODEL_CHECK + Bcolors.ENDC)

	print('\n>>> For checking architectures will be used as test values:\n')
	print('  Image size : (100, 100)')
	print('  Channels : 1')
	print('  Number of classes : 2')
	print('\n>>> Select one of the available architectures\n')
	indexes = []
	for index, model in enumerate(available_arc, start=1):
		indexes.append(index)
		print(Bcolors.OKGREEN + '  [{}] {}'.format(index, model) + Bcolors.ENDC)

	arc_choice = ''

	while arc_choice not in indexes:
		try:
			arc_choice = int(input(Headers.INPUT_CHASER))
			if arc_choice not in indexes:
				print('\n>>> Invalid input')
		except:
			print('\n>>> Invalid input')

	print('\n[*] Cheking model')
	for i in range(2):
		sleep(1)
		print('[*] Cheking model')

	print('\n>>> MODEL SUMMARY:\n')
	print(Bcolors.OKGREEN)

	if arc_choice == 1:
		model = build_FCNN((100, 100, 1), 2, 0.001)
		model.summary()
	elif arc_choice == 2:
		model = build_FABCONVNET((100, 100, 1), 2, 0.001)
		model.summary()
	elif arc_choice == 3:
		model = build_RAVNET((100, 100, 1), 2, 0.001)
		model.summary()
	elif arc_choice == 4:
		model = build_SCNN((100, 100, 1), 2, 0.001)
		model.summary()
	elif arc_choice == 5:
		model = build_ALEXNET((100, 100, 1), 2, 0.001)
		model.summary()
	elif arc_choice == 6:
		model = build_LENET((100, 100, 1), 2, 0.001)
		model.summary()
	if arc_choice == 7:
		model = build_FCNN2((100, 100, 1), 2, 0.001)
		model.summary()

	print(Bcolors.ENDC)

	waiter = input('\n>>> Press ENTER to continue')

	clear_screen()
	show_menu()
