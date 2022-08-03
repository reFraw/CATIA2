import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow

from utils.main_var import main_v
from utils.colors import bcolors
from utils.colors import header
from utils.mode_selection import select_mode
from utils.dataset_generation import select_dataset
from utils.arc_selection import select_arc
from utils.start_flow import workflow

from models_code.FCNN.FCNN import FCNN
from models_code.FAB_CONVNET.FAB_CONVNET import FAB_CONVNET

if __name__ == '__main__':

	current_path = os.getcwd()

	dataset_path = os.path.join(current_path, 'DATASETS')
	model_saved_path = os.path.join(current_path, 'models_saved')
	models_code_path = os.path.join(current_path, 'models_code')

	# ========== HEADER ========== #

	print('\n' + bcolors.HEADER +  header + bcolors.ENDC)
	print(bcolors.HEADER + '\t\tCNNs Aggregator Tool for Image Analysis 2\n\n' + bcolors.ENDC)
	print(bcolors.WARNING + '\t\t\tInsert EXIT to quit the program\n' + bcolors.ENDC)

	print(bcolors.OKCYAN + '\t\t\t' + '{1} --- Mode selection' + bcolors.ENDC)
	print(bcolors.OKCYAN + '\t\t\t' + '{2} --- Dataset generation' + bcolors.ENDC)
	print(bcolors.OKCYAN + '\t\t\t' + '{3} --- Architecture selection\n' + bcolors.ENDC)
	print(bcolors.OKCYAN + '\t\t\t' + '{4} --- Start the CNN\n\n' + bcolors.ENDC)

	# ========== MAIN ========== #

	chaser = '_'

	while chaser != 'EXIT':

		chaser = input(bcolors.WARNING + '\n>>> Select mode\n<<< ' + bcolors.ENDC)

		if chaser == '1':
			print(bcolors.OKCYAN + '\n----- MODE SELECTION -----\n' + bcolors.ENDC)
			select_mode(model_saved_path)
			print(bcolors.OKCYAN + '\n--------------------------\n' + bcolors.ENDC)

		elif chaser == '2':
			print(bcolors.OKCYAN + '\n----- DATASET GENERATION -----' + bcolors.ENDC)
			select_dataset(dataset_path, main_v['mode'])
			print(bcolors.OKCYAN + '\n------------------------------\n' + bcolors.ENDC)

		elif chaser == '3':
			print(bcolors.OKCYAN + '\n----- ARCHITECTURE SELECTION -----' + bcolors.ENDC)
			select_arc(models_code_path, main_v['mode'])
			print(bcolors.OKCYAN + '\n----------------------------------\n' + bcolors.ENDC)

		elif chaser == '4':

			if main_v['mode'] == 'train-val' or main_v['mode'] == 'train-test':

				main_v['input_model'] = None

				main_v['epochs'] = int(input('\n>>> Insert epochs for training\n<<< '))

				save_check = '_'

				while save_check.lower() != 'y' and save_check.lower() != 'n':
					save_check = input('\n>>> Save the model? [y/n]\n<<< ')

				if save_check == 'y':

					output_model = input('\n>>> Insert model name\n<<< ')
					output_model = output_model + '_m' + main_v['architecture'] + '_i' + str(main_v['input_net'][0]) + 'x' + str(main_v['input_net'][2])
					output_model = os.path.join(model_saved_path, output_model)
					main_v['output_model'] = output_model

				check_ok = '_'

				print('\n>>> Parameters: \n')

				for item in main_v:
					if item.startswith('train') or item.startswith('test'):
						pass
					else:
						print(bcolors.OKGREEN + '{} --> {}\n'.format(item, main_v[item]) + bcolors.ENDC)

				while check_ok.lower() != 'y' and check_ok.lower() != 'n':
					check_ok = input('\n>>> Correct parameters? [y/n]\n<<< ')

				if check_ok.lower() == 'y':

					if main_v['architecture'] == 'FCNN':
						model = FCNN(main_v['input_net'], main_v['num_classes'], main_v['learning_rate'])

					elif main_v['architecture'] == 'FAB_CONVNET':
						model = FAB_CONVNET(main_v['input_net'], main_v['num_classes'], main_v['learning_rate'])

					workflow(main_v['mode'], model)

				else:
					pass

			elif main_v['mode'] == 'test':

				main_v['output_model'] = None

				check_ok = '_'

				print('\n>>> Parameters: \n')

				for item in main_v:
					if item.startswith('train') or item.startswith('test'):
						pass
					else:
						print(bcolors.OKGREEN + '{} --> {}\n'.format(item, main_v[item]) + bcolors.ENDC)

				while check_ok.lower() != 'y' and check_ok.lower() != 'n':
					check_ok = input('\n>>> Correct parameters? [y/n]\n<<< ')

				if check_ok.lower() == 'y' and main_v['dataset_name'] is not None:
					workflow(main_v['mode'])

				else:
					print('\n>>> Function not enabled.')
					pass

			else:
				print('\n>>> SET MODE FIRST.')

		elif chaser == 'EXIT':
			print('>>> Exiting the program\n')
			print(quit())

		else:
			print('>>> Invalid input')

		



