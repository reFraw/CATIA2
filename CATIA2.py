import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow
import argparse

from utils.main_var import main_v
from utils.colors import bcolors
from utils.colors import header
from utils.mode_selection import select_mode
from utils.dataset_generation import select_dataset
from utils.arc_selection import select_arc
from utils.start_flow import workflow

from models_code.FCNN.FCNN import FCNN
from models_code.FAB_CONVNET.FAB_CONVNET import FAB_CONVNET

def parser():

	pars = argparse.ArgumentParser(prog='CATIA2', description='CNNs Aggregator Tool for Image Analysis 2')
	group = pars.add_argument_group('Arguments')

	group.add_argument('-u', '--usage', type=str, required=False, choices=['y'], help='Enable the one-line arguments mode.')
	group.add_argument('-a', '--architecture', type=str, required=False, choices=['FCNN', 'FAB_CONVNET'],
		help='One of the available architectures.')
	group.add_argument('-d', '--dataset', type=str, required=False, help='Name of the dataset.')
	group.add_argument('-o', '--output', type=str, required=False, help='Name of the output model.')
	group.add_argument('-l', '--loading', type=str, required=False, help='Name of the input model.')
	group.add_argument('-e', '--epochs', type=int, required=False, help='Number of epochs for the training phase. Defaul value 10.', default=10)
	group.add_argument('-i', '--image_size', type=str, required=False, help='Image size in the format SIZExCHANNELS. Default value 100x1.', default='100x1')
	group.add_argument('-b', '--batch_size', type=int, required=False, default=32, help='Default value 32.')
	group.add_argument('-r', '--l_rate', type=float, required=False, help='Learning rate value for optimizer. Default value 0.01.', default=0.01)
	group.add_argument('-v', '--validation_split', type=float, required=False, default=0.2, help='Default value 0.2.')
	group.add_argument('-m', '--mode', type=str, required=False, choices=['train-val', 'test', 'train-test'],
		help='One of the avaliable mode | train-val | train-test | test | -- Default value train-val.', default='train-val')

	args = pars.parse_args()

	return args

def check_args(args):

	global main_v

	main_v['mode'] = args.mode
	main_v['architecture'] = args.architecture
	main_v['learning_rate'] = args.l_rate
	main_v['batch_size'] = args.batch_size
	main_v['epochs'] = args.epochs
	main_v['train_path'] = os.path.join(dataset_path, args.dataset, 'training')
	main_v['test_path'] = os.path.join(dataset_path, args.dataset, 'test')
	main_v['num_classes'] = len(os.listdir(main_v['test_path']))
	main_v['val_split'] = args.validation_split

	try:
		main_v['output_model'] = os.path.join(model_saved_path, args.output)
	except:
		main_v['output_model'] = None

	try:	
		main_v['input_model'] = os.path.join(model_saved_path, args.loading)
	except:
		main_v['input_model'] = None

	imgSize = int(args.image_size.split('x')[0])
	channels = int(args.image_size.split('x')[1])

	if channels == 1:
		main_v['color_mode'] = 'grayscale'
	elif channels == 3:
		main_v['color_mode'] = 'rgb'
	elif channels == 4:
		main_v['color_mode'] = 'rgba'

	main_v['input_net'] = (imgSize, imgSize, channels)
	main_v['image_size'] = main_v['input_net'][0:2]


if __name__ == '__main__':

	current_path = os.getcwd()

	dataset_path = os.path.join(current_path, 'DATASETS')
	model_saved_path = os.path.join(current_path, 'models_saved')
	models_code_path = os.path.join(current_path, 'models_code')

	# ========== HEADER ========== #

	print('\n' + bcolors.HEADER +  header + bcolors.ENDC)
	print(bcolors.HEADER + '\t\tCNNs Aggregator Tool for Image Analysis 2\n\n' + bcolors.ENDC)

	args = parser()

	if args.usage == 'y':
		
		check_args(args)

		if main_v['architecture'] == 'FCNN':
			model = FCNN(main_v['input_net'], main_v['num_classes'], main_v['learning_rate'])

		elif main_v['architecture'] == 'FAB_CONVNET':
			model = FAB_CONVNET(main_v['input_net'], main_v['num_classes'], main_v['learning_rate'])

		if main_v['mode'] == 'test':
			workflow(main_v['mode'])
		else:
			workflow(main_v['mode'], model)

	else:
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

		



