#!/usr/bin/env python3

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow
import argparse
import platform

from utils.main_var import main_v
from utils.main_var import multi_v
from utils.main_var import path
from utils.colors import bcolors
from utils.colors import header
from utils.mode_selection import select_mode
from utils.dataset_generation import select_dataset
from utils.arc_selection import select_arc
from utils.start_flow import workflow
from utils.multitrain import multi_train
from utils.multitrain import start_multitrain

from models_code.FCNN.FCNN import FCNN
from models_code.FAB_CONVNET.FAB_CONVNET import FAB_CONVNET
from models_code.RAVNET.RAVNET import RAVNET

def parser():

	pars = argparse.ArgumentParser(prog='CATIA2', description='CNNs Aggregator Tool for Image Analysis 2')
	subparser = pars.add_subparsers(dest='command')

	# ======= One line mode ======= #
	usage = subparser.add_parser('one-line', help='Enable the one-line mode.')

	usage.add_argument('-u', '--usage', type=str, required=False, choices=['y'], help='Enable the one-line arguments mode.')
	usage.add_argument('-a', '--architecture', type=str, required=False, choices=['FCNN', 'FAB_CONVNET'],
		help='One of the available architectures.')
	usage.add_argument('-d', '--dataset', type=str, required=False, help='Name of the dataset.')
	usage.add_argument('-o', '--output', type=str, required=False, help='Name of the output model.')
	usage.add_argument('-l', '--loading', type=str, required=False, help='Name of the input model.')
	usage.add_argument('-e', '--epochs', type=int, required=False, help='Number of epochs for the training phase. Defaul value 10.', default=10)
	usage.add_argument('-i', '--image_size', type=str, required=False, help='Image size in the format SIZExCHANNELS. Default value 100x1.', default='100x1')
	usage.add_argument('-b', '--batch_size', type=int, required=False, default=32, help='Default value 32.')
	usage.add_argument('-r', '--l_rate', type=float, required=False, help='Learning rate value for optimizer. Default value 0.01.', default=0.01)
	usage.add_argument('-v', '--validation_split', type=float, required=False, default=0.2, help='Default value 0.2.')
	usage.add_argument('-m', '--mode', type=str, required=False, choices=['train-val', 'test', 'train-test'],
		help='One of the avaliable mode | train-val | train-test | test | -- Default value train-val.', default='train-val')

	args = pars.parse_args()

	return args

def create_dirs():
	global path

	if not os.path.exists(path['report_path']):
		os.makedirs(path['report_path'])

	if not os.path.exists(path['plot_path']):
		os.makedirs(path['plot_path'])

	if not os.path.exists(path['model_saved_path']):
		os.makedirs(path['model_saved_path'])

	if not os.path.exists(path['dataset_path']):
		os.makedirs(path['dataset_path'])


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

	if main_v['mode'] != 'test':
		imgSize = int(args.image_size.split('x')[0])
		main_v['channels'] = int(args.image_size.split('x')[1])

		if main_v['channels'] == 1:
			main_v['color_mode'] = 'grayscale'
		elif main_v['channels'] == 3:
			main_v['color_mode'] = 'rgb'
		elif main_v['channels'] == 4:
			main_v['color_mode'] = 'rgba'

		main_v['input_net'] = (imgSize, imgSize, main_v['channels'])
		main_v['image_size'] = main_v['input_net'][0:2]

	else:
		index = main_v['input_model'].index('_i')
		image = main_v['input_model'][index+2:]
		imgSize = int(image.split('x')[0])
		channels = int(image.split('x')[1])

		if channels == 1:
			main_v['color_mode'] = 'grayscale'
		elif channels == 3:
			main_v['color_mode'] = 'rgb'
		elif channels == 4:
			main_v['color_mode'] = 'rgba'

		main_v['input_net'] = (imgSize, imgSize, channels)
		main_v['image_size'] = main_v['input_net'][0:2]


def show_main_menu():

	name = '''\n\n\n ＣＮＮｓ ＡＧＧＲＥＧＡＴＯＲ ＴＯＯＬ ＦＯＲ ＩＭＡＧＥ ＡＮＡＬＹＳＩＳ ２'''
	repo = '\t\t  ＧｉｔＨｕｂ.ｃｏｍ/ｒｅＦｒａｗ/ＣＡＴＩＡ２'

	print(bcolors.HEADER + name + bcolors.ENDC)
	print(bcolors.HEADER +  header + bcolors.ENDC)
	print('\n' + bcolors.HEADER + repo + bcolors.ENDC)

	print(bcolors.OKCYAN + '\n\n\t   MENU\n' + bcolors.ENDC)
	print(bcolors.OKCYAN + '{1}--- Mode selection' + bcolors.ENDC)
	print(bcolors.OKCYAN + '{2}--- Dataset selection' + bcolors.ENDC)
	print(bcolors.OKCYAN + '{3}--- Architecture selection' + bcolors.ENDC)
	print(bcolors.OKCYAN + '{4}--- Start CNN\n' + bcolors.ENDC)

	print(bcolors.OKCYAN + '{77}-- Set Multi-Train mode' + bcolors.ENDC)
	print(bcolors.OKCYAN + '{99}-- Start Multi-Train mode\n' + bcolors.ENDC)

	print(bcolors.OKCYAN + '{P}--- Check parameters\n' + bcolors.ENDC)

	print(bcolors.OKCYAN + '{00}-- Exit program\n\n' + bcolors.ENDC)


def show_parameters():

	global main_v

	if main_v['mode'] == 'test':
		print(bcolors.OKGREEN + '\n------- PARAMETERS ------- \n' + bcolors.ENDC)
		print(bcolors.OKGREEN + 'Name : {} | Value : {}'.format('Mode', main_v['mode']) + bcolors.ENDC)
		print(bcolors.OKGREEN + 'Name : {} | Value : {}'.format('Dataset', main_v['dataset_name']) + bcolors.ENDC)
		print(bcolors.OKGREEN + 'Name : {} | Value : {}'.format('Input model', main_v['input_model_name']) + bcolors.ENDC)
		print(bcolors.OKGREEN + '\n-------------------------- \n' + bcolors.ENDC)

	elif main_v['mode'] == 'train-test':
		print(bcolors.OKGREEN + '\n------- PARAMETERS ------- \n' + bcolors.ENDC)
		print(bcolors.OKGREEN + 'Name : {} | Value : {}'.format('Mode', main_v['mode']) + bcolors.ENDC)
		print(bcolors.OKGREEN + 'Name : {} | Value : {}'.format('Dataset', main_v['dataset_name']) + bcolors.ENDC)
		print(bcolors.OKGREEN + 'Name : {} | Value : {}'.format('Image size', main_v['image_size']) + bcolors.ENDC)
		print(bcolors.OKGREEN + 'Name : {} | Value : {}'.format('channels', main_v['channels']) + bcolors.ENDC)
		print(bcolors.OKGREEN + 'Name : {} | Value : {}'.format('Color mode', main_v['color_mode']) + bcolors.ENDC)
		print(bcolors.OKGREEN + 'Name : {} | Value : {}'.format('Architecture', main_v['architecture']) + bcolors.ENDC)
		print(bcolors.OKGREEN + 'Name : {} | Value : {}'.format('Learning rate', main_v['learning_rate']) + bcolors.ENDC)
		print(bcolors.OKGREEN + 'Name : {} | Value : {}'.format('Batch size', main_v['batch_size']) + bcolors.ENDC)
		print(bcolors.OKGREEN + 'Name : {} | Value : {}'.format('Epochs', main_v['epochs']) + bcolors.ENDC)
		print(bcolors.OKGREEN + '\n-------------------------- \n' + bcolors.ENDC)

	elif main_v['mode'] == 'train-val':
		print(bcolors.OKGREEN + '\n------- PARAMETERS ------- \n' + bcolors.ENDC)
		print(bcolors.OKGREEN + 'Name : {} | Value : {}'.format('Mode', main_v['mode']) + bcolors.ENDC)
		print(bcolors.OKGREEN + 'Name : {} | Value : {}'.format('Validation split', main_v['val_split']) + bcolors.ENDC)
		print(bcolors.OKGREEN + 'Name : {} | Value : {}'.format('Dataset', main_v['dataset_name']) + bcolors.ENDC)
		print(bcolors.OKGREEN + 'Name : {} | Value : {}'.format('Image size', main_v['image_size']) + bcolors.ENDC)
		print(bcolors.OKGREEN + 'Name : {} | Value : {}'.format('channels', main_v['channels']) + bcolors.ENDC)
		print(bcolors.OKGREEN + 'Name : {} | Value : {}'.format('Color mode', main_v['color_mode']) + bcolors.ENDC)
		print(bcolors.OKGREEN + 'Name : {} | Value : {}'.format('Architecture', main_v['architecture']) + bcolors.ENDC)
		print(bcolors.OKGREEN + 'Name : {} | Value : {}'.format('Learning rate', main_v['learning_rate']) + bcolors.ENDC)
		print(bcolors.OKGREEN + 'Name : {} | Value : {}'.format('Batch size', main_v['batch_size']) + bcolors.ENDC)
		print(bcolors.OKGREEN + 'Name : {} | Value : {}'.format('Epochs', main_v['epochs']) + bcolors.ENDC)
		print(bcolors.OKGREEN + '\n-------------------------- \n' + bcolors.ENDC)

	elif main_v['mode'] == 'multitrain':
		print(bcolors.OKGREEN + '\n------- PARAMETERS ------- \n' + bcolors.ENDC)
		print(bcolors.OKGREEN + 'Name : {} | Value : {}'.format('Mode', main_v['mode']) + bcolors.ENDC)
		print(bcolors.OKGREEN + 'Name : {} | Value : {}'.format('Validation split', multi_v['val_split']) + bcolors.ENDC)
		print(bcolors.OKGREEN + 'Name : {} | Value : {}'.format('Dataset', main_v['dataset_name']) + bcolors.ENDC)
		print(bcolors.OKGREEN + 'Name : {} | Value : {}'.format('No of classes', main_v['num_classes']) + bcolors.ENDC)
		print(bcolors.OKGREEN + 'Name : {} | Value : {}'.format('Image size', multi_v['image_size']) + bcolors.ENDC)
		print(bcolors.OKGREEN + 'Name : {} | Value : {}'.format('channels', multi_v['channels']) + bcolors.ENDC)
		print(bcolors.OKGREEN + 'Name : {} | Value : {}'.format('Color mode', multi_v['color_mode']) + bcolors.ENDC)
		print(bcolors.OKGREEN + 'Name : {} | Value : {}'.format('Architecture', multi_v['architecture']) + bcolors.ENDC)
		print(bcolors.OKGREEN + 'Name : {} | Value : {}'.format('Learning rate', multi_v['l_rate']) + bcolors.ENDC)
		print(bcolors.OKGREEN + 'Name : {} | Value : {}'.format('Batch size', multi_v['batch_size']) + bcolors.ENDC)
		print(bcolors.OKGREEN + 'Name : {} | Value : {}'.format('Epochs', multi_v['epochs']) + bcolors.ENDC)
		print(bcolors.OKGREEN + '\n-------------------------- \n' + bcolors.ENDC)

	else:
		print(bcolors.OKGREEN + '\n------- PARAMETERS ------- \n' + bcolors.ENDC)

		for item in main_v:
			if item.startswith('input') or item.startswith('output') or item.startswith('train') or item.startswith('test'):
				pass
			else:
				print(bcolors.OKGREEN + 'Name : {} | Value : {}'.format(item, main_v[item]) + bcolors.ENDC)

		print(bcolors.OKGREEN + '\n-------------------------- \n' + bcolors.ENDC)


def check_system():

	sys = platform.system()

	return sys

# ========== MAIN ========== #

if __name__ == '__main__':

	sys = check_system()

	if sys == 'Windows':
		clear = 'cls'
	elif sys == 'Linux':
		clear = 'clear'

	ROOT = os.path.abspath(__file__)
	FILENAME = os.path.basename(__file__)
	ROOT = ROOT.replace(FILENAME, '')

	model_saved_path = os.path.join(ROOT, 'models_saved')
	models_code_path = os.path.join(ROOT, 'models_code')
	dataset_path = os.path.join(ROOT, 'DATASETS')

	path['report_path'] = os.path.join(ROOT, 'results', 'report')
	path['plot_path'] = os.path.join(ROOT, 'results', 'plot')
	path['model_saved_path'] = model_saved_path
	path['models_code_path'] = models_code_path
	path['dataset_path'] = dataset_path
	
	create_dirs()

	args = parser()

	### ============================= ###
	### ======= ONE-LINE MODE ======= ###
	### ============================= ###

	if args.command == 'one-line':
		check_args(args)

		# ======= ARCHITECTURE OL MODE ======= #

		if main_v['architecture'] == 'FCNN':
			model = FCNN(main_v['input_net'], main_v['num_classes'], main_v['learning_rate'])
		elif main_v['architecture'] == 'FAB_CONVNET':
			model = FAB_CONVNET(main_v['input_net'], main_v['num_classes'], main_v['learning_rate'])
		elif main_v['architecture'] == 'RAVNET':
			model = FAB_CONVNET(main_v['input_net'], main_v['num_classes'], main_v['learning_rate'])

		if main_v['mode'] == 'test':
			workflow(main_v['mode'])
		else:
			workflow(main_v['mode'], model)

	### =========================== ###
	### ======= WIZARD MODE ======= ###
	### =========================== ###

	else:
		os.system(clear)

		chaser = '_'
		show_main_menu()

		while True:
			chaser = input(bcolors.WARNING + '\nCATIA2~# ' + bcolors.ENDC)

			if chaser == '1':
				select_mode(model_saved_path)
				show_main_menu()

			elif chaser == '2':
				select_dataset(dataset_path, main_v['mode'])
				show_main_menu()

			elif chaser == '3':
				check = select_arc(models_code_path, main_v['mode'])
				if check:
					show_main_menu()

			elif chaser.lower() == 'p':
				show_parameters()

			elif chaser == '4':
				try:
					if main_v['mode'] == 'train-val' or main_v['mode'] == 'train-test':
						main_v['input_model'] = None
						save_check = '_'

						while save_check.lower() != 'y' and save_check.lower() != 'n':
							save_check = input('\n>>> Save the model? [y/n]\n<<< ')

						if save_check == 'y':
							output_model = input('\n>>> Insert model name\n<<< ')
							main_v['output_model_name'] = output_model + '_m' + main_v['architecture'] + '_i' + str(main_v['input_net'][0]) + 'x' + str(main_v['input_net'][2])
							output_model = os.path.join(model_saved_path, main_v['output_model_name'])
							main_v['output_model'] = output_model

						# ======= ARCHITECTURE WIZ MODE ======= #

						if main_v['architecture'] == 'FCNN':
							model = FCNN(main_v['input_net'], main_v['num_classes'], main_v['learning_rate'])
						elif main_v['architecture'] == 'FAB_CONVNET':
							model = FAB_CONVNET(main_v['input_net'], main_v['num_classes'], main_v['learning_rate'])
						elif main_v['architecture'] == 'RAVNET':
							model = RAVNET(main_v['input_net'], main_v['num_classes'], main_v['learning_rate'])

						try:
							workflow(main_v['mode'], model)
							for item in main_v:
								main_v[item] = None
							show_main_menu()
						except Exception as e:
							print('\n>>> Error occured. Please check parameters.')

					elif main_v['mode'] == 'test':
						main_v['output_model'] = None
						main_v['output_model_name'] = None

						if main_v['dataset_name'] is not None:
							try:
								workflow(main_v['mode'])
								for item in main_v:
									main_v[item] = None
								show_main_menu()
							except Exception as e:
								print('\n>>> Error occured. Please check parameters.')
						else:
							print('\n>>> Function not enabled.')
							pass

					elif main_v['mode'] == 'multitrain':
						print('\n>>> Use ' + bcolors.OKCYAN + 'START MULTITRAIN MODE ' + bcolors.ENDC + 'instead')
					else:
						print('\n>>> Set mode first.')
				except Exception as e:
					print('\n>>> Error occured.\n\n')

			elif chaser == '77':
				main_v['mode'] = 'multitrain'
				multi_train()
				show_main_menu()

			elif chaser == '99':

				if main_v['dataset_name'] != None and main_v['mode'] == 'multitrain':

					for it in range(multi_v['iter_train']):
						if multi_v['architecture'] == 'FCNN':
							model = FCNN(multi_v['input_net'][it], main_v['num_classes'], multi_v['l_rate'][it])
						elif multi_v['architecture'] == 'RAVNET':
							model = RAVNET(multi_v['input_net'][it], main_v['num_classes'], multi_v['l_rate'][it])
						elif multi_v['architecture'] == 'FAB_CONVNET':
							model = FAB_CONVNET(multi_v['input_net'][it], main_v['num_classes'], multi_v['l_rate'][it])

						start_multitrain(model, it)

					waiter = input('\nPress ENTER to continue')
					os.system(clear)
					show_main_menu()

				else:
					print('\n>>> Set parameters first')

			elif chaser.lower() == '00':
				waiter = input('\n>>> Press any key to close the program...')
				os.system(clear)
				print(quit())

			else:
				print('>>> Invalid input')

		



