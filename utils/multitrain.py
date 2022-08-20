import tensorflow as tf
import os
import platform
import matplotlib.pyplot as plt
import numpy as np

from .colors import bcolors
from .main_var import multi_v
from .main_var import main_v
from .main_var import path

from datetime import datetime
from tensorflow.keras.utils import image_dataset_from_directory
from matplotlib import style

def check_system():
	sys = platform.system()
	return sys

def metrics(result, n, dt_string):

	global main_v
	global multi_v
	global path

	axis_font = {'fontname':'DejaVu Sans', 'size':'25'}
	title_font = {'fontname':'DejaVu Sans', 'size':'30'}

	image_path = os.path.join(path['plot_path'], multi_v['output_model_name'][n])

	if not os.path.exists(image_path):
		os.makedirs(image_path)

	epochs = multi_v['epochs'][n] + 1
	EPOCHS = []

	for i in range(1, epochs):
		EPOCHS.append(i)

	train_acc = result.history['acc']
	train_prec = result.history['prec']
	train_rec = result.history['rec']
	train_auc = result.history['auc']
	train_loss = result.history['loss']

	val_acc = result.history['val_acc']
	val_prec = result.history['val_prec']
	val_rec = result.history['val_rec']
	val_auc = result.history['val_auc']
	val_loss = result.history['val_loss']

	fig = plt.figure(figsize=(27,14))
	style.use('ggplot')

	plt.plot(EPOCHS, train_acc, label='Training Accuracy', linestyle='dotted', linewidth=2, color='r')
	plt.plot(EPOCHS, train_prec, label='Training Precision', linestyle='dotted', linewidth=2, color='b')
	plt.plot(EPOCHS, train_rec, label='Training Recall', linestyle='dotted', linewidth=2, color='g')
	plt.plot(EPOCHS, train_auc, label='Training AUC', linestyle='dotted', linewidth=2, color='m')
	plt.plot(EPOCHS, train_loss, marker='s', label='Training Loss', linestyle='dotted', linewidth=2, color='c')

	plt.plot(EPOCHS, val_acc, marker='o', label='Validation Accuracy', linewidth=2, color='r')
	plt.plot(EPOCHS, val_prec, marker='^', label='Validation Precision', linewidth=2, color='b')
	plt.plot(EPOCHS, val_rec, marker='v', label='Validation Recall', linewidth=2, color='g')
	plt.plot(EPOCHS, val_auc, marker='d', label='Validation AUC', linewidth=2, color='m')
	plt.plot(EPOCHS, val_loss, marker='s', label='Validation Loss', linewidth=2, color='c')


	plt.xlabel('Epochs', **axis_font)
	plt.xticks(np.arange(1, multi_v['epochs'][n] + 1), fontsize=17)
	plt.xlim([0, multi_v['epochs'][n] + 1 ])
	plt.ylabel('Metrics', **axis_font)
	plt.yticks(np.arange(0,1.01, step=0.05), fontsize=17)
	plt.ylim([-0.05,1.05])
	plt.title('Metrics plot', **title_font)
	plt.grid(color='white')
	plt.legend(fontsize=20)

	plt.savefig(image_path+'/metrics_plot' + dt_string + '.png', dpi=fig.dpi)


def report(result, n, dt_string):

	global multi_v
	global main_v
	global path

	train_acc = result.history['acc']
	train_prec = result.history['prec']
	train_rec = result.history['rec']
	train_auc = result.history['auc']

	val_acc = result.history['val_acc']
	val_prec = result.history['val_prec']
	val_rec = result.history['val_rec']
	val_auc = result.history['val_auc']

	report_name = multi_v['output_model_name'][n] + dt_string + '.txt'
	report_file = os.path.join(path['report_path'], report_name)

	with open(report_file, 'w') as r:
		r.write('RESULT\n\n')
		r.write('Mode : {}\n'.format(main_v['mode']))
		r.write('Output model : {}\n'.format(multi_v['output_model_name'][n]))
		r.write('Dataset : {}\n'.format(main_v['dataset_name']))
		r.write('Number of classes : {}\n'.format(main_v['num_classes']))
		r.write('Validation split : {}\n'.format(multi_v['val_split'][n]))
		r.write('Image size : {}x{}\n'.format(multi_v['image_size'][n], multi_v['channels'][n]))
		r.write('Batch size : {}\n'.format(multi_v['batch_size'][n]))
		r.write('Architecture : {}\n'.format(multi_v['architecture']))
		r.write('Learning_rate : {}\n'.format(multi_v['l_rate'][n]))
		r.write('Epochs : {}\n\n'.format(multi_v['epochs'][n]))

		r.write('# ----- TRAIN METRICS ----- #\n\n')
		r.write('Train Accuracy : {}\n'.format(train_acc))
		r.write('Train Precision : {}\n'.format(train_prec))
		r.write('Train Recall : {}\n'.format(train_rec))
		r.write('Train AUC : {}\n\n'.format(train_auc))

		r.write('# ----- VALIDATION METRICS ----- #\n\n')
		r.write('Validation Accuracy : {}\n'.format(val_acc))
		r.write('Validation Precision : {}\n'.format(val_prec))
		r.write('Validation Recall : {}\n'.format(val_rec))
		r.write('Validation AUC : {}\n\n'.format(val_auc))


def multi_train():

	sys = check_system()

	if sys == 'Windows':
		clear = 'cls'
	elif sys == 'Linux':
		clear = 'clear'

	global multi_v
	global main_v
	global path

	for item in multi_v:
		multi_v[item] = []

	header = """\n _______ _     _ _    _______ _ _______ ______  _______ _ _______ 
(_______|_)   (_|_)  (_______) (_______|_____ \(_______) (_______)
 _  _  _ _     _ _       _   | |   _    _____) )_______| |_     _ 
| ||_|| | |   | | |     | |  | |  | |  |  __  /|  ___  | | |   | |
| |   | | |___| | |_____| |  | |  | |  | |  \ \| |   | | | |   | |
|_|   |_|\_____/|_______)_|  |_|  |_|  |_|   |_|_|   |_|_|_|   |_|
                                                                  
                 _______ _______ ______  _______                  
                (_______|_______|______)(_______)                 
                 _  _  _ _     _ _     _ _____                    
                | ||_|| | |   | | |   | |  ___)                   
                | |   | | |___| | |__/ /| |_____                  
                |_|   |_|\_____/|_____/ |_______)                 
                                                                  \n\n"""

	os.system(clear)

	print(bcolors.OKCYAN + header + bcolors.ENDC)

	try:
		iter_train = int(input('\n>>> Insert number of trainings\n<<< '))
		multi_v['iter_train'] = iter_train
	except:
		print('>>> Invalid insert. Set 10 by deafult.')
		iter_train = 10
		multi_v['iter_train'] = iter_train

	chaser = '_'

	# ===================================================================================================================

	print(bcolors.OKGREEN + '\n------- EPOCHS -------' + bcolors.ENDC)
	while chaser.lower() != 'y' and chaser.lower() != 'n':
		chaser = input('Set epochs with increasing step? [y/n]\n<<< ')
		if chaser.lower() != 'y' and chaser.lower() != 'n':
			print('>>> Invalid input')

	if chaser.lower() == 'y':
		try:
			print('>>> Set epochs step')
			epochs_step = int(input('<<< '))
		except:
			print('>>> Invalid input. Set default value 2.')
			epochs_step = 2

		try:
			print('>>> Set starting epoch')
			epochs_start = int(input('<<< '))
		except:
			print('>>> Invalid input. Set default value 10.')
			epochs_start = 10

		for n in range(0, epochs_step*iter_train, epochs_step):
			multi_v['epochs'].append(epochs_start + n)

	else:
		for n in range(iter_train):
			epochs_i = int(input('>>> Insert epochs for experiment {}/{}\n<<< '.format(n+1, iter_train)))
			multi_v['epochs'].append(epochs_i)

	# ==================================================================================================================

	print(bcolors.OKGREEN + '\n------- IMAGE SIZE -------' + bcolors.ENDC)
	choice1 = input('Use same image size for all experiments? [y/n]\n<<< ')

	if choice1.lower() == 'y':
		image_size_i = int(input('>>> Insert image size in format <<WIDTH>>\n<<< '))
		for n in range(iter_train):
			multi_v['image_size'].append(image_size_i)

	elif choice1.lower() == 'n':
		if choice1.lower() != 'n':
			print('>>> Invalid input. Set epochs for each experiment.')
		for n in range(iter_train):
			image_size_i = int(input('>>> Insert image size in format <<WIDTH>> for experiment {}/{}\n<<< '.format(n+1, iter_train)))
			multi_v['image_size'].append(image_size_i)

	# ===================================================================================================================

	print(bcolors.OKGREEN + '\n------- CHANNELS -------' + bcolors.ENDC)
	choice2 = input('Use same channels for all experiments? [y/n]\n<<< ')

	if choice2.lower() == 'y':
		image_channels = int(input('>>> Insert image channels\n<<< '))
		if image_channels != 1 and image_channels != 4 and image_channels != 3:
			print('\n>>> Invalid input. Set default value 1.')
			image_channels = 1
		for n in range(iter_train):
			multi_v['channels'].append(image_channels)

	else:
		for n in range(iter_train):
			image_channels = int(input('>>> Insert image channels for experiment {}/{}\n<<< '.format(n+1, iter_train)))
			if image_channels != 1 and image_channels != 4 and image_channels != 3:
				print('\n>>> Invalid input. Set default value 1.')
				image_channels = 1
			multi_v['channels'].append(image_channels)

	# ===================================================================================================================

	print(bcolors.OKGREEN + '\n------- BATCH SIZE -------' + bcolors.ENDC)
	choice3 = input('Use same batch size for all experiments? [y/n]\n<<< ')

	if choice3.lower() == 'y':
		batch_i = int(input('>>> Insert batch size\n<<< '))
		for n in range(iter_train):
			multi_v['batch_size'].append(batch_i)

	else:
		for n in range(iter_train):
			batch_i = int(input('>>> Insert batch size for experiment {}/{}\n<<< '.format(n+1, iter_train)))
			multi_v['batch_size'].append(batch_i)

	# ===================================================================================================================

	print(bcolors.OKGREEN + '\n------- VALIDATION SPLIT -------' + bcolors.ENDC)
	choice4 = input('Use same validation split for all experiments? [y/n]\n<<< ')

	if choice4.lower() == 'y':
		val_i = float(input('>>> Insert validation split\n<<< '))
		for n in range(iter_train):
			multi_v['val_split'].append(val_i)

	else:
		for n in range(iter_train):
			val_i = float(input('>>> Insert validation split for experiment {}/{}\n<<< '.format(n+1, iter_train)))
			multi_v['val_split'].append(val_i)		

	# ===================================================================================================================

	print(bcolors.OKGREEN + '\n------- ARCHITECTURE -------' + bcolors.ENDC)
	print('>>> Select one of the available architecture: \n')

	for item in os.listdir(path['models_code_path']):
		if item.startswith('__'):
			pass
		else:
			print(bcolors.OKCYAN + '\t' + item + bcolors.ENDC)

	multi_v['architecture'] = input('\n<<< ')

	# ===================================================================================================================

	print(bcolors.OKGREEN + '\n------- LEARNING RATE -------' + bcolors.ENDC)
	choice5 = input('Use same learning rate for all experiments? [y/n]\n<<< ')

	if choice5.lower() == 'y':
		lr_i = float(input('>>> Insert learning rate\n<<< '))
		for n in range(iter_train):
			multi_v['l_rate'].append(lr_i)

	else:
		for n in range(iter_train):
			lr_i = float(input('>>> Insert learning rate for experiment {}/{}\n<<< '.format(n+1, iter_train)))
			multi_v['l_rate'].append(lr_i)

	# ===================================================================================================================

	print(bcolors.OKGREEN + '\n------- OUTPUT MODEL NAME -------' + bcolors.ENDC)
	print('>>> Insert output models name')
	name = input('<<< ')

	for n in range(iter_train):
		name_i = name + '_' + str(n+1) + '_m' + multi_v['architecture'] +'_i' + str(multi_v['image_size'][n]) + 'x' + str(multi_v['channels'][n])
		multi_v['output_model_name'].append(name_i)

	# ===================================================================================================================

	for n in range(iter_train):
		multi_v['input_net'].append((multi_v['image_size'][n], multi_v['image_size'][n], multi_v['channels'][n]))
		multi_v['dataset_size'].append((multi_v['image_size'][n], multi_v['image_size'][n]))
		if multi_v['channels'][n] == 1:
			multi_v['color_mode'].append('grayscale')
		elif multi_v['channels'][n] == 3:
			multi_v['color_mode'].append('rgb')
		elif multi_v['channels'][n] == 4:
			multi_v['color_mode'].append('rgba')

	waiter = input('\nPress ENTER to continue')	

	os.system(clear)


def start_multitrain(model, n):

	global main_v
	global multi_v
	global path

	now = datetime.now()
	dt_string = now.strftime('_%d%m%Y-%H%M%S')

	print(bcolors.OKCYAN + '\nSTARTING EXPERIMENT {}'.format(n+1) + bcolors.ENDC)

	print('\n ------- TRAIN DATA -------')
	train_data = image_dataset_from_directory(
			main_v['train_path'],
			labels='inferred',
			label_mode='categorical',
			color_mode=multi_v['color_mode'][n],
			batch_size=multi_v['batch_size'][n],
			image_size=multi_v['dataset_size'][n],
			validation_split=multi_v['val_split'][n],
			subset='training',
			seed=0,
			shuffle=True)

	print('\n ------- VALIDATION DATA -------')
	valid_data = image_dataset_from_directory(
			main_v['train_path'],
			labels='inferred',
			label_mode='categorical',
			color_mode=multi_v['color_mode'][n],
			batch_size=multi_v['batch_size'][n],
			image_size=multi_v['dataset_size'][n],
			validation_split=multi_v['val_split'][n],
			subset='validation',
			seed=0,
			shuffle=True)

	print('\n')

	history = model.fit(
		train_data,
		epochs=multi_v['epochs'][n],
		validation_data=valid_data)

	model.save(os.path.join(path['model_saved_path'], multi_v['output_model_name'][n]))
	print('\n>>> Model saved successfully')
	report(history, n, dt_string)
	print('>>> Report saved successfully')
	metrics(history, n, dt_string)
	print('>>> Metrics plot saved successfully\n\n')


