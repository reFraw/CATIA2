import os

from .main_var import main_v
from .main_var import path

from matplotlib import style

import matplotlib.pyplot as plt
import numpy as np

def save_report(result, timedate, result2=None):

	print('\n>>> Saving report.')
	
	global main_v
	global path

	if main_v['mode'] == 'test':
		report_name = main_v['input_model_name'] + timedate + '.txt'
		report_file = os.path.join(path['report_path'], report_name)
		
		with open(report_file, 'w') as r:
			r.write('RESULT\n\n')
			r.write('mode : {}\n'.format(main_v['mode']))
			r.write('Input model : {}\n'.format(main_v['input_model_name']))
			r.write('Dataset : {}\n'.format(main_v['dataset_name']))
			r.write('Number of classes : {}\n\n'.format(main_v['num_classes']))

			r.write('# ----- TEST METRICS ----- #\n\n')
			r.write('Test Loss : {}\n'.format(result[0]))
			r.write('Test Accuracy : {}\n'.format(result[1]))
			r.write('Test Precision : {}\n'.format(result[2]))
			r.write('Test Recall : {}\n'.format(result[3]))
			r.write('Test AUC : {}\n'.format(result[4]))

	elif main_v['mode'] == 'train-val':

		if main_v['output_model_name'] is None:
			report_name = timedate[1:] + '.txt'
		else:
			report_name = main_v['output_model_name'] + timedate +'.txt'

		report_file = os.path.join(path['report_path'], report_name)

		train_acc = result.history['acc']
		train_prec = result.history['prec']
		train_rec = result.history['rec']
		train_auc = result.history['auc']

		val_acc = result.history['val_acc']
		val_prec = result.history['val_prec']
		val_rec = result.history['val_rec']
		val_auc = result.history['val_auc']

		with open(report_file, 'w') as r:
			r.write('RESULT\n\n')
			r.write('Mode : {}\n'.format(main_v['mode']))
			r.write('Output model : {}\n'.format(main_v['output_model_name']))
			r.write('Dataset : {}\n'.format(main_v['dataset_name']))
			r.write('Number of classes : {}\n'.format(main_v['num_classes']))
			r.write('Validation split : {}\n'.format(main_v['val_split']))
			r.write('Image size : {}x{}\n'.format(main_v['image_size'][0], main_v['channels']))
			r.write('Batch size : {}\n'.format(main_v['batch_size']))
			r.write('Architecture : {}\n'.format(main_v['architecture']))
			r.write('Learning_rate : {}\n'.format(main_v['learning_rate']))
			r.write('Epochs : {}\n\n'.format(main_v['epochs']))

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

	elif main_v['mode'] == 'train-test':

		if main_v['output_model_name'] is None:
			report_name = timedate[1:] + '.txt'
		else:
			report_name = main_v['output_model_name'] + timedate +'.txt'

		report_file = os.path.join(path['report_path'], report_name)

		train_acc = result.history['acc']
		train_prec = result.history['prec']
		train_rec = result.history['rec']
		train_auc = result.history['auc']

		with open(report_file, 'w') as r:

			r.write('RESULT\n\n')
			r.write('Mode : {}\n'.format(main_v['mode']))
			r.write('Output model : {}\n'.format(main_v['output_model_name']))
			r.write('Dataset : {}\n'.format(main_v['dataset_name']))
			r.write('Number of classes : {}\n'.format(main_v['num_classes']))
			r.write('Image size : {}x{}\n'.format(main_v['image_size'][0], main_v['channels']))
			r.write('Batch size : {}\n'.format(main_v['batch_size']))
			r.write('Architecture : {}\n'.format(main_v['architecture']))
			r.write('Learning_rate : {}\n'.format(main_v['learning_rate']))
			r.write('Epochs : {}\n\n'.format(main_v['epochs']))

			r.write('# ----- TRAIN METRICS ----- #\n\n')
			r.write('Train Accuracy : {}\n'.format(train_acc))
			r.write('Train Precision : {}\n'.format(train_prec))
			r.write('Train Recall : {}\n'.format(train_rec))
			r.write('Train AUC : {}\n\n'.format(train_auc))

			r.write('# ----- TEST METRICS ----- #\n\n')
			r.write('Test Loss : {}\n'.format(result2[0]))
			r.write('Test Accuracy : {}\n'.format(result2[1]))
			r.write('Test Precision : {}\n'.format(result2[2]))
			r.write('Test Recall : {}\n'.format(result2[3]))
			r.write('Test AUC : {}\n'.format(result2[4]))

	print('>>> Report saved.')

def plot_metrics(result, timedate):
	global main_v
	global path

	print('\n>>> Saving metrics plot.')


	try:
		axis_font = {'fontname':'DejaVu Sans', 'size':'25'}
		title_font = {'fontname':'DejaVu Sans', 'size':'30'}

		if main_v['output_model_name'] is None:
			main_v['output_model_name'] =  main_v['architecture'] + '_NOTSAVED' + timedate

		image_path = os.path.join(path['plot_path'], main_v['output_model_name'])

		epochs = main_v['epochs'] + 1
		EPOCHS = []

		for n in range(1, epochs):
			EPOCHS.append(n)

		if not os.path.exists(image_path):
			os.makedirs(image_path)

		if main_v['mode'] == 'train-test':

			train_acc = result.history['acc']
			train_prec = result.history['prec']
			train_rec = result.history['rec']
			train_auc = result.history['auc']
			train_loss = result.history['loss']

			fig = plt.figure(figsize=(27,14))
			style.use('ggplot')

			plt.plot(EPOCHS, train_acc, marker='o', label='Training Accuracy', linewidth=2, color='r')
			plt.plot(EPOCHS, train_prec, marker='^', label='Training Precision', linewidth=2, color='b')
			plt.plot(EPOCHS, train_rec, marker='v', label='Training Recall', linewidth=2, color='g')
			plt.plot(EPOCHS, train_auc, marker='d', label='Training AUC', linewidth=2, color='m')
			plt.plot(EPOCHS, train_loss, marker='s', label='Training Loss', linewidth=2, color='c')

			plt.xlabel('Epochs', **axis_font)
			plt.xticks(np.arange(1, main_v['epochs'] + 1), fontsize=17)
			plt.xlim([0, main_v['epochs'] + 1 ])
			plt.ylabel('Metrics', **axis_font)
			plt.yticks(np.arange(0,1.01, step=0.05), fontsize=17)
			plt.ylim([-0.05,1.05])
			plt.title('Metrics plot', **title_font)
			plt.grid(color='white')
			plt.legend(fontsize=20)

			plt.savefig(image_path+'/metrics_plot' + timedate + '.png', dpi=fig.dpi)

		elif main_v['mode'] == 'train-val':

			train_acc = result.history['acc']
			train_prec = result.history['prec']
			train_rec = result.history['rec']
			train_auc = result.history['auc']
			train_loss = result.history['loss']

			val_acc = result.history['val_acc']
			val_prec = result.history['val_prec']
			val_rec = result.history['val_rec']
			val_auc = result.history ['val_auc']
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
			plt.xticks(np.arange(1, main_v['epochs'] + 1), fontsize=17)
			plt.xlim([0, main_v['epochs'] + 1 ])
			plt.ylabel('Metrics', **axis_font)
			plt.yticks(np.arange(0,1.01, step=0.05), fontsize=17)
			plt.ylim([-0.05,1.05])
			plt.title('Metrics plot', **title_font)
			plt.grid(color='white')
			plt.legend(fontsize=20)

			plt.savefig(image_path+'/metrics_plot' + timedate + '.png', dpi=fig.dpi)

		print('>>> Metrics plot saved.')

	except Exception as e:
		print('>>> Error occurred.')