import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style

from .variables import path
from .variables import main_variables as main_v

def save_report(results=None, time=None, test_results=None, n=None):

	if main_v['mode'] == 'train-val' or main_v['mode'] == 'train-test':

		save_path = path['report']
		
		if main_v['output_model_name'] is None:
			report_name = 'NOTSAVED_m'+main_v['architecture']+'_i'+str(main_v['image_size'][0])+'x'+str(main_v['channels'])+time+ '.txt'
		else:
			report_name = main_v['output_model_name'] + '_m' + main_v['architecture']+'_i'+str(main_v['image_size'][0])+'x'+str(main_v['channels'])+time + '.txt'

		save_path = os.path.join(save_path, report_name)

		print('\n[*] Saving report at {}'.format(save_path))

		history = results.history
		metrics = list(history.keys())

		with open(save_path, 'w') as r:

			r.write('CATIA 2 REPORT\n\n')
			r.write('\n # ------ INFO ------ #\n')
			r.write('Mode --> {}\n'.format(main_v['mode']))
			r.write('Dataset --> {}\n'.format(main_v['dataset']))
			r.write('Number of Classes --> {}\n'.format(main_v['num_classes']))
			r.write('Validation split --> {}\n'.format(main_v['val_split']))
			r.write('Output model name --> {}\n\n'.format(main_v['output_model_name']))
			r.write('\n # ------ HYPERPARAMETERS ------ #\n')
			r.write('Epochs --> {}\n'.format(main_v['epochs']))
			r.write('Image size --> Height : {} - Width : {}\n'.format(main_v['image_size'][0], main_v['image_size'][1]))
			r.write('Color mode (Channels) --> {} ({})\n'.format(main_v['color_mode'], main_v['channels']))
			r.write('Batch size --> {}\n'.format(main_v['batch_size']))
			r.write('Architecture --> {}\n'.format(main_v['architecture']))
			r.write('Learning rate --> {}\n\n'.format(main_v['learning_rate']))
			r.write('\n # ------ TRAINING RESULTS ------ #\n')
			for metric in metrics:
				r.write('{} --> {}\n'.format(metric, history[metric]))

			if test_results != None:
				r.write('\n\n # ------ TEST RESULTS ------ #\n')
				for index, metric in enumerate(metrics):
					r.write('{} --> {}\n'.format(metric, test_results[index]))

		print('[*] Report saved at {}'.format(save_path))

	elif main_v['mode'] == 'test':

		save_path = path['report']
		save_path = os.path.join(save_path, main_v['input_model_name'] + time + '.txt')

		print('\n[*] Saving report at {}'.format(save_path))

		with open(save_path, 'w') as r:

			r.write('CATIA 2 REPORT\n\n')
			r.write('\n # ------ INFO ------ #\n')
			r.write('Mode --> {}\n'.format(main_v['mode']))
			r.write('Dataset --> {}\n'.format(main_v['dataset']))
			r.write('Number of Classes --> {}\n'.format(main_v['num_classes']))
			r.write('Model loaded : {}\n'.format(main_v['input_model_name']))
			r.write('\n # ------ HYPERPARAMETERS ------ #\n')
			r.write('Image size --> Height : {} - Width : {}\n'.format(main_v['image_size'][0], main_v['image_size'][1]))
			r.write('Color mode (Channels) --> {} ({})\n'.format(main_v['color_mode'], main_v['input_model_name'][-1]))
			r.write('\n\n # ------ TEST RESULTS ------ #\n')
			for metric in test_results:
				r.write('\n {} --> {}\n'.format(metric, test_results[metric]))

		print('[*] Report saved at {}'.format(save_path))

	else:

		return True


def save_graph(results, time):

	save_path = path['plot']

	if main_v['output_model_name'] is None:
		folder_name = 'NOTSAVED_m'+main_v['architecture']+'_i'+str(main_v['image_size'][0])+'x'+str(main_v['channels'])+time
	else:
		folder_name = main_v['output_model_name'] + '_m' + main_v['architecture']+'_i'+str(main_v['image_size'][0])+'x'+str(main_v['channels'])+time

	save_path = os.path.join(save_path, folder_name) +'.png'

	print('[*] Plotting metrics at {}'.format(save_path))

	history = results.history
	metrics = list(history.keys())
	epochs = range(1, len(history[metrics[0]]) + 1)

	fig = plt.figure(figsize=(25,13))
	style.use('ggplot')
	for metric in metrics:
		if main_v['mode'] == 'train-val':
			if metric.startswith('val'):
				plt.plot(epochs, history[metric], marker='o', label=metric)
			else:
				plt.plot(epochs, history[metric], linestyle='dotted', label=metric)
		else:
			plt.plot(epochs, history[metric], marker='o', label=metric)

	plt.xlabel('Epochs')
	plt.xlim([0, epochs[-1] + 1])
	plt.xticks(np.arange(1, epochs[-1] + 1))
	plt.ylabel('Metrics')
	plt.ylim([-0.05,1.05])
	plt.yticks(np.arange(0,1.01, step=0.05))
	plt.title('Metrics-Epochs Plot')

	plt.legend()

	plt.savefig(save_path, dpi=fig.dpi)

	print('[*] Metrics plotted at {}'.format(save_path))

	return True