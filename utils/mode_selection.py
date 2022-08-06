import os
from .colors import bcolors
from .main_var import main_v

def select_mode(current_path):

	global main_v

	header = """\n
                        $$\      $$\  $$$$$$\  $$$$$$$\  $$$$$$$$\                     
                        $$$\    $$$ |$$  __$$\ $$  __$$\ $$  _____|                    
                        $$$$\  $$$$ |$$ /  $$ |$$ |  $$ |$$ |                          
                        $$\$$\$$ $$ |$$ |  $$ |$$ |  $$ |$$$$$\                        
                        $$ \$$$  $$ |$$ |  $$ |$$ |  $$ |$$  __|                       
                        $$ |\$  /$$ |$$ |  $$ |$$ |  $$ |$$ |                          
                        $$ | \_/ $$ | $$$$$$  |$$$$$$$  |$$$$$$$$\                     
                        \__|     \__| \______/ \_______/ \________|                    
 $$$$$$\  $$$$$$$$\ $$\       $$$$$$$$\  $$$$$$\ $$$$$$$$\ $$$$$$\  $$$$$$\  $$\   $$\ 
$$  __$$\ $$  _____|$$ |      $$  _____|$$  __$$\\__$$  __|\_$$  _|$$  __$$\ $$$\  $$ |
$$ /  \__|$$ |      $$ |      $$ |      $$ /  \__|  $$ |     $$ |  $$ /  $$ |$$$$\ $$ |
\$$$$$$\  $$$$$\    $$ |      $$$$$\    $$ |        $$ |     $$ |  $$ |  $$ |$$ $$\$$ |
 \____$$\ $$  __|   $$ |      $$  __|   $$ |        $$ |     $$ |  $$ |  $$ |$$ \$$$$ |
$$\   $$ |$$ |      $$ |      $$ |      $$ |  $$\   $$ |     $$ |  $$ |  $$ |$$ |\$$$ |
\$$$$$$  |$$$$$$$$\ $$$$$$$$\ $$$$$$$$\ \$$$$$$  |  $$ |   $$$$$$\  $$$$$$  |$$ | \$$ |
 \______/ \________|\________|\________| \______/   \__|   \______| \______/ \__|  \__|
                                                                                       
                                                                                       
                                                                                       
\n\n"""

	os.system('clear')
	print(bcolors.OKCYAN + header + bcolors.ENDC)

	mode = input('>>> Insert mode between [TRAIN-VAL], [train-test], [test].\n<<< ')

	if mode == '' or mode.lower() == 'train-val':
		main_v['mode'] = 'train-val'

		try:
			main_v['val_split'] = float(input('\n>>> Enter the validation split. Press ENTER to set default value 0.2.\n<<< '))
		except:
			main_v['val_split'] = 0.2

		if main_v['val_split'] <= 0 or main_v['val_split'] >= 1:
			print('\n>>> Invalid input. Set default value.')

			main_v['val_split'] = 0.2


	elif mode.lower() == 'train-test':
		main_v['mode'] = 'train-test'


	elif mode.lower() == 'test':
		main_v['mode'] = 'test'

		if len(os.listdir(current_path)) == 0:
			print('\n>>> No model available.')

		else:
			print('\n>>> Select one of the available models:\n')

			for item in os.listdir(current_path):
				print(bcolors.OKGREEN + '\t' + item + bcolors.ENDC)

			input_model = input('\n<<< ')

			try:
				index = input_model.index('_i')
				input_size = input_model[index + 2 :]

				image_size = int(input_size.split('x')[0])
				channels = int(input_size.split('x')[1])

				if channels == 1:
					main_v['color_mode'] = 'grayscale'
				elif channels == 3:
					main_v['color_mode'] = 'rgb'
				elif channels == 4:
					main_v['color_mode'] = 'rgba'

				main_v['batch_size'] = 32
				main_v['input_net'] = (image_size, image_size, channels)
				main_v['image_size'] = (image_size, image_size)

				input_model = os.path.join(current_path, input_model)
				main_v['input_model'] = input_model

			except:
				print('\n>>> Wrong model name.')

	else:
		print('\n>>> Invalid input. Set default value.')
		main_v['mode'] = 'train-val'

		try:
			main_v['val_split'] = float(input('\n>>> Enter the validation split. Press ENTER to set default value 0.2.\n<<< '))
		except:
			main_v['val_split'] = 0.2

		if main_v['val_split'] <= 0 or main_v['val_split'] >= 1:
			print('\n>>> Invalid input. Set default value.')
			main_v['val_split'] = 0.2

	os.system('clear')