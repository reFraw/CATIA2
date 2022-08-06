from .main_var import main_v 
from .colors import bcolors

import os

def select_dataset(dataset_path, mode):	

	global main_v

	header = """\n      $$$$$$$\   $$$$$$\ $$$$$$$$\  $$$$$$\   $$$$$$\  $$$$$$$$\ $$$$$$$$\             
      $$  __$$\ $$  __$$\\__$$  __|$$  __$$\ $$  __$$\ $$  _____|\__$$  __|            
      $$ |  $$ |$$ /  $$ |  $$ |   $$ /  $$ |$$ /  \__|$$ |         $$ |               
      $$ |  $$ |$$$$$$$$ |  $$ |   $$$$$$$$ |\$$$$$$\  $$$$$\       $$ |               
      $$ |  $$ |$$  __$$ |  $$ |   $$  __$$ | \____$$\ $$  __|      $$ |               
      $$ |  $$ |$$ |  $$ |  $$ |   $$ |  $$ |$$\   $$ |$$ |         $$ |               
      $$$$$$$  |$$ |  $$ |  $$ |   $$ |  $$ |\$$$$$$  |$$$$$$$$\    $$ |               
      \_______/ \__|  \__|  \__|   \__|  \__| \______/ \________|   \__|               
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

	print("\n>>> If you haven't chosen the mode yet, press ENTER and come back later.")
	print('\n>>> Select one of the availabe dataset:\n')

	for item in os.listdir(dataset_path):
		print(bcolors.OKGREEN + '\t' + item + bcolors.ENDC)

	data_name = input('\n<<< ')

	if data_name == '':
		pass
		os.system('clear')

	else:

		main_v['dataset_name'] = data_name

		data_path = os.path.join(dataset_path, data_name)
		train_path = os.path.join(data_path, 'training')
		test_path = os.path.join(data_path, 'test')

		main_v['num_classes'] = len(os.listdir(train_path))
		main_v['train_path'] = train_path
		main_v['test_path'] = test_path

		if mode == 'train-test' or mode == 'train-val':

			input_size = input('\n>>> Insert input size in the format SIZExCHANNELS.\n<<< ')

			try:
				image_size = int(input_size.split('x')[0])
				image_channels = int(input_size.split('x')[1])

				if image_channels == 1:
					main_v['color_mode'] = 'grayscale'
				elif image_channels == 3:
					main_v['color_mode'] = 'rgb'
				elif image_channels == 4:
					main_v['color_mode'] = 'rgba'

				main_v['input_net'] = (image_size, image_size, image_channels)
				main_v['image_size'] = (image_size, image_size)

			except:
				print('\n>>> Invalid input format. Set default value 100x1.')
				main_v['input_net'] = (100, 100, 1)
				main_v['image_size'] = (100, 100)
				main_v['color_mode'] = 'grayscale'

			main_v['batch_size'] = int(input('\n>>> Insert the batch size\n<<< '))

		os.system('clear')