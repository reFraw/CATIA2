main_variables = {
	'mode' : None,
	'dataset' : None,
	'num_classes' : None,
	'val_split' : None,
	'learning_rate' : None,
	'image_size' : None,
	'channels' : None,
	'color_mode' : None,
	'input_net' : None,
	'output_model_path' : None,
	'output_model_name' : None,
	'input_model_path' : None,
	'input_model_name' : None,
	'architecture' : None,
	'epochs' : None,
	'batch_size' : None
}

path = {
	'dataset' : None,
	'models_code' : None,
	'models_saved' : None,
	'results' : None,
	'report' : None,
	'plot' : None,
	'gradcam' : None
}

multi_variables = {
	'val_split' : [],
	'learning_rate' : [],
	'image_size' : [],
	'channels' : [],
	'color_mode' : [],
	'input_net' : [],
	'output_model_path' : [],
	'output_model_name' : [],
	'input_model_path' : [],
	'input_model_name' : [],
	'epochs' : []
}

available_arc = [
	'FCNN',
	'FAB_CONVNET',
	'RAVNET',
	'SCNN',
	'ALEXNET',
	'LE_NET',
	'FCNN-2',
	'VGG16',
	'VGG19',
	'ResNet50',
	'DenseNet121',
	'EfficientNetB0',
	'InceptionV3',
	'Xception'
	]