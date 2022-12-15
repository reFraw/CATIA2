import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras import layers
from keras import models
from keras.metrics import Precision, Recall, AUC
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

def build_FCNN2(input_net, num_classes, learning_rate):

	# INPUT LAYER
    input_layer = layers.Input(shape=input_net, name='input_layer')

    # BLOCK 01
    conv1_01 = layers.Conv2D(32, (3, 3), padding='same', activation='elu', name='conv1_block01') (input_layer)
    conv2_01 = layers.Conv2D(32, (5, 5), padding='same', activation='elu', name='conv2_block01') (input_layer)
    conv3_01 = layers.Conv2D(32, (7, 7), padding='same', activation='elu', name='conv3_block01') (input_layer)
    unit_01 = layers.Concatenate(name='concatenate_1') ([conv1_01, conv2_01, conv3_01])
    conv_final_01 = layers.Conv2D(16, (1, 1), padding='same', activation='relu', name='conv1d_1') (unit_01)
    final_01 = layers.MaxPooling2D((2, 2), name='maxpooling_1') (conv_final_01)

    # BLOCK 02
    conv1_02 = layers.Conv2D(64, (3, 3), padding='same', activation='elu', name='conv1_block02') (final_01)
    conv2_02 = layers.Conv2D(64, (5, 5), padding='same', activation='elu', name='conv2_block02') (final_01)
    conv3_02 = layers.Conv2D(64, (7, 7), padding='same', activation='elu', name='conv3_block02') (final_01)
    unit_02 = layers.Concatenate(name='concatenate_2') ([conv1_02, conv2_02, conv3_02])
    conv_final_02 = layers.Conv2D(32, (1, 1), padding='same', activation='relu', name='conv1d_2') (unit_02)
    final_02 = layers.MaxPooling2D((2, 2), name='maxpooling_2') (conv_final_02)

    # CLASSIFICATION BLOCK
    vectorized = layers.Flatten(name='flatten') (final_02)
    dropout_1 = layers.Dropout(0.5, name='dropout_1') (vectorized)
    dense_2 = layers.Dense(64, activation='relu', name='dense_1') (dropout_1)
    dropout_2 = layers.Dropout(0.5, name='dropout_2') (dense_2)

    # OUTPUT LAYER
    output_layer = layers.Dense(num_classes, activation='softmax', name='softmax') (dropout_2)
    
    model = models.Model(inputs=input_layer, outputs=output_layer, name='FCNNplus')

    model.compile(
        loss=CategoricalCrossentropy(),
        optimizer=Adam(learning_rate=learning_rate),
        metrics=['acc', Precision(name='prec'), Recall(name='rec'), AUC(name='auc')])

    return model