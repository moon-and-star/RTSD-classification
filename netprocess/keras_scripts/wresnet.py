#!/usr/bin/env python
# coding: utf-8

# In[1]:

# https://github.com/transcranial/wide-resnet/blob/master/wide-resnet.ipynb


# In[2]:

if __name__ == '__main__':
    from util.config import get_config
    config = get_config('./config.json')

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(config['train_params']['gpu_num'])
    os.environ["KERAS_BACKEND"] = 'tensorflow'


import keras

import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from keras.layers import add,merge, Dense, Activation, Flatten, Lambda, Conv2D, AveragePooling2D, BatchNormalization, Dropout
from keras.engine import Input, Model
from keras.optimizers import SGD
from keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard,ReduceLROnPlateau
from keras.utils import np_utils
import keras.backend as K

import json
import time

from util.util import  num_of_classes, dataset_size, checkpoint_dir, log_path, log_directory, experiment_directory
from .image_datagen import image_generator


def zero_pad_channels(x, pad=0):
    """
    Function for Lambda layer
    """
    pattern = [[0, 0], [0, 0], [0, 0], [pad - pad // 2, pad // 2]]
    return tf.pad(x, pattern)


def get_subsample(x, nb_filters=16, subsample_factor=1):
    if subsample_factor > 1:
        subsample = (subsample_factor, subsample_factor)
    else:
        subsample = (1, 1)


def subsample_and_shortcut(x, nb_filters=16, subsample_factor=1):
    prev_nb_channels = K.int_shape(x)[3]

    if subsample_factor > 1:
        subsample = (subsample_factor, subsample_factor)
        # shortcut: subsample + zero-pad channel dim
        shortcut = AveragePooling2D(pool_size=subsample, data_format="channels_last")(x)
    else:
        subsample = (1, 1)
        # shortcut: identity
        shortcut = x
        
    if nb_filters > prev_nb_channels:
        shortcut = Lambda(zero_pad_channels,
                          arguments={'pad': nb_filters - prev_nb_channels})(shortcut)

    return subsample, shortcut


def residual_block(x, nb_filters=16, subsample_factor=1, enable_dropout=False):
    
    subsample, shortcut = subsample_and_shortcut(x, nb_filters, subsample_factor)
    
    y = BatchNormalization(axis=3)(x)
    y = Activation('relu')(y)
    y = Conv2D(nb_filters, (3, 3), 
               kernel_initializer="he_normal", padding="same", data_format="channels_last", strides=subsample)(y)
    
    y = BatchNormalization(axis=3)(y)
    y = Activation('relu')(y)
    if enable_dropout:
        y = Dropout(0.5)(y)
    y = Conv2D(nb_filters, (3, 3), 
        kernel_initializer="he_normal", padding="same", data_format="channels_last", strides=(1,1))(y)
    
    out = add([y, shortcut])
    
    return out


def input_layer(config):
    
    size = config['img']['img_size'] + 2 * config['img']['padding']
    img_rows, img_cols = size, size
    
    img_channels = 3
    inputs = Input(shape=(img_rows, img_cols, img_channels))
    print(inputs.shape)

    return inputs


def conv1(inputs, config):
    x = Conv2D(16, (3, 3), 
               padding="same", data_format="channels_last", kernel_initializer="he_normal")(inputs)
    return x


def conv2(x, config):
    blocks_per_group = config['wide_resnet']['metablock_depth']
    widening_factor = config['wide_resnet']['width']
    dropout = config['wide_resnet']['dropout']
    
    for i in range(0, blocks_per_group):
        nb_filters = 16 * widening_factor
        x = residual_block(x, nb_filters=nb_filters,
                           subsample_factor=1,enable_dropout=dropout)
        
    return x


def conv3(x, config):
    blocks_per_group = config['wide_resnet']['metablock_depth']
    widening_factor = config['wide_resnet']['width']
    
    for i in range(0, blocks_per_group):
        nb_filters = 32 * widening_factor
        if i == 0:
            subsample_factor = 2
        else:
            subsample_factor = 1
        x = residual_block(x, nb_filters=nb_filters, subsample_factor=subsample_factor)
    
    return x


def conv4(x, config):
    blocks_per_group = config['wide_resnet']['metablock_depth']
    widening_factor = config['wide_resnet']['width']
    
    for i in range(0, blocks_per_group):
        nb_filters = 64 * widening_factor
        if i == 0:
            subsample_factor = 2
        else:
            subsample_factor = 1
        x = residual_block(x, nb_filters=nb_filters, subsample_factor=subsample_factor)

    return x


def wresnet(config):
    
    inputs = input_layer(config)
    x = conv1(inputs, config)
    x = conv2(x, config)
    x = conv3(x, config)
    x = conv4(x, config)
   
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=(8, 8), strides=None, padding='valid', data_format="channels_last")(x)
    x = Flatten()(x)

    predictions = Dense(num_of_classes(config), activation='softmax')(x)

    model = Model(inputs=inputs, outputs=predictions)
    
    return model







def prepare_optimizer(config):
    lr = config['solver_exact']['base_lr']
    opt = SGD(lr=lr, decay=5e-4, momentum=0.9, nesterov=True)
    
    return opt


def prepare_scheduler(config):
    
    def lr_sch(epoch):
        lr = config['solver_exact']['base_lr']
        gamma = config['solver_exact']['gamma']
        step = config['train_params']['lr_step']
        
        return lr * gamma**(epoch//step)
    
    return LearningRateScheduler(lr_sch) 


def prepare_model(config):
    
    opt = prepare_optimizer(config)
    model = wresnet(config)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def prepare_checkpointer(config):
    period = config['train_params']['snap_epoch']
    name = 'weights.{epoch:02d}.hdf5'
    filepath = "{}/{}".format(checkpoint_dir(config), name)

    checkpointer = ModelCheckpoint(filepath, 
                                   monitor='val_acc', 
                                   verbose=1, 
                                   save_best_only=True, 
                                   save_weights_only=True, 
                                   mode='auto', period=period)
    return checkpointer


def get_callbacks(config):
    lr_scheduler = prepare_scheduler(config)
    # reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5, verbose=1, epsilon=0.005, min_lr=0.0001)
    checkpointer = prepare_checkpointer(config)
    logger = CSVLogger(log_path(config), separator=',', append=False)
    stopping = EarlyStopping(monitor='val_acc', patience=10, mode='auto')
    visualizer = TensorBoard(log_dir=log_directory(config, 'tensorboard_logs'), write_graph=True, histogram_freq=1)

    return [lr_scheduler, checkpointer, logger, stopping, visualizer]


def save_history(history, config):
    import pickle
    directory = experiment_directory(config)
    path = '{directory}/history'.format(**locals())

    with open(path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


def train_wresnet(config):
    val_ratio = config['img']['val_ratio']
    batch_size = config['train_params']['batch_size']
    nb_epoch = config['train_params']['epoch_num']

    train_gen = image_generator(config, 'train')
    train_size = dataset_size(config, 'train')

    if val_ratio > 0:
        val_gen = image_generator(config, 'val')
        val_size = dataset_size(config, 'val')
    else:
        val_gen = image_generator(config, 'train', test_on_train=True)
        val_size = dataset_size(config, 'train')
    
    
    model = prepare_model(config)
    model.summary()
     
    callbacks = get_callbacks(config)
    print("got callbacks")
    
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=train_size / batch_size,
                                  epochs=nb_epoch, verbose=1,
                                  validation_data=val_gen,
                                  validation_steps=val_size, #batch_size for validation == 1
                                  callbacks=callbacks)
    model.save('{}/latest_version.h5'.format(checkpoint_dir(config)))
    save_history(history, config)
    

if __name__ == '__main__':
    train(config)






