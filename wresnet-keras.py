
# coding: utf-8

# In[1]:

# https://github.com/transcranial/wide-resnet/blob/master/wide-resnet.ipynb


# In[2]:

# model.summary()
from util.config import get_config
config = get_config('./config.json')

# from pprint import pprint
# pprint(config)


# In[3]:

get_ipython().magic("env CUDA_VISIBLE_DEVICES=config['train_params']['gpu_num']")
get_ipython().magic('env KERAS_BACKEND=tensorflow')
import keras


# In[4]:

import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from keras.layers import add,merge, Dense, Activation, Flatten, Lambda, Conv2D, AveragePooling2D, BatchNormalization, Dropout
from keras.engine import Input, Model
from keras.optimizers import SGD
from keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import keras.backend as K
import json
import time

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


# In[5]:

def zero_pad_channels(x, pad=0):
    """
    Function for Lambda layer
    """
    pattern = [[0, 0], [0, 0], [0, 0], [pad - pad // 2, pad // 2]]
    return tf.pad(x, pattern)


# In[6]:

def get_subsample(x, nb_filters=16, subsample_factor=1):
    if subsample_factor > 1:
        subsample = (subsample_factor, subsample_factor)
    else:
        subsample = (1, 1)


# In[7]:

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


# In[8]:

def residual_block(x, nb_filters=16, subsample_factor=1):
    
    subsample, shortcut = subsample_and_shortcut(x, nb_filters, subsample_factor)
    
    y = BatchNormalization(axis=3)(x)
    y = Activation('relu')(y)
    y = Conv2D(nb_filters, (3, 3), 
               kernel_initializer="he_normal", padding="same", data_format="channels_last", strides=subsample)(y)
    
    y = BatchNormalization(axis=3)(y)
    y = Activation('relu')(y)
    y = Dropout(0.5)(y)
    y = Conv2D(nb_filters, (3, 3), 
        kernel_initializer="he_normal", padding="same", data_format="channels_last", strides=(1,1))(y)
    
    out = add([y, shortcut])
    
    return out


# In[9]:

def input_layer(config):
    
    size = config['img']['img_size'] + 2 * config['img']['padding']
    img_rows, img_cols = size, size
    
    img_channels = 3
    inputs = Input(shape=(img_rows, img_cols, img_channels))
    print(inputs.shape)

    return inputs


# In[10]:

def conv1(inputs, config):
    x = Conv2D(16, (3, 3), 
               padding="same", data_format="channels_last", kernel_initializer="he_normal")(inputs)
    return x


# In[11]:

def conv2(x, config):
    blocks_per_group = config['wide_resnet']['metablock_depth']
    widening_factor = config['wide_resnet']['width']
    
    for i in range(0, blocks_per_group):
        nb_filters = 16 * widening_factor
        x = residual_block(x, nb_filters=nb_filters, subsample_factor=1)
        
    return x


# In[12]:

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


# In[13]:

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


# In[14]:

from util.util import  num_of_classes
def wresnet(config):
    get_ipython().magic('%time')
    
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


# In[15]:

# if False:
#     batch_size = 64
#     nb_epoch = 200
#     data_augmentation = False

#     # Learning rate schedule
#     def lr_sch(epoch):
#         if epoch < 60:
#             return 0.1
#         elif epoch < 120:
#             return 0.02
#         elif epoch < 160:
#             return 0.004
#         else:
#             return 0.0008

#     # Learning rate scheduler callback
#     lr_scheduler = LearningRateScheduler(lr_sch)

#     # Model saving callback
#     #checkpointer = ModelCheckpoint(filepath='stochastic_depth_cifar10.hdf5', verbose=1, save_best_only=True)

#     if not data_augmentation:
#         print('Not using data augmentation.')
#         history = model.fit(x_train, y_train, 
#                             batch_size=batch_size, nb_epoch=nb_epoch, verbose=1,
#                             validation_data=(x_test, y_test), shuffle=True,
#                             callbacks=[lr_scheduler])
#     else:
#         print('Using real-time data augmentation.')

#         # realtime data augmentation
#         datagen_train = ImageDataGenerator(
#             featurewise_center=False,
#             samplewise_center=False,
#             featurewise_std_normalization=False,
#             samplewise_std_normalization=False,
#             zca_whitening=False,
#             rotation_range=0,
#             width_shift_range=0.125,
#             height_shift_range=0.125,
#             horizontal_flip=False,
#             vertical_flip=False)
#         datagen_train.fit(x_train)

#         # fit the model on the batches generated by datagen.flow()
#         history = model.fit_generator(datagen_train.flow(x_train, y_train, batch_size=batch_size, shuffle=True),
#                                       samples_per_epoch=x_train.shape[0], 
#                                       nb_epoch=nb_epoch, verbose=1,
#                                       validation_data=(x_test, y_test),
#                                       callbacks=[lr_scheduler])


# In[16]:

def init_generator(config, phase):
    if phase == 'train':
        return ImageDataGenerator(
                    featurewise_center=False,
                    samplewise_center=False,
                    featurewise_std_normalization=False,
                    samplewise_std_normalization=False,
                    zca_whitening=False,
                    rotation_range=0,
    #                 width_shift_range=0.125,
    #                 height_shift_range=0.125,
                    horizontal_flip=False,
                    vertical_flip=False)
    else:
        return ImageDataGenerator(
                    featurewise_center=False,
                    samplewise_center=False,
                    featurewise_std_normalization=False,
                    samplewise_std_normalization=False,
                    zca_whitening=False,
                    rotation_range=0,
    #                 width_shift_range=0.125,
    #                 height_shift_range=0.125,
                    horizontal_flip=False,
                    vertical_flip=False)


# In[17]:

def image_generator(config, phase):
     # datagen_train.fit(x_train)
    datagen = init_generator(config, phase)
    root = config['img']['processed_path']
    directory = '{root}/{phase}'.format(**locals())
    size = config['img']['img_size'] + 2 * config['img']['padding']
    
    if phase == 'train':
        shuffle = True
    else: 
        shuffle = False
    
    generator = datagen.flow_from_directory(
            directory,  
            target_size=(size, size),
            batch_size=config['train_params']['batch_size'],
            class_mode='categorical',
#             seed=42,
            save_to_dir=None, 
            shuffle=shuffle)
    
    return generator


# In[ ]:




# In[18]:

def prepare_optimizer(config):
    lr = config['solver_exact']['base_lr']
    opt = SGD(lr=lr, decay=5e-4, momentum=0.9, nesterov=True)
    
    return opt


# In[19]:

def prepare_scheduler(config):
    
    def lr_sch(epoch):
        lr = config['solver_exact']['base_lr']
        gamma = lr = config['solver_exact']['gamma']
        step = config['train_params']['lr_step']
        
        return lr * gamma**(epoch//step)
    
    return LearningRateScheduler(lr_sch) 


# In[20]:

def prepare_model(config):
    
    opt = prepare_optimizer(config)
    model = wresnet(config)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# In[ ]:

from util.util import dataset_size
def train(config):
    batch_size = config['train_params']['batch_size']
    nb_epoch = config['train_params']['epoch_num']


    train_gen = image_generator(config, 'train')
    train_size = dataset_size(config, 'train')
    
    val_gen = image_generator(config, 'val')
    val_size = dataset_size(config, 'val')
    
    
    model = prepare_model(config)
#     model.summary()
    
    lr_scheduler = prepare_scheduler(config)
    history = model.fit_generator(train_gen,
#                                   samples_per_epoch=train_size, 
                                  steps_per_epoch=train_size / batch_size,
                                  epochs=nb_epoch, verbose=1,
                                  validation_data=val_gen,
                                  validation_steps=val_size / batch_size,
                                  callbacks=[lr_scheduler])

# for i in range(1):
#     x, y = train_generator.__next__()
#     imgplot = plt.imshow(x[0, :, :, :].astype(np.uint8))


# In[ ]:

train(config)


# In[ ]:



