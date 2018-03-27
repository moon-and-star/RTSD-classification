#!/usr/bin/env python

import keras
from keras.layers import Dense, Activation, Flatten, Lambda, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from keras.engine import Input, Model
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard,ReduceLROnPlateau

from util.util import  num_of_classes, dataset_size, checkpoint_dir, log_path, log_directory, experiment_directory
from .image_datagen import image_generator


def input_layer(config):
    size = config['img']['img_size'] + 2 * config['img']['padding']
    img_rows, img_cols = size, size

    img_channels = 3
    inputs = Input(shape=(img_rows, img_cols, img_channels))
    print(inputs.shape)

    return inputs

from keras import backend as K
# from keras.backend import relu, tanh
def add_activation(input, activation):
    x = None

    if activation == 'stanh':
        x = Lambda(lambda x: x * 2./3.)(input)
        # x = Activation('tanh')(x)
        x = Activation(K.tanh)(x)
        # x = K.tanh(x)
        x = Lambda(lambda x: x * 1.7159)(x)
    elif activation == 'relu':
        x = Activation(lambda x: K.relu(x, alpha=0.0))(input)
        # x = K.relu(input, alpha=1)
        print(x)
        
    return x


# "valid" padding means "no padding"
def conv_act_pool(inputs, kernel_size=(3,3), filters=1, activation='stanh'):
    init = keras.initializers.RandomUniform(minval=-0.05, maxval=0.05)
    x = Conv2D(filters, kernel_size, padding="valid", data_format="channels_last",
               kernel_initializer=init)(inputs)
    x = add_activation(x, activation)
    x = MaxPooling2D(pool_size=(2, 2),
                     data_format="channels_last", padding='valid')(x)
    return x


def fc_block(input, config):
    init = keras.initializers.RandomUniform(minval=-0.05, maxval=0.05)
    x = Dense(300, kernel_initializer=init)(input)
    x = add_activation(x, config['mcdnn']["activation"])
    x = Dense(num_of_classes(config), activation='softmax',
              kernel_initializer=init)(x)
    return x


def conv_block(inputs, config):
    act = config['mcdnn']["activation"]
    x = conv_act_pool(inputs, kernel_size=(7, 7), filters=100, activation=act)
    x = conv_act_pool(x, kernel_size=(4, 4), filters=150, activation=act)
    x = conv_act_pool(x, kernel_size=(4, 4), filters=250, activation=act)

    return x

def get_MCDNN(config):
    inputs = input_layer(config)
    x = conv_block(inputs, config)
    x = Flatten()(x)
    outputs = fc_block(x, config)

    model = Model(inputs=inputs, outputs=outputs)
    return model


# TODO check optimizer parameters
def prepare_optimizer(config):
    lr = config['solver_exact']['base_lr']
    opt = SGD(lr=lr, decay=5e-4, momentum=0.9, nesterov=True)
    return opt


def prepare_model(config):
    opt = prepare_optimizer(config)
    model = get_MCDNN(config)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model





def prepare_scheduler(config):
    def lr_sch(epoch):
        lr = config['solver_exact']['base_lr']
        gamma = config['solver_exact']['gamma']
        step = config['train_params']['lr_step']

        return lr * gamma ** (epoch // step)

    return LearningRateScheduler(lr_sch)


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


def train(config):
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
                                  validation_steps=val_size,  # batch_size for validation == 1
                                  callbacks=callbacks)
    model.save('{}/latest_version.h5'.format(checkpoint_dir(config)))
    save_history(history, config)
