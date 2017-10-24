#!/usr/bin/env python
# https://github.com/tdeboissiere/DeepLearningImplementations/blob/master/DenseNet/densenet.py

from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import Input, merge
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint
from keras.callbacks import EarlyStopping, CSVLogger, TensorBoard,ReduceLROnPlateau
from keras.utils import np_utils
import keras.backend as K

from util.util import  num_of_classes, dataset_size, checkpoint_dir, log_path, log_directory, experiment_directory
from .image_datagen import image_generator



def conv_factory(x, nb_filter, dropout_rate=None, weight_decay=1E-4):
    """Apply BatchNorm, Relu 3x3Conv2D, optional dropout
    :param x: Input keras network
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor
    :returns: keras network with b_norm, relu and convolution2d added
    :rtype: keras network
    """

    x = BatchNormalization(mode=0,
                           axis=1,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Convolution2D(nb_filter, 3, 3,
                      init="he_uniform",
                      border_mode="same",
                      bias=False,
                      W_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition(x, nb_filter, dropout_rate=None, weight_decay=1E-4):
    """Apply BatchNorm, Relu 1x1Conv2D, optional dropout and Maxpooling2D
    :param x: keras model
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor
    :returns: model
    :rtype: keras model, after applying batch_norm, relu-conv, dropout, maxpool
    """

    x = BatchNormalization(mode=0,
                           axis=1,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Convolution2D(nb_filter, 1, 1,
                      init="he_uniform",
                      border_mode="same",
                      bias=False,
                      W_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)

    return x


def denseblock(x, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1E-4):
    """Build a denseblock where the output of each
       conv_factory is fed to subsequent ones
    :param x: keras model
    :param nb_layers: int -- the number of layers of conv_
                      factory to append to the model.
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor
    :returns: keras model with nb_layers of conv_factory appended
    :rtype: keras model
    """

    list_feat = [x]

    if K.image_dim_ordering() == "th":
        concat_axis = 1
    elif K.image_dim_ordering() == "tf":
        concat_axis = -1

    for i in range(nb_layers):
        x = conv_factory(x, growth_rate, dropout_rate, weight_decay)
        list_feat.append(x)
        x = merge(list_feat, mode='concat', concat_axis=concat_axis)
        nb_filter += growth_rate

    return x, nb_filter


def denseblock_altern(x, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1E-4):
    """Build a denseblock where the output of each conv_factory
       is fed to subsequent ones. (Alternative of a above)
    :param x: keras model
    :param nb_layers: int -- the number of layers of conv_
                      factory to append to the model.
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor
    :returns: keras model with nb_layers of conv_factory appended
    :rtype: keras model
    * The main difference between this implementation and the implementation
    above is that the one above
    """

    if K.image_dim_ordering() == "th":
        concat_axis = 1
    elif K.image_dim_ordering() == "tf":
        concat_axis = -1

    for i in range(nb_layers):
        merge_tensor = conv_factory(x, growth_rate, dropout_rate, weight_decay)
        x = merge([merge_tensor, x], mode='concat', concat_axis=concat_axis)
        nb_filter += growth_rate

    return x, nb_filter


def DenseNet(nb_classes, img_dim, depth, nb_dense_block, growth_rate, nb_filter, dropout_rate=None, weight_decay=1E-4):
    """ Build the DenseNet model
    :param nb_classes: int -- number of classes
    :param img_dim: tuple -- (channels, rows, columns)
    :param depth: int -- how many layers
    :param nb_dense_block: int -- number of dense blocks to add to end
    :param growth_rate: int -- number of filters to add
    :param nb_filter: int -- number of filters
    :param dropout_rate: float -- dropout rate
    :param weight_decay: float -- weight decay
    :returns: keras model with nb_layers of conv_factory appended
    :rtype: keras model
    """

    model_input = Input(shape=img_dim)

    # assert (depth - 4) % 3 == 0, "Depth must be 3 N + 4"
    assert (depth - 4) % nb_dense_block == 0, "Depth must be {} N + 4".format(nb_dense_block)

    # layers in each dense block
    # nb_layers = int((depth - 4) / 3)
    nb_layers = int((depth - 4) / nb_dense_block)

    # Initial convolution
    x = Convolution2D(nb_filter, 3, 3,
                      init="he_uniform",
                      border_mode="same",
                      name="initial_conv2D",
                      bias=False,
                      W_regularizer=l2(weight_decay))(model_input)



    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        x, nb_filter = denseblock(x, nb_layers, nb_filter, growth_rate,
                                  dropout_rate=dropout_rate,
                                  weight_decay=weight_decay)
        # add transition
        x = transition(x, nb_filter, dropout_rate=dropout_rate,
                       weight_decay=weight_decay)

    # The last denseblock does not have a transition
    x, nb_filter = denseblock(x, nb_layers, nb_filter, growth_rate,
                              dropout_rate=dropout_rate,
                              weight_decay=weight_decay)



    x = BatchNormalization(mode=0,
                           axis=1,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D(dim_ordering="th")(x)
    x = Dense(nb_classes,
              activation='softmax',
              W_regularizer=l2(weight_decay),
              b_regularizer=l2(weight_decay))(x)

    densenet = Model(input=[model_input], output=[x], name="DenseNet")

    return densenet





def densenet(config):
    nb_classes = num_of_classes(config)
    img_dim = (config['img']['img_size'], config['img']['img_size'],3)
    depth = config['densenet']['depth'] # 40, 100
    nb_dense_block = config['densenet']['nb_blocks']
    growth_rate = config['densenet']['growth_rate']
    nb_filter= config['densenet']['first_conv_filters']
    dropout_rate = config['densenet']['dropout_rate']
    weight_decay = config['densenet']['weight_decay']

    model = DenseNet(nb_classes, img_dim, depth, nb_dense_block, growth_rate, nb_filter, 
                     dropout_rate=dropout_rate, weight_decay=weight_decay)

    return model


def prepare_optimizer(config):
    lr = config['solver_exact']['base_lr']
    opt = SGD(lr=lr, decay=1e-4, momentum=0.9, nesterov=True)
    
    return opt


def prepare_model(config):
    
    opt = prepare_optimizer(config)
    model = densenet(config)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def prepare_scheduler(config):
    
    def lr_sch(epoch):
        lr = config['solver_exact']['base_lr']
        gamma = lr = config['solver_exact']['gamma']
        epoch_num = config['train_params']['epoch_num']
        if epoch >= epoch_num * 0.75:
            return lr * gamma**2
        elif epoch >= epoch_num * 0.5:
            return lr * gamma
        else:
            return lr
    
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
    visualizer = TensorBoard(log_dir=log_directory(config, 'tensorboard_logs'), 
                            write_graph=True, histogram_freq=1, write_grads=True, 
                            write_images=True)

    return [lr_scheduler, checkpointer, logger, stopping, visualizer]


def save_history(history, config):
    import pickle
    directory = experiment_directory(config)
    path = '{directory}/history'.format(**locals())

    with open(path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


def train_densenet(config):
    batch_size = config['train_params']['batch_size']
    nb_epoch = config['train_params']['epoch_num']

    train_gen = image_generator(config, 'train')
    train_size = dataset_size(config, 'train')
    
    val_gen = image_generator(config, 'val')
    val_size = dataset_size(config, 'val')
    
    
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
    model.save('{}/latest_version.h5'.format(checkpoint_path(config)))
    save_history(history, config)
    