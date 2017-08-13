#!/usr/bin/env python

import numpy as np
from util.util import dataset_size, experiment_directory
from util.util import proto_path, snapshot_path, dataset_size, epoch_size, num_of_classes
from util.util import safe_mkdir, read_gt, checkpoint_path

from keras.models import load_model
import tensorflow as tf
from .image_datagen import image_generator






def test_net(config):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    phase = 'test'

    filepath = "{}/checkpoint".format(checkpoint_path(config))
    from .wresnet import prepare_model
    model = prepare_model(config) 
    # model = load_model(filepath,custom_objects={"tf": tf})

    test_gen = image_generator(config, phase)
    steps = dataset_size(config, phase)
    res = model.evaluate_generator(test_gen, steps)
    print(res)
