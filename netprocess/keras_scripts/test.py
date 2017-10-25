#!/usr/bin/env python

import numpy as np
from util.util import dataset_size, experiment_directory
from util.util import dataset_size, epoch_size, num_of_classes
from util.util import safe_mkdir, checkpoint_dir

from keras.models import load_model
from .image_datagen import image_generator

import os
import os.path as osp


from itertools import islice
import numpy as np
from skimage.io import imsave


def best_checkpoint(config):
    directory = checkpoint_dir(config) 
    name = sorted(os.listdir(directory))[-1] #get latest checkpoint

    return "{directory}/{name}".format(**locals())



def save_misclassified(misclass, config, phase):
    pred = misclass['pred_clid']
    actual = misclass['act_clid']

    clid, filename = misclass['name'].split('/')
    if actual != int(clid):
        print('ERROR! Class id in filename does not match its label in generator')
        print(actual, clid)
        print(misclass['name'])
        exit()

    new_name = 'act_{actual}_pred_{pred}_{filename}'.format(**locals()) #act_8_pred_99_000000.png
    print(new_name)
    exp_dir = experiment_directory(config)


    directory = '{exp_dir}/misclassified/{phase}/actual/{actual}'.format(**locals())
    safe_mkdir(directory)
    imsave('{directory}/{new_name}'.format(**locals()), misclass['image'].astype(np.uint8))


    directory = '{exp_dir}/misclassified/{phase}/predicted/{pred}'.format(**locals())
    safe_mkdir(directory)
    imsave('{directory}/{new_name}'.format(**locals()), misclass['image'].astype(np.uint8))
    


def get_clid_by_label(label, class_indices):
    for key, value in class_indices.items():
        # print('key = ',key)
        if value == label:
            return int(key)


import shutil
def gather_misclassified(model, config, phase, batch_size=1):
    test_gen = image_generator(config, phase, test_batch_size=batch_size, test_on_train=True)
    print(test_gen.class_indices)
    names = test_gen.filenames
    

    count = 0
    for item in islice(test_gen, dataset_size(config, phase)): #item = [batch_of_imgs, batch_of_labels]
        actual_label = np.argmax(item[1][0])
        pred = model.predict(item[0], batch_size=1)
        prediction = np.argmax(pred)

        if prediction != actual_label:
            misclass = {'image': item[0][0], 
                        'name': test_gen.filenames[count],
                        'pred_clid': get_clid_by_label(prediction, test_gen.class_indices),
                        'act_clid':get_clid_by_label(actual_label, test_gen.class_indices)}
            save_misclassified(misclass, config, phase)         

        count += 1



def class_accuracies(model, config, phase, batch_size=1):
    test_gen = image_generator(config, phase, test_batch_size=batch_size, test_on_train=True)
    size = dataset_size(config, phase)
    class_num = num_of_classes(config)

    accuracies = [0] * class_num
    class_sizes = [0] * class_num


    for item in islice(test_gen, size): #item = [batch_of_imgs, batch_of_labels]
        actual_label = np.argmax(item[1][0])
        pred = model.predict(item[0], batch_size=1)
        prediction = np.argmax(pred)

        actual_clid = get_clid_by_label(actual_label, test_gen.class_indices)
        if actual_label == prediction:
            accuracies[actual_clid] += 1
        class_sizes[actual_clid] += 1


    for i in range(class_num):
        if class_sizes[i] == 0:
            accuracies[i] = -1
        else:
            accuracies[i] /= class_sizes[i]

    print(accuracies)
    print(class_sizes)
    return accuracies




def save_results(config, phase, res, class_acc, names):
    exp_dir = experiment_directory(config)
    
    with open('{exp_dir}/test_results_{phase}.txt'.format(**locals()), 'w') as out:
        content = ''
        for i in range(len(names)):
            content += '{:20}{:20}\n'.format(names[i], res[i])
        out.write(content)   
 
    with open('{exp_dir}/class_acc_{phase}.txt'.format(**locals()), 'w') as out:
        content = '{:20}{:20}\n'.format('class_id', 'class_acc')
        for i in range(len(class_acc)):
            content += '{:20}{:20}\n'.format(i, class_acc[i])
        out.write(content)  



def rm_old_results(config):
    path = '{}/misclassified/'.format(experiment_directory(config))
    if osp.exists(path):
        shutil.rmtree(path)
    


def test_net(config, phases):
    rm_old_results(config)
    print('gpu_num = ', config['train_params']['gpu_num'])
    
    if config['model'] == 'wresnet':
        from .wresnet import prepare_model
    elif config["model"] == 'densenet':
        from .densenet import prepare_model

    print('preparing model')    
    model = prepare_model(config) 
    model.load_weights(best_checkpoint(config)) 


    for phase in phases:
        print('gettig results for {}'.format(phase))
        
        gather_misclassified(model, config, phase)
        class_acc = class_accuracies(model, config, phase)
        # class_acc = class_accuracies(model, test_gen, dataset_size(config, phase), num_of_classes(config))
        test_gen = image_generator(config, phase, test_on_train=True)
        res = model.evaluate_generator(test_gen, dataset_size(config, phase))
        
        print('saving results for {}'.format(phase))
        save_results(config, phase, res, class_acc, model.metrics_names)
        
