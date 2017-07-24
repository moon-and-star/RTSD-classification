#!/usr/bin/env python

import numpy as np
from util.util import dataset_size, experiment_directory
from solver import max_iter
from util.util import proto_path, snapshot_path, dataset_size, epoch_size, num_of_classes

from util.util import  load_image_mean, safe_mkdir, read_gt
import math
import fileinput
import sys
sys.path.append('/opt/caffe/python/')

from skimage.io import imread, imsave
import caffe



def set_batch_size(model, n):
    tmp = ''
    with  open(model, 'r') as file:
        for line in file:
            if 'batch_size' in line:
                s = line.split(':')
                tmp += line.replace('batch_size: {}'.format(int(s[1])), 'batch_size: {}'.format(n)) 
            else:
                tmp += line
    
    with  open(model, 'w') as file:
        file.write(tmp)
    print('batch size has been set to {}'.format(n))



def load_net(config, phase):
    prefix = snapshot_path(config)
    # batch_size = config['train_params']['batch_size']
    # iter_num = int(epoch_size(config) * math.ceil(float(dataset_size(config, phase)) / batch_size))
    iter_num = max_iter(config)
    weights = '{prefix}_iter_{iter_num}.caffemodel'.format(**locals())

    model = proto_path(config, phase)
    set_batch_size(model, 1)
    net = caffe.Net(model,1, weights=weights)

    return net





def net_output(config, phase='test'):  
    net = load_net(config, phase)
    size = dataset_size(config, phase)
    classes = num_of_classes(dataset)
    softmax = np.zeros((size, classes))

    for i in range(size):
        if i % 100 == 0:
            print(i)
        out = net.forward()
        softmax[i] = net.blobs['softmax'].data

    return softmax





def init_answers(length):
    res = []
    for i in range(length):
        tmp = {}
        tmp['correct'] = 0
        tmp['total'] = 0

        res += [tmp]  
    return res  



def get_accuracy(softmax, gt):
    acc = {}
    acc['total']= 0
    for i in range(size):
        label = int(gt[i].replace('\n', '').split(' ')[1])
        prediction = np.argmax(softmax[i])
        class_answers[label]['total'] += 1.0

        if label == prediction:
            class_answers[label]['correct'] += 1.0
            acc['total'] += 1.0 

    acc['total'] /= size
    for label in len(class_answers):
        correct = class_answers[label]['correct']
        total = class_answers[label]['total']
        if total > 0:
            acc[str(label)] = correct/total
        else:
            acc[str(label)] = "none"

    return acc



def save_misclassified(softmax, gt):
    misclass = []
    size = softmax.shape[0]

    for i in range(size):
        label = int(gt[i].replace('\n', '').split(' ')[1])
        prediction = np.argmax(softmax[i])

        if label != prediction:
            gt_entry = gt[i].replace('\n', '')
            misclass.append((gt_entry, prediction))

    return misclass
            

def save_accuracy(accuracy, config):
    root = experiment_directory(config)
    acc_path = '{root}/accuracy_{phase}.txt'.format(**locals())
    
    with open(acc_path, 'w') as out:
        out.write('total accuracy: {}\n'.format(accuracy['total']))
        out.write('{:<10} {:<10}\n'.format('class', 'class_acc'))

        for label in sorted(accuracy.remove('total')):
            acc = accuracy[label]
            content = '{label:<10} {acc:<10}\n'.format(**locals())
            out.write(content)




def save_misclassified(misclass, config):
    root = experiment_directory(config)
    misclassified = '{root}/misclassified_{phase}.txt'.format(**locals())
    with open(misclassified, 'w') as out:
        out.write('{:<20} {:<10}'.format('gt_entry', 'prediction'))
        for gt_entry, prediction in misclass:
            content = '{gt_entry:<20} {prediction:<10}\n'.format(**locals())
            out.write(content)



def test_net(config):
    caffe.set_mode_gpu()
    caffe.set_device(0)
    # for phase in ['test', 'train', 'val']:
    for phase in ['test']:
        softmax = net_output(config, phase)
        gt = read_gt(config, phase)
        accuracy = get_accuracy(softmax, gt)
        misclassified = get_misclassified(softmax, gt)

        save_accuracy(accuracy, config)
        save_misclassified(misclassified, config)




if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('EXPERIMENT_NUMBER',type=int, 
    #                     help='the number of current experiment with nets ')
    # args = parser.parse_args()
    # TestCommitee(args.EXPERIMENT_NUMBER, 'RTSD')                   
    pass