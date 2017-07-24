import sys
import argparse
import os.path as osp
from os import makedirs
import json
from math import ceil



def line_count(path):
    with open(path) as f:
        return len(f.readlines())


def removekey(d, key):
    r = dict(d)
    del r[key]
    return r


def load_image_mean(mean_path):
    if osp.exists(mean_path):
        return map(float, open(mean_path, 'r').read().split())
    else:
        print('WARNING: no mean file!')
    return None


def safe_mkdir(directory_name):
    if not osp.exists(directory_name):
        makedirs(directory_name)



def num_of_classes(config):
    gt_path = '{}/gt_train.txt'.format(config['img']['processed_path'])

    with open(gt_path) as f:
        classes = set()
        for line in f.readlines():
            s = line.split(' ')
            classes.add(s[1].replace('\n', ''))

        return len(classes)


def read_gt(config, phase):
    root = config['img']['processed_path']
    gt_path = '{root}/gt_{phase}.txt'.format(**locals())        
    with open(gt_path) as f:
        lines = f.readlines()

    return lines


def proto_path(config, phase):
    directory = experiment_directory(config)
    return "{directory}/{phase}.prototxt".format(**locals())


def dataset_size(config, phase):
    directory = config['img']['processed_path']
    path = '{directory}/gt_{phase}.txt'.format(**locals())

    return line_count(path)
       

def epoch_size(config, phase):
    data_size = dataset_size(config, phase)
    batch_size = config['train_params']['batch_size']
    # print(data_size, batch_size)

    return int(ceil(data_size / float(batch_size)))

def experiment_directory(config):
    group_num = config['exp']['group']
    group = "group_{group_num}".format(**locals())

    exp_dir = config['exp']['exp_path']
    exp_num = config['exp']['exp_num']
    exp = 'exp_{exp_num}'.format(**locals())
    
    return "{exp_dir}/{group}/{exp}".format(**locals())



def snapshot_path(config):
    exp_dir = experiment_directory(config)
    snap_path = '{exp_dir}/snapshots'.format(**locals())
    safe_mkdir(snap_path)
    return '{snap_path}/snap'.format(**locals())



def confirm():
    ans = sys.stdin.readline().replace('\n', '').lower()
    while not(ans in ['yes', 'no']):
        print('Please answer yes or no:')
        ans = sys.stdin.readline().replace('\n', '').lower()

    if ans !='yes':
        print("Terminating.")
        exit()  
    else:
        print("ACTION CONFIRMED.")


class Params:
    def __setattr__(self, name, value):
        self.__dict__[name] = value

    def __getattr__(self, name):
        return self.__dict__[name]

    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    def __getitem__(self, item):
        return self.__getattr__(item)



