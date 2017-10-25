#!/usr/bin/env python

import os

def test_caffe(config):
    from .caffe_scripts.test_net import test_net
    phases = ['test']
    test_net(config, phases)



def test_keras(config):
    from .keras_scripts.test import test_net 
    
    gpu_num = "1, 2"
    config["train_params"]['gpu_num'] = gpu_num
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num

    phases = ['val', 'test', 'train'] # do not set phase to "train" without changing test script. set train batch size to 1 first
    test_net(config, phases)



def test(args):
    from util.config import get_config, load_exp_config
    from util.util import confirm


    config = get_config(args.confpath)
    config = load_exp_config(config, args.group_num, args.exp_num)

    framework = args.framework
    if framework == 'caffe':
        test_caffe(config)
    elif framework == 'keras':
        test_keras(config)
    else:
        print("ERROR: Unknown framework: {framework}".format(**locals()))

    
  

def setupTestParser(subparsers):
    test_parser = subparsers.add_parser("test", help='- Tests net from experiment with specified group \
                                        and number on train (0), validation (1) or test (2) set')
    test_parser.set_defaults(func=test)

    test_parser.add_argument('group_num',action='store', type=int, help='group number')
    test_parser.add_argument('exp_num',action='store', type=int, help='experiment number')
    test_parser.add_argument('--framework',action='store', type=str, default='keras',
                        help='')



