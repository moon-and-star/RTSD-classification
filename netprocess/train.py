#!/usr/bin/env python
from util.config import get_config, set_config
from util.util import confirm, experiment_directory 
from util.util import read_gt, exp_gt_path, log_path

import os.path as osp
import os
from random import shuffle


def check_existence(config):
    exp_num = config['exp']['exp_num']
    group = config['exp']['group']
    if exp_num == None or group == None:
        print("ERROR: no experiment created")
        exit()



def check_overwriting(config):
    root = config['exp']['exp_path']
    exp_num = config['exp']['exp_num']
    group = config['exp']['group']
    log = config['exp']['log_pref']
    path = '{root}/group_{group}/exp_{exp_num}/{log}'.format(**locals())
    if osp.exists(path):
        print("WARNING: current experiment has been trained earlier. Do you want to overwrite it?\
            \n(group = {group}, exp_num = {exp_num})".format(**locals()))
        confirm()



def check_experiment(args):
    config = get_config(args.confpath)
    check_existence(config)
    if args.force == False:
        check_overwriting(config)



def train_caffe(config):
    from .caffe_scripts.solver import solver_path

    shuffle_gt(config)

    solver = solver_path(config)
    log = log_path(config)
    gpu_num = config['train_params']['gpu_num']

    tools = '/opt/caffe/build/tools'
    os.system("GLOG_logtostderr=0 {tools}/caffe train -gpu {gpu_num}\
     --solver={solver}  2>&1| tee {log}".format(**locals()))



def train_keras(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(config['train_params']['gpu_num'])
    os.environ["KERAS_BACKEND"] = 'tensorflow'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    if config['model'] == 'wresnet':
        from .keras_scripts.wresnet import train_wresnet
        train_wresnet(config)
    elif config['model'] == 'densenet':
        from .keras_scripts.densenet import train_densenet
        train_densenet(config)
    else:
        print("ERROR: unknown model {}".format(config['model']))
        exit()



def launch_training(config, framework):
    if framework == 'caffe':
        train_caffe(config)
    elif framework == 'keras':
        train_keras(config)
    else:
        print("ERROR: Unknown framework: {framework}".format(**locals()))



def copy_config(config):
    root = experiment_directory(config)
    path = '{root}/config.json'.format(**locals())
    set_config(path, config)



# needed for caffe
def shuffle_gt(config):
    for phase in ['train', 'test', 'val']:
        gt = read_gt(config, phase)
        shuffle(gt)

        gt_path = exp_gt_path(config, phase)
        with open(gt_path, 'w') as f:
            for line in gt:
                f.write(line)
       
       
def upload_results(config):
    exp_path = experiment_directory(config)
    os.system("git add {exp_path}".format(**locals()))

    exp_num = config['exp']['exp_num']
    group = config['exp']['group']
    os.system("git commit -m 'results for {group}.{exp_num}'".format(**locals()))

    os.system("git push")
    


def train(args):   
    check_experiment(args)
    config = get_config(args.confpath)
    copy_config(config)

    framework = args.framework
    if framework == 'caffe':
        from .caffe_scripts.netgen import netgen # for caffe
        netgen(args)

    launch_training(config, framework)

    if args.upload:
        upload_results(config)
    



def setupTrainParser(subparsers):
    train_parser = subparsers.add_parser("train", help='- Does the same action as netgen, but also \
                                        trains generated network. Increases experiment number after \
                                        training. Does not change experiment group.')
    train_parser.set_defaults(func=train)

    train_parser.add_argument('-f','--force',action='store_true',
                        help='overwrites existing experiment if set')
    train_parser.add_argument('-u','--upload',action='store_true',
                        help='upload results (to github)')
    train_parser.add_argument('--framework',action='store', type=str, default='keras',
                        help='')
