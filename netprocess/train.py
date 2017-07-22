#!/usr/bin/env python
import sys
import pathlib2 as pathlib
local_path = pathlib.Path('./')
absolute_path = local_path.resolve()
sys.path.append(str(absolute_path))

from util.config import get_config, set_config
from util.util import confirm
from netgen import netgen
import os.path as osp
import os
from solver import solver_path, experiment_directory


# from termcolor import colored


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
        print("WARNING: current experiment has been trained earlier. Do you want to overwrite it?")
        confirm()



def check_experiment(args):
    config = get_config(args.confpath)
    check_existence(config)
    if args.force == False:
        check_overwriting(config)




def log_path(config):
    directory = experiment_directory(config)
    prefix = config['exp']['log_pref']
    return "{directory}/{prefix}".format(**locals())


def launch_training(config):
    solver = solver_path(config)
    log = log_path(config)
    gpu_num = config['train_params']['gpu_num']
    tools = '/opt/caffe/.build_release/tools'
    os.system("GLOG_logtostderr=0 {tools}/caffe train -gpu {gpu_num}\
     --solver={solver}  2>&1| tee {log}".format(**locals()))




def copy_config(config):
    root = experiment_directory(config)
    path = '{root}/config.json'.format(**locals())
    set_config(path, config)


def train(args):
    check_experiment(args)
    netgen(args)

    config = get_config(args.confpath)
    copy_config(config)
    launch_training(config)
    



def setupTrainParser(subparsers):
    train_parser = subparsers.add_parser("train", help='- Does the same action as netgen, but also \
                                        trains generated network. Increases experiment number after \
                                        training. Does not change experiment group.')
    train_parser.set_defaults(func=train)

    train_parser.add_argument('-f','--force',action='store_true',
                        help='overwrites existing experiment if set')
