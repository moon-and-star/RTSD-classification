#!/usr/bin/env python
import sys
import pathlib2 as pathlib
local_path = pathlib.Path('./')
absolute_path = local_path.resolve()
sys.path.append(str(absolute_path))

from util.config import get_config, set_config
from util.util import confirm
import os.path as osp
import os
from solver import solver_path, experiment_directory
from train import launch_training


def load_exp_config(config, group, exp):
    root = config['exp']['exp_path']
    path = '{root}/group_{group}/exp_{exp}/config.json'.format(**locals())
    if not osp.exists(path):
        print('ERROR: no config file ("{path}" not found)'.format(**locals()))
        exit()

    return get_config(path)


def repeat(args):
    config = get_config(args.confpath)
    config = load_exp_config(config, args.group_num, args.exp_num)
    launch_training(config)
    


def setupRepeatParser(subparsers):
     repeat_parser = subparsers.add_parser("repeat", help='- Repeats experiment (training) with specified group and number. \
                        Repeats current experiment by default.')
     repeat_parser.set_defaults(func=repeat)

     repeat_parser.add_argument('group_num',action='store', type=int,
                        help='group number')
     repeat_parser.add_argument('exp_num',action='store', type=int,
                        help='experiment number')



