#!/usr/bin/env python
import sys
import pathlib2 as pathlib
local_path = pathlib.Path('./')
absolute_path = local_path.resolve()
sys.path.append(str(absolute_path))

from util.config import get_config, load_exp_config
from util.util import confirm
from shutil import copyfile
from train import launch_training


def change_gt(config, phase):
    root = config['exp']['exp_path']
    exp = config['exp']['exp_num']
    group = config['exp']['group']
    src = '{root}/group_{group}/exp_{exp}/gt_{phase}.txt'.format(**locals()) 

    root = config['img']['processed_path']
    dst = '{root}/gt_{phase}.txt'.format(**locals()) 

    copyfile(src, dst)



def repeat(args):
    config = get_config(args.confpath)
    config = load_exp_config(config, args.group_num, args.exp_num)

    print('WARNING: this action will overwrite previous experiment logs. Are you sure? (yes/no)')
    confirm()
    change_gt(config, 'train')
    launch_training(config)
    


def setupRepeatParser(subparsers):
     repeat_parser = subparsers.add_parser("repeat", help='- Repeats experiment (training) with specified group and number. \
                        Repeats current experiment by default.')
     repeat_parser.set_defaults(func=repeat)

     repeat_parser.add_argument('group_num',action='store', type=int, help='group number')
     repeat_parser.add_argument('exp_num',action='store', type=int, help='experiment number')



