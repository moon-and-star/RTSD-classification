#!/usr/bin/env python
import sys
import pathlib2 as pathlib

local_path = pathlib.Path('./')
absolute_path = local_path.resolve()
sys.path.append(str(absolute_path))


from .config import get_config, set_config
from .util import safe_mkdir



def update_config(args):
    config = get_config(args.confpath)

    config['exp']['exp_description'] = args.description
    group_num = config['exp']['group']
    group = config['group_list'][group_num]
    config['exp']['exp_num'] = group['exp_count']
    group['exp_count'] += 1

    set_config(args.confpath, config)
   
    
def exp_dir(config):
    root = config['exp']['exp_path']
    group = config['exp']['group']
    exp_num = config['exp']['exp_num']
    return '{root}/group_{group}/exp_{exp_num}'.format(**locals())


def update_dir(config):
    path = exp_dir(config)
    safe_mkdir(path)

    with open('{path}/exp_description.txt'.format(**locals()), 'w') as f:
        f.write(config['exp']['exp_description'])



def newexp(args):
    update_config(args)
    config = get_config(args.confpath)
    update_dir(config)
   

def setupNewExpParser(subparsers):
    newexp_parser = subparsers.add_parser("newexp", help='- Sets experiment number to max_experiment_number + 1 (within current group)')
    newexp_parser.set_defaults(func=newexp)

    newexp_parser.add_argument('-d','--description',action='store', default="NO_DESCRIPTION", type=str,
                        help='add description for experiment')


