#!/usr/bin/env python
import sys
import pathlib2 as pathlib

local_path = pathlib.Path('./')
absolute_path = local_path.resolve()
sys.path.append(str(absolute_path))


from config import get_config, set_config
from pprint import pprint

# exp = {}
# exp['exp_path'] = './Experiments'
# exp['group'] = None
# exp['group_description'] = ''
# exp['exp_num'] = None
# exp['exp_description'] = ''
# config['curr_exp'] = exp
# # config['exp_list'] = [[0]] #group 0 contains only 1 experiment - 0 
# config['exp_list'] = []

def newgroup(args):
    config = get_config(args.confpath)

    if config['curr_exp']['group'] == None:
        config['curr_exp']['group'] = 0
    else:
        config['curr_exp']['group'] += 1

    group = {'experiments': [], 'description': args.description}
    config['exp_list'].append(group)
    config['curr_exp']['exp'] = None

    print(config['exp_list'])
    print('\n\n')
    



def setupNewGroupParser(subparsers):
     newgroup_parser = subparsers.add_parser("newgroup", help='- Sets experiment group number to max_group_number + 1.')
     newgroup_parser.set_defaults(func=newgroup)
     newgroup_parser.add_argument('-d','--description',action='store', default="NO_DESCRIPTION", type=str,
                        help='add description for group of experiments')

