#!/usr/bin/env python
# import sys
# import pathlib2 as pathlib

# local_path = pathlib.Path('./')
# absolute_path = local_path.resolve()
# sys.path.append(str(absolute_path))


from .config import get_config, set_config
from .util import safe_mkdir


# def update_group_list(config):
#     config['group_list']
#     description = config['exp']['group_description']

#     # config['group_list'].append({
#     #     'group_description': description,
#     #     'group_num': config['exp']['group'],
#     #     'experiments': []})



    
def update_config(args):
    config = get_config(args.confpath)

    config['exp']['group_description'] = args.description
    config['exp']['exp_num'] = None 
    config['exp']['group'] = len(config['group_list'])
    config['group_list'].append({
        'group_num': config['exp']['group'],
        'exp_count': 0})

    set_config(args.confpath, config)
   
    
def group_dir(config):
    root = config['exp']['exp_path']
    group = config['exp']['group']
    return '{root}/group_{group}'.format(**locals())


def update_dir(config):
    path = group_dir(config)
    safe_mkdir(path)

    with open('{path}/group_description.txt'.format(**locals()), 'w') as f:
        f.write(config['exp']['group_description'])



def newgroup(args):
    update_config(args)
    config = get_config(args.confpath)
    update_dir(config)
   
    



def setupNewGroupParser(subparsers):
     newgroup_parser = subparsers.add_parser("newgroup", help='- Sets experiment group number to max_group_number + 1.')
     newgroup_parser.set_defaults(func=newgroup)
     newgroup_parser.add_argument('-d','--description',action='store', default="NO_DESCRIPTION", type=str,
                        help='add description for group of experiments')

