#!/usr/bin/env python

from .config import init_config
import sys
from .util import safe_mkdir



def confirm():
    ans = sys.stdin.readline().replace('\n', '').lower()
    while not(ans in ['yes', 'no']):
        print('Please answer yes or no:')
        ans = sys.stdin.readline().replace('\n', '').lower()

    if ans !='yes':
        print("Configuration is unchanged. Programm will be terminated")
        exit()  
    else:
        print("ACTION CONFIRMED.")



def init(args):
    print("WARNING: this action will reset configuration to default values. List of experiments will be lost. \nAre you sure? (yes/no):")
    confirm()
    init_config(args.confpath)
    

def setupInitParser(subparsers):
    init_parser = subparsers.add_parser("init", help='- Initializes configuration with default values. \
                        Experiment and group numbers will be set to zero, list of experiments will be lost.')
    init_parser.set_defaults(func=init)

