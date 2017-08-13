#!/usr/bin/env python


def test_caffe(config):
    from .caffe_scripts.test_net import test_net
    phases = ['test']
    test_net(config, phases)



def test_keras(config):
    from .keras_scripts.test import test_net
    test_net(config)



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



