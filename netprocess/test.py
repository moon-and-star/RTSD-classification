#!/usr/bin/env python
import sys
import pathlib2 as pathlib
local_path = pathlib.Path('./')
absolute_path = local_path.resolve()
sys.path.append(str(absolute_path))

from util.config import get_config, load_exp_config
from util.util import confirm
from test_net import test_net


def test(args):
    config = get_config(args.confpath)
    config = load_exp_config(config, args.group_num, args.exp_num)

    phases = ['train']
    test_net(config, phases)
  

def setupTestParser(subparsers):
    test_parser = subparsers.add_parser("test", help='- Tests net from experiment with specified group \
                                        and number on train (0), validation (1) or test (2) set')
    test_parser.set_defaults(func=test)

    test_parser.add_argument('group_num',action='store', type=int, help='group number')
    test_parser.add_argument('exp_num',action='store', type=int, help='experiment number')



