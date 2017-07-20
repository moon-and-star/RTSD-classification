#!/usr/bin/env python
import sys
import pathlib2 as pathlib
local_path = pathlib.Path('./')
absolute_path = local_path.resolve()
sys.path.append(str(absolute_path))


from wresnet import gen_wresnet
from util.config import get_config


def netgen(args):
    config = get_config(args.confpath)
    gen_wresnet(config)


def setupNetgenParser(subparsers):
     netgen_parser = subparsers.add_parser("netgen", help='- generates prototxt files (nets and solver) for \
                                        experiment with parameters from current config file; saves \
                                        this files into an experiment folder containing copy of the config file.')
     netgen_parser.set_defaults(func=netgen)


