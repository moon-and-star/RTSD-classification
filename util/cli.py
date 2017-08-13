#!/usr/bin/env python

import argparse
import sys
import pathlib2 as pathlib
local_path = pathlib.Path('./')
absolute_path = local_path.resolve()
sys.path.append(str(absolute_path))


from imgprocess.improc import setupImprocParser
# from netprocess.netgen import setupNetgenParser
from netprocess.train import setupTrainParser
from netprocess.test import setupTestParser
from netprocess.repeat import setupRepeatParser

from .newgroup import setupNewGroupParser
from .newexp import setupNewExpParser
from .showconfig import setupShowConfigParser
from .init import setupInitParser




def setupCLI():
    description = """
    DESCRIPTION:
    No description
    """
    parser = argparse.ArgumentParser(description=description)
    default='./config.json'
    parser.add_argument('-c', '--confpath', help='sets config file path, default: "{}"'.format(default),type=str, default=default)

    subparsers = parser.add_subparsers(help='List of commands', dest='command_name')
    setupImprocParser(subparsers)
    # setupNetgenParser(subparsers)
    setupTrainParser(subparsers)
    setupTestParser(subparsers)   
    setupRepeatParser(subparsers)
    setupNewGroupParser(subparsers)
    setupNewExpParser(subparsers)
    setupShowConfigParser(subparsers)
    setupInitParser(subparsers)


    if len(sys.argv) == 1:
        parser.print_help()
        exit()

    args = parser.parse_args()
    return args

 


if __name__ == '__main__':
    pass
