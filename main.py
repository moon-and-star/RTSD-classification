#!/usr/bin/env python


import sys
import pathlib2 as pathlib
local_path = pathlib.Path('./')
absolute_path = local_path.resolve()
sys.path.append(str(absolute_path))



from util.cli import setupCLI

if __name__ == '__main__':
    args = setupCLI()
    args.func(args)
