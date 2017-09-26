#!/usr/bin/env python

import sys
home = '/home/GRAPHICS2/20e_ame'
sys.path.append("{home}/anaconda2/pkgs/pathlib2-2.2.1-py27_0/lib/python2.7/site-packages/".format(**locals()))
sys.path.append("{home}/anaconda2/pkgs/scandir-1.4-py27_0/lib/python2.7/site-packages/".format(**locals()))

import pathlib2 as pathlib
absolute_path = pathlib.Path('./').resolve()
sys.path.append(str(absolute_path))


from util.cli import setupCLI

if __name__ == '__main__':
    args = setupCLI()
    args.func(args)
