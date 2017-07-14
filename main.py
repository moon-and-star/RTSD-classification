#!/usr/bin/env python

from util.cli import setupCLI

if __name__ == '__main__':
    args = setupCLI()
    args.func(args)
