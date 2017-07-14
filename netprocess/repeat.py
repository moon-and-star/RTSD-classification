#!/usr/bin/env python

def repeat(args):
    pass

def setupRepeatParser(subparsers):
     repeat_parser = subparsers.add_parser("repeat", help='- Repeats experiment (training) with specified group and number. \
                        Repeats current experiment by default.')
     repeat_parser.set_defaults(func=repeat)


