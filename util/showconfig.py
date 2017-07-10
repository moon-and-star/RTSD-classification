#!/usr/bin/env python

def showconfig(args):
    pass

def setupShowConfigParser(subparsers):
    showconfig_parser = subparsers.add_parser("showconfig", help='- Prints current configuration file.')
    showconfig_parser.set_defaults(func=showconfig)
