#!/usr/bin/env python

def newexp(args):
    pass

def setupNewExpParser(subparsers):
    newexp_parser = subparsers.add_parser("newexp", help='- Sets experiment number to max_experiment_number + 1 (within current group)')
    newexp_parser.set_defaults(func=newexp)

