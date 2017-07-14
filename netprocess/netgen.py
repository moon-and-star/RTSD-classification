#!/usr/bin/env python

def netgen(args):
    pass

def setupNetgenParser(subparsers):
     netgen_parser = subparsers.add_parser("netgen", help='- Generates net and solver files for \
                                        experiment with parameters from current config file. Plases \
                                        them into a experiment folder containing copy of a config file.')
     netgen_parser.set_defaults(func=netgen)

