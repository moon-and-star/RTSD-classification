#!/usr/bin/env python

def newgroup(args):
    pass

def setupNewGroupParser(subparsers):
     newgroup_parser = subparsers.add_parser("newgroup", help='- Sets experiment group number to max_group_number + 1.')
     newgroup_parser.set_defaults(func=newgroup)

