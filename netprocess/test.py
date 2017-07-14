#!/usr/bin/env python

def test(args):
    pass

def setupTestParser(subparsers):
    test_parser = subparsers.add_parser("test", help='- Tests net from experiment with specified group \
                                        and number on train (0), validation (1) or test (2) set')
    test_parser.set_defaults(func=test)

