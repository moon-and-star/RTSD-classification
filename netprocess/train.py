#!/usr/bin/env python

def train(args):
    pass

def setupTrainParser(subparsers):
    train_parser = subparsers.add_parser("train", help='- Does the same action as netgen, but also \
                                        trains generated network. Increases experiment number after \
                                        training. Does not change experiment group.')
    train_parser.set_defaults(func=train)