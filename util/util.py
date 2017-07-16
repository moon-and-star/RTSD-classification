import sys
import argparse
import os.path as osp
from os import makedirs
import json


def removekey(d, key):
    r = dict(d)
    del r[key]
    return r


def load_image_mean(mean_path):
    if osp.exists(mean_path):
        return map(float, open(mean_path, 'r').read().split())
    else:
        print('WARNING: no mean file!')
    return None


def safe_mkdir(directory_name):
    if not osp.exists(directory_name):
        makedirs(directory_name)




class Params:
    def __setattr__(self, name, value):
        self.__dict__[name] = value

    def __getattr__(self, name):
        return self.__dict__[name]

    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    def __getitem__(self, item):
        return self.__getattr__(item)



