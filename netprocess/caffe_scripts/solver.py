#!/usr/bin/env python
import sys
import pathlib2 as pathlib
local_path = pathlib.Path('./')
absolute_path = local_path.resolve()
sys.path.append(str(absolute_path))

from util.util import safe_mkdir, proto_path, epoch_size, experiment_directory, snapshot_path
from math import ceil

    
def max_iter(config):
    epoch_sz = epoch_size(config, 'train')
    epoch_num = config['train_params']['epoch_num']
    # print(epoch_sz)

    return int(epoch_sz * epoch_num)


def append_train(args, config):
    args['train_net'] = proto_path(config, 'train')
    args['max_iter'] = max_iter(config)
    args['iter_size'] = 1    




def test_frequency(config):
    epoch_freq = config['train_params']['test_frequency']
    iters = epoch_size(config, 'train')

    return int(epoch_freq * iters)


def append_val(args, config):
    args['test_net'] = proto_path(config, 'val')   
    args['test_iter'] = epoch_size(config, 'val')
    args['test_interval'] = test_frequency(config)




def lr_step(config):
    epoch_step = config['train_params']['lr_step']
    epoch_sz = epoch_size(config, 'train')
    return epoch_step * epoch_sz


def append_optimizer(args, config):
    exact = config['solver_exact']
    for param in exact:
        args[param] = exact[param]
        if param == 'lr_policy':
            if args[param] == 'step':
                args['stepsize'] = lr_step(config)



def snapshot_iters(config):
    snap_epoch = config['train_params']['snap_epoch']
    epoch_sz = epoch_size(config, 'train')

    return int(snap_epoch * epoch_sz)


def append_snap(args, config):
    args['snapshot_prefix'] = snapshot_path(config)
    args['snapshot'] = snapshot_iters(config)
  



def solver_args(config):
    args = {}  
    append_train(args, config)
    append_val(args, config)
    append_optimizer(args, config)
    append_snap(args, config)
    args['display'] = 1

    return args
  
    


class SolverParameters(object):    
    def __init__(self, **kwargs):
        super(SolverParameters, self).__init__()
        content = ''

        for arg in sorted(kwargs):
            if type(kwargs[arg]) in [int, float] or arg == 'solver_mode':
                content += '{}: {}\n'.format(arg, kwargs[arg])
            else:
                content += '{}: "{}"\n'.format(arg, kwargs[arg])


        self.content = content


def solver_path(config):
    directory = experiment_directory(config)
    return "{directory}/solver.prototxt".format(**locals())


def gen_solver(config):
    print("Generating solver")

    kwargs = solver_args(config)
    p = SolverParameters(**kwargs)  
    
    path = solver_path(config)
    # path = './solver.txt'
    with open(path, 'w') as f:
        f.write(p.content) 
        print(p.content)






from util.config import get_config
from pprint import pprint

if __name__ == '__main__':
    config = get_config('./config.json')
    gen_solver(config)


