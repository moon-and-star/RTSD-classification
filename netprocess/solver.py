#!/usr/bin/env python
import sys
import pathlib2 as pathlib
local_path = pathlib.Path('./')
absolute_path = local_path.resolve()
sys.path.append(str(absolute_path))

from util.util import safe_mkdir
from math import ceil


def line_count(path):
    with open(path) as f:
        return len(f.readlines())


def dataset_size(config, phase):
    directory = config['img']['processed_path']
    path = '{directory}/gt_{phase}.txt'.format(**locals())

    return line_count(path)
       

def epoch_size(config, phase):
    data_size = dataset_size(config, phase)
    batch_size = config['train_params']['batch_size']

    return int(ceil(data_size / float(batch_size)))




def experiment_directory(config):
    group_num = config['curr_exp']['group']
    group = "group_{group_num}".format(**locals())

    exp_dir = config['curr_exp']['exp_path']
    exp_num = config['curr_exp']['exp_num']
    exp = 'exp_{exp_num}'.format(**locals())
    
    return "{exp_dir}/{group}/{exp}".format(**locals())


def proto_path(config, phase):
    directory = experiment_directory(config)
    return "{directory}/{phase}.prototxt".format(**locals())



    
def max_iter(config):
    epoch_sz = epoch_size(config, 'train')
    epoch_num = config['train_params']['epoch_num']

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




def snapshot_path(config):
    exp_dir = experiment_directory(config)
    snap_path = '{exp_dir}/snapshots'.format(**locals())
    safe_mkdir(snap_path)
    return '{snap_path}/snap'.format(**locals())


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






from util.config import getConfig
from pprint import pprint

if __name__ == '__main__':
    config = getConfig('./config.json')
    gen_solver(config)


