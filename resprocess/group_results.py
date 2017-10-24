#!/usr/bin/env python
import sys
import pathlib2 as pathlib
absolute_path = pathlib.Path('./').resolve()
sys.path.append(str(absolute_path))



import os
from pprint import pprint
import json 
from util.config import get_config, set_config
from util.util import experiment_directory


def save_table(table, path=''):
    with open(path + '.txt', 'w') as out:
        out.write('{:10}{:10}{:10}{:10}{:10}\n'.format('exp_num', 'val_acc', 'val_loss', 'test_acc', 'test_loss'))
        for entry in reversed(table):
            content = '{:<10}{:<10.5}{:<10.5}{:<10.5}{:<10.5}\n'.format(entry['exp_num'], entry['val']['acc'], entry['val']['loss'], entry['test']['acc'], entry['test']['loss'])
            print(content)
            out.write(content)

    with open(path + '.json', 'w') as out:
        json.dump(table, out, indent=4)


def gather_test(group, exp_range):
    table = []
    for exp_num in range(19):
        entry = {}
        entry['exp_num'] = exp_num
        for phase in ['val', 'test']:
            phase_entry = {}
            exp_dir = './Experiments/group_{group}/exp_{exp_num}'.format(**locals())
            with open('{exp_dir}/test_results_{phase}.txt'.format(**locals())) as f:
                for line in f.readlines():
                    s = line.split()
                    metric = s[0].strip()
                    print(s)
                    value = float(s[1])
                    phase_entry[metric] = value
            entry[phase] = phase_entry
        table +=[entry]
    
    sorted_table = sorted(table, key = lambda x: x['test']['acc'])
    print(json.dumps(sorted_table, indent=4)

    save_table(sorted_table, path='./Experiments/group_{group}/results'.format(**locals()))




# gather all results
# group: number of group to gather results from
# exp_range: list of exp numbers to take into account
def gather_results(group, exp_range):
    gather_test(group, exp_range)



if __name__ == '__main__':
    group = 6
    exp_range = range(10)
    gather_results(group, exp_range)
    # table = []
    # for exp_num in range(19):
    #     entry = {}
    #     entry['exp_num'] = exp_num
    #     for phase in ['val', 'test']:
    #         phase_entry = {}
    #         exp_dir = './Experiments/group_{group}/exp_{exp_num}'.format(**locals())
    #         with open('{exp_dir}/test_results_{phase}.txt'.format(**locals())) as f:
    #             for line in f.readlines():
    #                 s = line.split()
    #                 metric = s[0].strip()
    #                 print(s)
    #                 value = float(s[1])
    #                 phase_entry[metric] = value
    #         entry[phase] = phase_entry
    #     table +=[entry]
    
    # sorted_table = sorted(table, key = lambda x: x['test']['acc'])
    # print(json.dumps(sorted_table, indent=4)

    # save_table(sorted_table, path='./Experiments/group_{group}/results'.format(**locals()))
