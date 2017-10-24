#!/usr/bin/env python
    
import numpy as np
import matplotlib
matplotlib.use('agg')
import argparse
from matplotlib import pyplot as plt
from operator import itemgetter
import seaborn

def read_accuracies(exp_dir, phase):
    data = []
    labels = []
    with open('{exp_dir}/class_acc_{phase}.txt'.format(**locals()), 'r') as f:
        for line in f.readlines()[1:]:
            line = ' '.join(line.split())
            print(line)
            s = line.split(" ")
            clid = int(s[0])
            acc = float(s[1])
            data += [acc]
            labels += [clid]

    return zip(*sorted(zip(data, labels), reverse=True, key=itemgetter(0)))



def plot(data, labels, captions, path=None):
    plt.figure(figsize=(20,10))
    
    ax = plt.subplot()
    ax.yaxis.grid(color='gray', linestyle='dashed')
    ax.xaxis.grid(color='gray', linestyle='dashed')


    tics = list(range(len(labels)))
    ax.set_xticks(tics)
    ax.set_xticklabels(labels, rotation="vertical", fontsize=20)
    ax.tick_params(labelsize=20)


    bins = list(range(len(data)))
    r = plt.bar(bins, data, alpha=0.8, color= '#006040' , width=1.0)

    for i, p in enumerate(r):
        if i % 2 == 0:
            p.set_color('#008060')


    plt.xlabel(captions['x'], fontsize=24)
    plt.title(captions['title'], fontsize=28)
    plt.ylabel(captions['y'], fontsize=24)

    # plt.show()
    if path != None:
        plt.savefig(path)




def histogram(exp_dir, phase):
    acc, labels = read_accuracies(exp_dir, phase)
    print(acc, labels)
    
    captions = {}
    captions['x'] = 'Class labels'
    captions['title'] = 'Accuracy distibution {}'.format(300)
    captions['y'] = 'Accuracy, %'

    outpath = "{}/acc_dist.png".format(exp_dir)
    plot(acc, labels, captions, outpath)



    err = list(1 - np.array(acc))
    print(err, labels)

    captions['title'] = 'Error distibution {}'.format(300)
    captions['y'] = 'Error, %'

    outpath = "{}/err_dist.png".format(exp_dir)
    plot(err, labels, captions, outpath)
    





if __name__ == '__main__':
    group = 5
    for exp_num in range(19): 
        for phase in ['val', 'test']:
            print(phase)
            exp_dir = './Experiments/group_{group}/exp_{exp_num}'.format(**locals())
            histogram(exp_dir, phase)

    


