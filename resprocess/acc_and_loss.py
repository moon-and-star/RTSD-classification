#!/usr/bin/env python
    
import numpy as np
import argparse
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

from operator import itemgetter
import seaborn

    
def init_plot(log, start_epoch):
    plt.figure(figsize=(20,10))
    ax = plt.subplot()

    ax.set_xticks(range(len(log['epoch'])))
    ax.tick_params(labelsize=20)

    return ax



def plot_logscale_err(log, path, start_epoch=0, ax=None):
    ax.semilogy(log['epoch'][start_epoch:], 1 - np.array(log['acc'][start_epoch:]), label='train error')
    ax.semilogy(log['epoch'][start_epoch:], 1 - np.array(log['val_acc'][start_epoch:]), label='validation error')
    ax.legend(bbox_to_anchor=(1.0, 1.0), fontsize=24)

    plt.xlabel('epoch number', fontsize=20)
    plt.title('Logscale error rate', fontsize= 28)
    plt.ylabel('error, %', fontsize=24)

    if path != None:
        plt.savefig('{path}/logscale_error_rate_{start_epoch}.png'.format(**locals()))




def plot_err(log, path, start_epoch=0, ax=None):
    ax.plot(log['epoch'][start_epoch:], 1 - np.array(log['acc'][start_epoch:]), label='train error')
    ax.plot(log['epoch'][start_epoch:], 1 - np.array(log['val_acc'][start_epoch:]), label='validation error')
    ax.legend(bbox_to_anchor=(1.0, 1.0), fontsize=24)

    plt.xlabel('epoch number', fontsize=20)
    plt.title('Error rate', fontsize= 28)
    plt.ylabel('error, %', fontsize=24)

    if path != None:
        plt.savefig('{path}/error_rate_{start_epoch}.png'.format(**locals()))



def plot_logscale_loss(log, path, start_epoch=0):
    plt.semilogy(log['epoch'][start_epoch:], np.array(log['loss'][start_epoch:]), label='train loss')
    plt.semilogy(log['epoch'][start_epoch:], np.array(log['val_loss'][start_epoch:]), label='validation loss')
    plt.legend(bbox_to_anchor=(1.0, 1.0), fontsize=24)

    plt.xlabel('epoch number', fontsize=20)
    plt.title('Logscale loss', fontsize= 28)
    plt.ylabel('loss', fontsize=24)

    if path != None:
        plt.savefig('{path}/logscale_loss_{start_epoch}.png'.format(**locals()))
   


def plot_loss(log, path, start_epoch=0):
    plt.plot(log['epoch'][start_epoch:], np.array(log['loss'][start_epoch:]), label='train loss')
    plt.plot(log['epoch'][start_epoch:], np.array(log['val_loss'][start_epoch:]), label='validation loss')
    plt.legend(bbox_to_anchor=(1.0, 1.0), fontsize=24)

    plt.xlabel('epoch number', fontsize=20)
    plt.title('Loss', fontsize= 28)
    plt.ylabel('loss', fontsize=24)

    if path != None:
        plt.savefig('{path}/loss_{start_epoch}.png'.format(**locals()))

    

def close_plot():
    plt.clf()
    plt.cla()
    plt.close()



def plot_log(log, path=None, start_epoch=0):
    ax = init_plot(log, start_epoch)
    plot_err(log, path, start_epoch=start_epoch, ax=ax)
    close_plot()

    ax = init_plot(log, start_epoch)
    plot_logscale_err(log, path, start_epoch=start_epoch, ax=ax)
    close_plot()

    init_plot(log, start_epoch)
    plot_loss(log, path, start_epoch=start_epoch)
    close_plot()

    init_plot(log, start_epoch)
    plot_logscale_loss(log, path, start_epoch=start_epoch)
    close_plot()
    # plt.savefig('./ololo.png')
    # exit()
   
 


def pictogram(class_name):
    pict_name = class_name + ".png"
    directory = "../RTSD/pictograms/"
    if pict_name in listdir(directory):
        return (rescale(imread(directory + pict_name), 0.5) * 255).astype(np.uint8)
    else:
        for pict_name in reversed(sorted(listdir(directory))):
            if pict_name.split('.')[0] in class_name and \
                    ("n" in class_name or "r" in class_name or "3_4_1" in class_name):
                return (rescale(imread(directory + pict_name), 0.5) * 255).astype(np.uint8)
        return np.ones((32, 32, 3)).astype(np.uint8) * 255


def get_log(exp_dir):
    log = {}
    with open('{exp_dir}/training_log'.format(**locals())) as f:
        column_names = f.readline().replace('\n', '').split(',')
        for name in column_names:
            log[name] = []

        for line in f.readlines():
            split = line.split(',')
            log['epoch'] += [int(split[0])]
            for i in range(1, len(column_names)):
                log[column_names[i]] += [float(split[i])]

    return log



from pprint import pprint
if __name__ == '__main__':
    group = 5
    for exp_num in range(19): 
        exp_dir = './Experiments/group_{group}/exp_{exp_num}'.format(**locals())
        print(exp_dir)
        log = get_log(exp_dir)
        # pprint(list(zip(*[log[key] for key in log])))
        for start_epoch in [0, 3, 5]:
            plot_log(log, path=exp_dir, start_epoch=start_epoch)

    


