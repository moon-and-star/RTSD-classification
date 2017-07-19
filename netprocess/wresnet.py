#!/usr/bin/env python

import sys
sys.path.append('/opt/caffe/python/')

import pathlib2 as pathlib
local_path = pathlib.Path('./')
absolute_path = local_path.resolve()
sys.path.append(str(absolute_path))


from util.util import safe_mkdir, load_image_mean
from solver import gen_solver, proto_path

import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2



def get_caffe_phase(phase):
    if phase == "train":
        return "TRAIN"
    elif phase == "val":
        return "TEST"
    elif phase == "test":
        return "TEST"
    

def get_transform_param(mean, img_size):
    if mean is not None:
        return dict(mirror=False, crop_size=img_size , mean_value = map(int, mean), scale=1.0/255)
    else:
        return dict(mirror=False, crop_size=img_size , scale=1.0/255)
        

def append_data(net, phase, **kwargs):
    img_size, img_path = kwargs['img_size'], kwargs['img_path']
    batch_size, pad = kwargs['batch_size'], kwargs['pad']

    mean_path = '{img_path}/{phase}/mean.txt'.format(**locals())
    mean = load_image_mean(mean_path)
    root = "{img_path}/{phase}/".format(**locals())

    image_data_param = dict(
        source = '{img_path}/gt_{phase}.txt'.format(**locals()), 
        batch_size = batch_size,
        new_height = img_size + 2 * pad,
        new_width = img_size + 2 * pad,
        shuffle=True,
        root_folder=root)

    PHASE = get_caffe_phase(phase)
    net['data'], net['label'] = L.ImageData(
        image_data_param = image_data_param,
        transform_param = get_transform_param(mean, img_size),
        ntop = 2,
        include = dict(phase = caffe_pb2.Phase.Value(PHASE)),
        name = "data")

    return net.data, net.label



def convolution(bottom, **kwargs):
    return L.Convolution(bottom,
        kernel_size=kwargs['kernel_size'],
        stride=kwargs['stride'],
        num_output=kwargs['num_output'],
        pad=kwargs['pad'],
        group=kwargs['group'],
        weight_filler = dict(type = 'xavier'),
        convolution_param = dict(engine=1), #1 means CAFFE
        name = kwargs['name'])    



def bn_relu_conv(bottom, **kwargs):
    # bn = L.BatchNorm(bottom, use_global_stats=True, in_place=True)
    # relu = L.ReLU(bn, in_place = True, engine=1)

    bn = L.BatchNorm(bottom, use_global_stats=True)
    relu = L.ReLU(bn, engine=1)

    # relu = L.ReLU(bottom, engine=1)

    conv = convolution(relu, **kwargs)

    return conv


def append_conv(net, bottom, **kwargs):
    # conv = convolution(bottom, **kwargs)
    conv = bn_relu_conv(bottom, **kwargs)
    setattr(net, kwargs['name'], conv)
    return conv




def c1_layer(bottom, **kwargs):
    kwargs["name"] = kwargs['block_name'] + '_c1'

    if kwargs['downsample']:
        kwargs['stride'] = kwargs['downsampling_stride']
        conv = bn_relu_conv(bottom, **kwargs)
    else:
        conv = bn_relu_conv(bottom, **kwargs)

    return conv


def c2_layer(bottom, **kwargs):
    kwargs["name"] = kwargs['block_name'] + '_c2'
    return bn_relu_conv(bottom, **kwargs)


def shortcut_layer(bottom, **kwargs):
    if kwargs['downsample']:
        kwargs['kernel_size'] = 1
        kwargs['stride'] = kwargs['downsampling_stride']
        kwargs['pad'] = 0
        kwargs['name'] = kwargs['name'] + '_shortcut'
        return convolution(bottom, **kwargs)
    else:
        return bottom


def sum_layer(layer1, layer2, name):
    return L.Eltwise(layer1, layer2, 
        name = name,
        eltwise_param = dict(operation=1)) # 1 means sum



def basic_block(bottom, **kwargs):
    conv1 = c1_layer(bottom, **kwargs)
    conv2 = c2_layer(conv1, **kwargs)
    shortcut = shortcut_layer(bottom, **kwargs)

    out = sum_layer(shortcut, conv2, name=kwargs['block_name'] + '_sum')
    return out



def append_conv_block(net, bottom, **kwargs):
    for i in range(kwargs['depth']):
        downsample = (i == 0) and (kwargs['downsample'] == True)
        kwargs['downsample'] = downsample
        kwargs['block_name'] = '{}_b{}'.format(kwargs['name'], i+1)
        bottom = basic_block(bottom, **kwargs)

    setattr(net, kwargs['name'], bottom)
    return bottom



def bn_relu_avepool(bottom, **kwargs ):
    # bn = L.BatchNorm(bottom, use_global_stats=True, in_place=True)
    # relu = L.ReLU(bn, in_place = True)
    bn = L.BatchNorm(bottom, use_global_stats=True)
    relu = L.ReLU(bn, engine=1, in_place=True)
    return L.Pooling(relu, **kwargs)


def append_pool(net, bottom, **kwargs):
    pool = bn_relu_avepool(bottom, **kwargs)
    setattr(net, kwargs['name'], pool)    

    return pool


def append_fc(net, bottom, **kwargs):
    fc = L.InnerProduct(bottom, **kwargs)
    setattr(net, kwargs['name'], fc) 

    return fc


def dropout(name, bottom, dropout_ratio):
    return L.Dropout(
        bottom,
        in_place = True,
        dropout_param = dict(dropout_ratio = dropout_ratio),
        name = name
    )



def append_softmax(net, bottom):
    activ = L.Softmax(bottom, in_place=False, engine=1)
    setattr(net, 'softmax', activ)
    return activ    


def accuracy(name, bottom, labels, top_k):
    return L.Accuracy(
        bottom,
        labels,
        accuracy_param = dict(top_k = top_k),
        name = name
    )



def append_loss(net, bottom, label, phase):
    net.loss = L.MultinomialLogisticLoss(bottom, label, ntop=1)
    
    if phase != "train":
        net.accuracy_1 = accuracy("accuracy_1", bottom, label, 1)
        net.accuracy_5 = accuracy("accuracy_5", bottom, label, 5)

    return net.loss




def data_args(config):
    args = {}
    args['img_path'] = config['img']['processed_path']
    args['img_size'] = config['img']['img_size']
    args['pad'] = config['img']['padding']
    args['batch_size'] = config['train_params']['batch_size']
    # args['root_folder'] = config[]

    return args


def conv1_args(config=None):
    args = {}
    args['num_output'] = 16
    args['name'] = 'conv1'
    args['kernel_size'] = 3
    args['stride'] = 1
    args['pad'] = 1
    args['group'] = 1
    return args


def conv2_args(config):
    args = {}

    args['name'] = 'conv2'
    args['kernel_size'] = 3
    args['stride'] = 1
    args['pad'] = int(args['kernel_size'] / 2)
    args['group'] = 1

    args['depth'] = config['wide_resnet']['metablock_depth']
    width = config['wide_resnet']['width'] 
    args['num_output'] = width * 16

    if width > 1:
        args['downsample'] = True  
        args['downsampling_stride'] = 1 
    else:
        args['downsample'] = False 

    return args


def conv3_args(config):
    args = {}
    
    args['name'] = 'conv3'
    args['kernel_size'] = 3
    args['stride'] = 1
    args['pad'] = int(args['kernel_size'] / 2)
    args['group'] = 1

    args['depth'] = config['wide_resnet']['metablock_depth']
    args['num_output'] = config['wide_resnet']['width'] * 32
    args['downsample'] = True
    args['downsampling_stride'] = 2

    return args


def conv4_args(config):
    args = {}
    
    args['name'] = 'conv4'
    args['kernel_size'] = 3
    args['stride'] = 1
    args['pad'] = int(args['kernel_size'] / 2)
    args['group'] = 1

    args['depth'] = config['wide_resnet']['metablock_depth']
    args['num_output'] = config['wide_resnet']['width'] * 64
    args['downsample'] = True
    args['downsampling_stride'] = 2

    return args


def avgpool_args():
    args = {}
    args['name'] = 'avg_pool'
    args['kernel_size'] = 8
    args['stride'] = 1

    return args


def NumOfClasses(config):
    gt_path = '{}/gt_train.txt'.format(config['img']['processed_path'])

    with open(gt_path) as f:
        classes = set()
        for line in f.readlines():
            s = line.split(' ')
            classes.add(s[1].replace('\n', ''))

        return len(classes)


def fc_args(config):
    args = {}
    args['name'] = 'fc'
    args['num_output'] = NumOfClasses(config)
    args['weight_filler'] = dict(type = 'xavier')

    return args


def append_tail(net, bottom, label, phase):
    if phase == 'train':
        net.loss = L.SoftmaxWithLoss(bottom, label, in_place=True)
    else:
        softmax = append_softmax(net, bottom)
        loss = append_loss(net, softmax, label, phase)



def wresnet(config, phase):
    net = caffe.NetSpec()

    data, label = append_data(net, phase, **data_args(config))
    # net.silence = L.Silence(data, ntop=0)
    conv1 = append_conv(net, data, **conv1_args(config))
    conv2 = append_conv_block(net, conv1, **conv2_args(config))
    conv3 = append_conv_block(net, conv2, **conv3_args(config))
    conv4 = append_conv_block(net, conv3, **conv4_args(config))
    pool = append_pool(net, conv4, **avgpool_args())

    
    fc = append_fc(net, pool, **fc_args(config))
    append_tail(net, fc, label, phase)
    
    # net.scale = L.Scale(label, scale_param=dict(filler=dict(value=1)))

    return net.to_proto()




def construct(config):
    for phase in ['train', 'val', 'test']:
        wrn = wresnet(config, phase)
        path = proto_path(config, phase)

        with open(path, 'w') as out:
            out.write(str(wrn))

    gen_solver(config)




from util.config import getConfig
from pprint import pprint

if __name__ == '__main__':
    config = getConfig('./config.json')
    construct(config)