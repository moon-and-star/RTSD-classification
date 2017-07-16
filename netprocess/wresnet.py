#!/usr/bin/env python

import sys
sys.path.append('/opt/caffe/python/')

import pathlib2 as pathlib
local_path = pathlib.Path('./')
absolute_path = local_path.resolve()
sys.path.append(str(absolute_path))


from util.util import safe_mkdir, load_image_mean
# from gen_solver import GenSingleNetSolver

import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2

# def conv_relu(n, name, bottom, kernel_size, num_output, stride=1, pad=1, group=1):
#     conv = L.Convolution(
#         bottom,
#         kernel_size=kernel_size,
#         stride=stride,
#         num_output=num_output,
#         pad=pad,
#         group=group,
#         weight_filler = dict(type = 'xavier'),
#         name = name,
#     )

#     setattr(n, name, conv)

#     return conv, L.ReLU(conv, in_place = True, name = "{}_relu".format(name))


# def conv_stanh(n, name, bottom, kernel_size, num_output, stride=1, pad=1, group=1):
#     conv = L.Convolution(
#         bottom,
#         kernel_size=kernel_size,
#         stride=stride,
#         num_output=num_output,
#         pad=pad,
#         group=group,
#         weight_filler = dict(type = 'xavier'),
#         name = name,
#     )
#     setattr(n, name, conv)

    
#     scale1 = L.Scale(conv, in_place = True, name = "{}_prescale".format(name),
#                      param=dict(lr_mult=0, decay_mult=0),
#                      scale_param=dict(filler=dict(value=0.6666), bias_term=False)
#                      )
#     tanh =  L.TanH(scale1, in_place = True, name = "{}_sTanH".format(name))
#     scale2 = L.Scale(tanh, in_place = True, name = "{}_postscale".format(name),
#                      param=dict(lr_mult=0, decay_mult=0),
#                      scale_param=dict(filler=dict(value=1.7159), bias_term=False)
#                      )

#     return conv, scale2


# def conv1(n, name, bottom, num_output, kernel_size = 3, pad = None, activ = "relu", group=1):
#     if pad is None: pad = kernel_size / 2
#     if activ == "relu":
#         conv1, relu1 = conv_relu(n, "{}".format(name), bottom, kernel_size, num_output, pad=pad, group=group)
#         return relu1
#     elif activ == "scaled_tanh":
#         conv1, scale2 = conv_stanh(n, "{}".format(name), bottom, kernel_size, num_output, pad=pad, group=group)
#         return scale2
    


def avepool(name, bottom, kernel_size = 2, stride = 2):
    return L.Pooling(bottom, kernel_size = kernel_size, stride = stride, pool = P.Pooling.AVE, name = name)


def fc(name, bottom, num_output, activ="relu"):
    fc = L.InnerProduct(
        bottom,
        num_output = num_output,
        weight_filler = dict(type = 'xavier'),
        name = "{}_{}".format(name, num_output)
    )

    if activ=="relu":
        return fc, L.ReLU(fc, in_place = True, name = "{}_relu".format(name))

    elif activ=="scaled_tanh":
        scale1 = L.Scale(fc, in_place = True, name = "{}_prescale".format(name),
                         param=dict(lr_mult=0, decay_mult=0),
                         scale_param=dict(filler=dict(value=0.6666), bias_term=False))
        tanh =  L.TanH(scale1, in_place = True, name = "{}_sTanH".format(name))
        scale2 = L.Scale(tanh, in_place = True, name = "{}_postscale".format(name),
                         param=dict(lr_mult=0, decay_mult=0),
                         scale_param=dict(filler=dict(value=1.7159), bias_term=False))
        # scale2 =  L.TanH(fc, in_place = True, name = "{}_sTanH".format(name))
        return fc, scale2

    elif activ=="softmax":
        # return fc, L.Softmax(fc, in_place=True)
        return fc, L.Softmax(fc, in_place=False)
    else:
        return fc


def dropout(name, bottom, dropout_ratio):
    return L.Dropout(
        bottom,
        in_place = True,
        dropout_param = dict(dropout_ratio = dropout_ratio),
        name = name
    )


def accuracy(name, bottom, labels, top_k):
    return L.Accuracy(
        bottom,
        labels,
        accuracy_param = dict(top_k = top_k),
        # include = dict(phase = caffe_pb2.Phase.Value("TEST")),
        name = name
    )







def NumOfClasses(dataset):
    # set_size = {"rtsd-r1": 67, "rtsd-r3": 106, 'RTSD': 116}
    # return set_size[dataset]
    pass


def ConvPoolAct(n, args):
    n.pool1 = maxpool("pool1", conv1(n, "conv1", n.data, 100, kernel_size = 7, pad = 0, 
                        activ=args.activation, group=1))
    n.pool2 = maxpool("pool2", conv1(n, "conv2", n.pool1, 150, kernel_size = 4, pad = 0, 
                        activ=args.activation, group=args.conv_group))
    n.pool3 = maxpool("pool3", conv1(n, "conv3", n.pool2, 250, kernel_size = 4, pad = 0, 
                        activ=args.activation, group=args.conv_group))



def FcDropAct(n, args, dataset):
    num_of_classes = NumOfClasses(dataset)

    n.fc4_300, n.relu4 = fc("fc4", n.pool3, num_output = 300, activ=args.activation)
    if args.dropout == True:
        n.drop4 = dropout("drop4", n.relu4, dropout_ratio = args.drop_ratio)

    n.fc5_classes, n.softmax = fc("fc5", n.relu4, num_output = num_of_classes, activ="softmax")



#needed for commitee testing
def NoLMDB_Net(args, dataset, mode, phase):
    data_prefix = "../local_data"
    mean_path = '{}/lmdb/{}/{}/{}/mean.txt'.format(data_prefix,dataset, mode, phase)
    # mean_path = '{}/{}/{}/{}/mean.txt'.format(data_prefix,dataset, mode, phase)
    src_path = PrepareSrcFromGT(data_prefix, dataset, mode, phase) 

    n = caffe.NetSpec()
    # DataOnly(n)
    DataOnly(n, 'test', src_path, mean_path)
    ConvPoolAct(n, args)
    FcDropAct(n, args, dataset)
    
    return n.to_proto()



# def make_net(args, dataset, mode, phase):
#     data_prefix = "../local_data"
#     mean_path = '{}/lmdb/{}/{}/{}/mean.txt'.format(data_prefix,dataset, mode, phase)
#     mean_path = '{}/{}/{}/{}/mean.txt'.format(data_prefix,dataset, mode, phase)
#     lmdb = '{}/lmdb/{}/{}/{}/lmdb'.format(data_prefix, dataset, mode, phase)  

#     n = caffe.NetSpec()
#     Data(n, lmdb, phase, args.batch_size, mean_path)
#     ConvPoolAct(n, args)
#     FcDropAct(n, args, dataset)
    
#     n.loss = L.MultinomialLogisticLoss(n.softmax, n.label)
#     n.accuracy_1 = accuracy("accuracy_1", n.fc5_classes, n.label, 1)
#     n.accuracy_5 = accuracy("accuracy_5", n.fc5_classes, n.label, 5)

#     if phase == "train":
#         n.silence = L.Silence(n.accuracy_1, n.accuracy_5, ntop=0)

#     return n.to_proto()



 
def launch():
    parser = gen_parser()
    args = parser.parse_args()
    dataset = 'RTSD'
    for phase in ['train', 'test']:
        print("Generating architectures")
        print("{}  {}  {}".format(dataset,mode, phase))
        directory = '{}/experiment_{}/{}/{}/trial_{}/'.format(
            args.proto_pref,args.EXPERIMENT_NUMBER, dataset,mode, args.trial_number)
        safe_mkdir(directory)
        with open('{}/{}.prototxt'.format(directory, phase), 'w') as f:
            f.write(str(make_net(
                            args=args,
                            dataset=dataset,
                            mode=mode,
                            phase=phase
            )))

        print("")
    GenSingleNetSolver(dataset, mode, args)



# def Data(net, img_path=None, img_size=32, batch_size=512, pad=4):
def Data(net, **kwargs):
    img_size, img_path = kwargs['img_size'], kwargs['img_path']
    batch_size, pad = kwargs['batch_size'], kwargs['pad']

    for phase in ['train', 'val']:
        mean_path = '{img_path}/{phase}/mean.txt'.format(**locals())
        mean = load_image_mean(mean_path)

        if mean is not None:
            transform_param = dict(mirror=False, crop_size=img_size , mean_value = map(int, mean), scale=1.0/255)
        else:
            transform_param = dict(mirror=False, crop_size=img_size , scale=1.0/255)

        if phase == "train":
            PHASE = "TRAIN"
        elif phase == "val":
            PHASE = "TEST"

        image_data_param = dict(
            source = '{img_path}/gt_{phase}.txt'.format(**locals()), 
            batch_size = batch_size,
            new_height = img_size + 2 * pad,
            new_width = img_size + 2 * pad)

        net.data, net.label = L.ImageData(
            image_data_param = image_data_param,
            transform_param = transform_param,
            ntop = 2,
            include = dict(phase = caffe_pb2.Phase.Value(PHASE)),
            name = "data")
     
# def conv_relu(n, name, bottom, kernel_size, num_output, stride=1, pad=1, group=1):
#     conv = L.Convolution(
#         bottom,
#         kernel_size=kernel_size,
#         stride=stride,
#         num_output=num_output,
#         pad=pad,
#         group=group,
#         weight_filler = dict(type = 'xavier'),
#         name = name,
#     )

#     setattr(n, name, conv)

#     return conv, L.ReLU(conv, in_place = True, name = "{}_relu".format(name))


def conv1(net, name, bottom, num_output, kernel_size=3, stride=1, pad=1, group=1):
    conv = L.Convolution(bottom,
        kernel_size=kernel_size,
        stride=stride,
        num_output=num_output,
        pad=pad,
        group=group,
        weight_filler = dict(type = 'xavier'),
        name = name)

    setattr(n, name, conv)

    return conv, L.ReLU(conv, in_place = True, name = "{}_relu".format(name))
    


def wresnet(data_args):
    net = caffe.NetSpec()

    data_args = get_data_args(config)
    Data(net, **data_args)

    conv1(net)
    # conv2(net)
    # conv3(net)
    # conv4(net)
    # avg_pool



def get_data_args(config):
    args = {}
    args['img_path'] = config['img']['processed_path']
    args['img_size'] = config['img']['img_size']
    args['pad'] = config['img']['padding']
    args['batch_size'] = config['train_params']['batch_size']

    return args



def construct(config):
    wrn = wresnet(config)
    pass



from util.config import getConfig
from pprint import pprint

if __name__ == '__main__':
    config = getConfig('./config.json')
    pprint(config)
    construct(config)