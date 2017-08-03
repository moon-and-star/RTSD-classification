#!/usr/bin/env python

import sys
home = '/home/GRAPHICS2/20e_ame'
sys.path.append("{home}/anaconda2/pkgs/pathlib2-2.2.1-py27_0/lib/python2.7/site-packages/".format(**locals()))
sys.path.append("{home}/anaconda2/pkgs/scandir-1.4-py27_0/lib/python2.7/site-packages/".format(**locals()))

import pathlib2 as pathlib
absolute_path = pathlib.Path('./').resolve()
sys.path.append(str(absolute_path))



import os
from util.util import safe_mkdir




for phase in ['val', 'train', 'test']:
	tools = '/opt/caffe/build/tools'
	prefix = "./local_data/RTSD"
	safe_mkdir('{prefix}/lmdb/{phase}'.format(**locals()))
	os.system("GLOG_logtostderr=1 {tools}/convert_imageset -shuffle -backend lmdb\
	 {prefix}/{phase}/ {prefix}/gt_{phase}.txt  {prefix}/lmdb/{phase}/lmdb".format(**locals()))

# GLOG_logtostderr=1 $TOOLS/convert_imageset -shuffle -backend lmdb  \
# 			     ${prefix}/${i}/${j}/${k}/          \
# 			     ${prefix}/${i}/${j}/gt_${k}.txt    \
# 			     ${prefix}/lmdb/${i}/${j}/${k}/lmdb