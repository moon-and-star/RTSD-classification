#!/usr/bin/env python

import os
from util.util import safe_mkdir

for phase in ['val', 'train', 'test']:
	tools = '/opt/caffe/build/tools'
	prefix = "./local_data/RTSD"
	safe_mkdir('{prefix}/lmdb/{phase}'.format(**locals()))
	os.system("GLOG_logtostderr=1 {tools}/convert_imageset -shuffle -backend lmdb\
	 {prefix}/{phase} {prefix}/gt_{phase}.txt  {prefix}/lmdb/{phase}/lmdb".format(**locals()))

# GLOG_logtostderr=1 $TOOLS/convert_imageset -shuffle -backend lmdb  \
# 			     ${prefix}/${i}/${j}/${k}/          \
# 			     ${prefix}/${i}/${j}/gt_${k}.txt    \
# 			     ${prefix}/lmdb/${i}/${j}/${k}/lmdb