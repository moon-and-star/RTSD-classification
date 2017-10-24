#!/usr/bin/env python

import numpy as np
from itertools import islice
import numpy as np
from skimage.io import imsave, imread


from io import BytesIO
from PIL import Image
from base64 import b64encode
from os import listdir
import os

def write_style(f):
    print('\
<style type="text/css">\n \
    figure {\n\
        display: inline-block; \n\
        width: 40px;\n\
        float: left;\n\
        margin: 0;\n\
        text-align: center;\n\
        padding: 0;\n\
    }\n\
    figcaption {\n\
        vertical-align: top;\n\
        margin-top: 0;\n\
        margin-bottom: 5;\n\
    }\n\
    block {\n\
        display: inline-block;\n\
        margin-bottom: 30;\n\
    }\n\
</style>', file=f)


def img_to_base64(img):
    stream = BytesIO()
    Image.fromarray(img).save(stream, format='PNG')
    stream.seek(0)
    return b64encode(stream.getvalue()).decode('utf-8')


def write_misclassified(img_dir, fhandle):
    print('<block>', file=fhandle)
    for class_dir in sorted(listdir(img_dir)):
        for filename in sorted(listdir('{img_dir}/{class_dir}'.format(**locals()))):
            img = imread('{img_dir}/{class_dir}/{filename}'.format(**locals()))
            print(filename)
            b64 = img_to_base64(img)

        
            act = filename.split('_')[1]
            pred = filename.split('_')[3]
            print('<figure>', file=fhandle)
            print('<img src="data:image/png;base64,%s">' % b64, file=fhandle)
            print('<figcaption>\
                <font size="1" color="#44aa44">{act} </font><font size="1" color="#dd3333">{pred}</font>\
                </figcaption>'.format(**locals()), file=fhandle)
            print('</figure>', file=fhandle)
    print('</block>', file=fhandle)



def write_correct(img_dir, fhandle):
    print('<block>', file=fhandle)
    dirlist = sorted([d for d in listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))])

    for class_dir in dirlist:
        filename = sorted(listdir('{img_dir}/{class_dir}'.format(**locals())))[-1]
        img = imread('{img_dir}/{class_dir}/{filename}'.format(**locals()))
        print(filename)
        b64 = img_to_base64(img)

    
        print('<figure>', file=fhandle)
        print('<img src="data:image/png;base64,%s">' % b64, file=fhandle)
        print('<figcaption>\
            <font size="1">{class_dir}</font>\
            </figcaption>'.format(**locals()), file=fhandle)
        print('</figure>', file=fhandle)

    print('</block>', file=fhandle)



if __name__ == '__main__':
    phase = 'test'
    group = 5
    train_img_dir = './local_data/RTSD/train'
    for exp_num in range(19): 
        exp_dir = './Experiments/group_{group}/exp_{exp_num}'.format(**locals())

        img_dir = '{exp_dir}/misclassified/{phase}/predicted/'.format(**locals())
        with open('{exp_dir}/collage.html'.format(**locals()), 'w') as fhandle:
            write_style(fhandle)
            write_misclassified(img_dir, fhandle)
            write_correct(train_img_dir, fhandle)

        