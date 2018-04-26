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
        width: 60px;\n\
        float: left;\n\
        margin: 0;\n\
        text-align: center;\n\
        padding: 5;\n\
    }\n\
    img {\n\
        max-height: 100%;\n\
        max-width: 100%;\n\
        min-width: 100%;\n\
    }\n\
    td {\n\
        width: 50%;\n\
        border-collapse: collapse;\n\
    }\n\
    table {\n\
        min-width: 100%;\n\
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
    img {\n\
        max-height: 100%; \n\
        max-width: 100%; \n\
    }\n\
    .actual {\n\
        border: 1px solid #00aa00;\n\
    }\n\
    .predicted {\n\
        border: 1px solid rgb(204, 16, 44);\n\
    }\n\
</style>', file=f)


def img_to_base64(img):
    stream = BytesIO()
    Image.fromarray(img).save(stream, format='PNG')
    stream.seek(0)
    return b64encode(stream.getvalue()).decode('utf-8')


def get_class_members(img_dir):
    dirlist = sorted([d for d in listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))])

    members = {}
    for class_dir in dirlist:
        filename = sorted(listdir('{img_dir}/{class_dir}'.format(**locals())))[-1]
        img = imread('{img_dir}/{class_dir}/{filename}'.format(**locals()))
        b64 = img_to_base64(img)
        members[class_dir] = b64

    return members




# act = filename.split('_')[1]
# pred = filename.split('_')[3]
# print("<figcaption>\
#                 <font size=\"2\" color=\"#44aa44\">{act}</font>\
#                 <font size=\"2\" color=\"#dd3333\">{pred}</font>\
#                 <font size=\"1\" color=\"#000000\">{filename}</font>\
#        </figcaption>".format(**locals()), file=fhandle)

def write_caption(actual, predicted, fhandle):
    print('<figcaption>', file=fhandle)
    print('<table>', file=fhandle)
    print('<tr>', file=fhandle)

    print('<td class="actual">', file=fhandle)
    print('<img src="data:image/png;base64,%s" ' % actual, file=fhandle)
    print('</td>', file=fhandle)

    print('<td class="predicted">', file=fhandle)
    print('<img src="data:image/png;base64,%s" ' % predicted, file=fhandle)
    print('</td>', file=fhandle)

    print('</tr>', file=fhandle)
    print('</table>', file=fhandle)
    print('</figcaption>', file=fhandle)


def write_figure(img64, actual, predicted, fhandle):
    print('<figure>', file=fhandle)
    print('<img src="data:image/png;base64,%s" alt="ololo">' % img64, file=fhandle)
    write_caption(actual, predicted, fhandle)
    print('</figure>', file=fhandle)



def write_misclassified(img_dir, correct_dir,fhandle):
    print('<block>', file=fhandle)

    members = get_class_members(correct_dir)
    for class_dir in sorted(listdir(img_dir)):
        for filename in sorted(listdir('{img_dir}/{class_dir}'.format(**locals()))):
            img = imread('{img_dir}/{class_dir}/{filename}'.format(**locals()))
            img64 = img_to_base64(img)
            actual = filename.split('_')[1]
            predicted = filename.split('_')[3]
            write_figure(img64, members[actual], members[predicted], fhandle)

    print('</block>', file=fhandle)



def write_correct(correct_dir, fhandle):
    print('<block>', file=fhandle)

    members = get_class_members(correct_dir)
    for class_dir in members:
        b64 = members[class_dir]
    
        print('<figure>', file=fhandle)
        print('<img src="data:image/png;base64,%s">' % b64, file=fhandle)
        print('<figcaption>\
            <font size="1">{class_dir}</font>\
            </figcaption>'.format(**locals()), file=fhandle)
        print('</figure>', file=fhandle)

    print('</block>', file=fhandle)



if __name__ == '__main__':
    phase = 'test'
    group = 7
    train_img_dir = './local_data/RTSD/train'
    for exp_num in range(2):
        for phase in ["train", "test"]:
            print("exp_num = ", exp_num)
            exp_dir = './Experiments/group_{group}/exp_{exp_num}'.format(**locals())

            img_dir = '{exp_dir}/misclassified/{phase}/predicted/'.format(**locals())
            with open('{exp_dir}/collage_{phase}.html'.format(**locals()), 'w') as fhandle:
                write_style(fhandle)
                write_misclassified(img_dir, train_img_dir, fhandle)
                write_correct(train_img_dir, fhandle)

        
