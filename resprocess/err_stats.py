#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Dec  4 15:10:11 2017

@author: katydagoth
"""

import json
from pprint import pprint
from io import BytesIO
from PIL import Image
from base64 import b64encode
from skimage.io import imread
from skimage.transform import resize, rescale
import numpy as np
from os import listdir


def img_to_base64(img):
    stream = BytesIO()
    Image.fromarray(img).save(stream, format='PNG')
    stream.seek(0)
    return b64encode(stream.getvalue()).decode('utf-8')



def get_sign(filename, bbox):
    full_img = imread("../RTSD/imgs/" + filename)
    x, y, w, h = bbox["x"], bbox["y"], bbox["w"],bbox["h"]
    m, n, _ = full_img.shape

    cropped = full_img[max(y, 0) : min(y+h, m), max(x, 0) : min(x+w, n)]
    resized = (resize(cropped, (48,48,3)) * 255).astype(np.uint8)
    return resized


def pictogram(class_name, directory):
    pict_name = class_name + ".png"
    # directory = "../RTSD/pictograms/"
    if pict_name in listdir(directory):
        return (rescale(imread(directory + pict_name), 0.5) * 255).astype(np.uint8)
    else:
        for pict_name in reversed(sorted(listdir(directory))):
            if pict_name.split('.')[0] in class_name and \
                    ("n" in class_name or "r" in class_name or "3_4_1" in class_name):
                return (rescale(imread(directory + pict_name), 0.5) * 255).astype(np.uint8)
        return np.ones((32, 32, 3)).astype(np.uint8) * 255


def get_accuracies(acc_path, mapping_path):
    accuracies = {}
    with open(mapping_path) as f_map:
        lab2class = json.load(f_map)
        with open(acc_path) as f_acc:
            for acc in f_acc.readlines()[1:]:
                acc = " ".join(acc.split())
                class_id = acc.split(" ")[0]
                err_percent = float(acc.split(" ")[1])

                class_name = lab2class[class_id]
                accuracies[class_name] = err_percent
    return accuracies


def get_class_sizes(stats_path, phase):
    sizes = {}
    with open(stats_path) as f:
        for stat in f.readlines()[1:]:
            stat = " ".join(stat.split())
            class_name = stat.split(" ")[0]

            if phase == "train":
                size = int(stat.split(" ")[1])
            else:
                size = int(stat.split(" ")[3])

            sizes[class_name] = size
    return  sizes

def get_err_stats(group, exp , phase):
    acc_path = "./Experiments/group_{group}/exp_{exp}/class_acc_{phase}.txt".format(**locals())
    mapping_path = "../global_data/Traffic_signs/RTSD/classification/labels-to-classes.json"
    stats_path = "../global_data/Traffic_signs/RTSD/class_stats.txt"

    acc = get_accuracies(acc_path, mapping_path)
    class_size = get_class_sizes(stats_path, phase)
    err_stats = {}
    for class_name in sorted(acc):
        err_stats.setdefault(class_name, {})["err"] = 1.0 - acc[class_name]
        err_stats[class_name][ "class_size"] = class_size[class_name]
    return  err_stats








style = """<style type="text/css">
   BODY {
    background: white; /* Цвет фона веб-страницы */
   }
   TABLE {
    width: 300px; /* Ширина таблицы */
    border-collapse: collapse; /* Убираем двойные линии между ячейками */
    border: 2px solid; /* Прячем рамку вокруг таблицы */
   }
   TD, TH {
    padding: 3px; /* Поля вокруг содержимого таблицы */
    border: 1px solid maroon; /* Параметры рамки */
    text-align: left; /* Выравнивание по левому краю */
   }
  </style>
"""


def write_head(fhandle):
    print('<html><body><table>', file=fhandle)
    print(style, file=fhandle)

    row = '<tr><td>Class name</td><td>Pict</td><td>Error(%)</td><td>Class size</td></tr>'
    print(row, file=fhandle)

import operator
def write_err_stats(err_stats, save_path):
    with open(save_path, 'w') as fhandle:
        print(save_path)
        write_head(fhandle)

        for stat in sorted(err_stats.items(), key=lambda x: x[1]["err"]):
            class_name = stat[0]
            err = stat[1]["err"]
            size = stat[1]["class_size"]
            # img = img_to_base64(get_sign(filename, bbox))
            pict_dir = "../global_data/Traffic_signs/RTSD/pictograms/"
            print('<tr>', file=fhandle)
            print('<td>{}</td>'.format(class_name), file=fhandle)
            print('<td><img src="data:image/png;base64,{}"></img></td>'.format(
                img_to_base64(pictogram(class_name, pict_dir))), file=fhandle)
            # print('<td><img src="data:image/png;base64,{}"></img></td>'.format(img), file=fhandle)
            print('<td>{}</td>'.format(err), file=fhandle)
            print('<td>{}</td>'.format(size), file=fhandle)
            print('</tr>', file=fhandle)

        print('</table></body></html>', file=fhandle)






if __name__ == '__main__':
    group = 8
    exp = 0
    phase = "test"
    print("in main")
    stats = get_err_stats(group, exp, phase)
    save_path = "./Experiments/group_{group}/exp_{exp}/err_stats_{phase}.html".format(**locals())
    write_err_stats(stats, save_path)





