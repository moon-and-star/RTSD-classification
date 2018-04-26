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



def img_to_base64(img):
    stream = BytesIO()
    Image.fromarray(img).save(stream, format='PNG')
    stream.seek(0)
    return b64encode(stream.getvalue()).decode('utf-8')



def get_class_stats():
    with open("../RTSD/new_marking.json") as f:
        marking = json.load(f)
        class_stats = {}
        for filename in marking:
            for bbox in marking[filename]:
                class_id = bbox["sign_class"]
                sign_id = bbox["sign_id"]

                default = {"total": 0, "phys": 0, "id": set(), "bbox": {}, "filename": ""}
                class_stats.setdefault(class_id, default)

                # if class_stats[class_id]["bbox"] == {}:
                class_stats[class_id]["bbox"] = bbox
                class_stats[class_id]["filename"] = filename

                class_stats[class_id]["total"] += 1
                class_stats[class_id]["id"].add(sign_id)
                class_stats[class_id]["phys"] = len(class_stats[class_id]["id"])


        name = sorted(class_stats)[1]
        pprint(class_stats[name])

    return class_stats


from skimage.io import imread
from skimage.transform import resize, rescale
import numpy as np

def get_sign(filename, bbox):
    full_img = imread("../RTSD/imgs/" + filename)
    x, y, w, h = bbox["x"], bbox["y"], bbox["w"],bbox["h"]
    m, n, _ = full_img.shape

    cropped = full_img[max(y, 0) : min(y+h, m), max(x, 0) : min(x+w, n)]
    resized = (resize(cropped, (48,48,3)) * 255).astype(np.uint8)
    return resized


from os import listdir

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

import operator

def write_class_stats(class_stats):
    with open("../RTSD/stats(sorted_by_name).html", 'w') as fhandle:
        print('<html><body><table>', file=fhandle)
        print(style, file=fhandle)
        row = '<tr><td>Пиктограмма</td><td>Пример</td><td>Имя класса</td>\
            <td>Количество изображений знаков</td><td>Количество физических знаков</td></tr>'
        print(row, file=fhandle)

        # for class_name in sorted(class_stats,  key=lambda x: class_stats[x]["total"], reverse=True):
        for class_name in sorted(class_stats):
            total = class_stats[class_name]['total']
            phys = class_stats[class_name]["phys"]
            filename = class_stats[class_name]["filename"]
            bbox = class_stats[class_name]["bbox"]
            img = img_to_base64(get_sign(filename, bbox))

            print('<tr>', file=fhandle)
            print('<td><img src="data:image/png;base64,{}"></img></td>'.format(img_to_base64(pictogram(class_name))), file=fhandle)
            print('<td><img src="data:image/png;base64,{}"></img></td>'.format(img), file=fhandle)
            print('<td>{}</td>'.format(class_name), file=fhandle)
            print('<td>{}</td>'.format(total), file=fhandle)
            print('<td>{}</td>'.format(phys), file=fhandle)
            print('</tr>', file=fhandle)

            row = '<tr><td><img src="data:image/png;base64,{}"></img>\
                <td>{}</td></td><td>{}</td><td>{}</td></tr>'.format(img,class_name, total, phys)
            # print(row, file=fhandle)
        print('</table></body></html>', file=fhandle)

#
# def save_stats(stats):
#     with open("")


if __name__ == '__main__':
    stats = get_class_stats()
    write_class_stats(stats)
    # save_stats(stats, "./")





