#!/usr/bin/env python
import json
from pprint import pprint
from io import BytesIO
from PIL import Image
from base64 import b64encode

from skimage.io import imread
from skimage.transform import resize, rescale
import numpy as np
from marking import load_marking, organize_by_classes


def get_sign(filename, bbox):
    full_img = imread("../global_data/Traffic_signs/RTSD/imgs/" + filename)
    x, y, w, h = bbox["x"], bbox["y"], bbox["w"],bbox["h"]
    m, n, _ = full_img.shape

    cropped = full_img[max(y, 0) : min(y+h, m), max(x, 0) : min(x+w, n)]
    resized = (resize(cropped, (48,48,3)) * 255).astype(np.uint8)
    return resized


def img_to_base64(img):
    stream = BytesIO()
    Image.fromarray(img).save(stream, format='PNG')
    stream.seek(0)
    return b64encode(stream.getvalue()).decode('utf-8')


def get_stats(prefix):
    path = '../global_data/Traffic_signs/RTSD'
    stats = {}
    for phase in ['train', 'test']:
        filename = "{path}/{prefix}_{phase}.json".format(**locals())
        with open(filename) as f:
            marking = json.load(f)

        for class_name in marking:
            stats.setdefault(class_name, {})
            stats[class_name].setdefault(phase, {})
            stats[class_name][phase]['imgs'] = len(marking[class_name])
            if phase == 'test':
                stats[class_name]['filename'] = marking[class_name][0]['pict_name']
                stats[class_name]['bbox'] = marking[class_name][0]
                # print(marking[class_name][0]['pict_name'])

            id_set = set()
            for image in marking[class_name]:
                id_set.add(image['sign_id'])
            stats[class_name][phase]['phys'] = len(id_set)
    return stats


def print_stats(stats):
    content = "{:10}{:>15}{:>15}{:>15}{:>15}".format(
        'class', 'train_imgs', 'train_phys', 'test_imgs', 'test_phys')
    print(len(stats))
    print(content)

    for class_ in sorted(stats.items(), key=lambda x: x[1]['train']['imgs'], reverse=True):
        class_name = class_[0]
        train_imgs = stats[class_name]['train']['imgs']
        test_imgs = stats[class_name]['test']['imgs']
        train_phys = stats[class_name]['train']['phys']
        test_phys = stats[class_name]['test']['phys']

        content = "{class_name:10}{train_imgs:15}{train_phys:15}{test_imgs:15}{test_phys:15}".format(**locals())
        print(content)




from os import listdir

def pictogram(class_name):
    pict_name = class_name + ".png"
    directory = "../global_data/Traffic_signs/RTSD/pictograms/"
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


def write_header(fhandle):
    print('<html><body><table>', file=fhandle)
    print(style, file=fhandle)
    header = '<tr><td>Pictogram</td><td>Sample</td><td>Class name</td><td>train images</td>\
             <td>train physical</td><td>test images</td><td>test physical</td></tr>'
    print(header, file=fhandle)

def write_marking_stats(class_stats, prefix='', write_path=''):
    with open("{write_path}/{prefix}_stats.html".format(**locals()), 'w') as fhandle:
        write_header(fhandle)

        names = sorted(class_stats, reverse=True, key=lambda x: (
            class_stats[x]['train']['imgs'], class_stats[x]['train']['phys'],
            class_stats[x]['test']['imgs'], class_stats[x]['test']['phys']))
        for class_name in names:
            train_imgs = class_stats[class_name]['train']['imgs']
            train_phys = class_stats[class_name]['train']["phys"]
            test_imgs = class_stats[class_name]['test']['imgs']
            test_phys = class_stats[class_name]['test']["phys"]
            filename = class_stats[class_name]["filename"]
            bbox = class_stats[class_name]["bbox"]
            img = img_to_base64(get_sign(filename, bbox))

            print('<tr>', file=fhandle)
            print('<td><img src="data:image/png;base64,{}"></img></td>'.format(
                img_to_base64(pictogram(class_name))), file=fhandle)
            print('<td><img src="data:image/png;base64,{}"></img></td>'.format(img), file=fhandle)
            print('<td>{}</td>'.format(class_name), file=fhandle)
            print('<td>{}</td>'.format(train_imgs), file=fhandle)
            print('<td>{}</td>'.format(train_phys), file=fhandle)
            print('<td>{}</td>'.format(test_imgs), file=fhandle)
            print('<td>{}</td>'.format(test_phys), file=fhandle)
            print('</tr>', file=fhandle)

        print('</table></body></html>', file=fhandle)



def extend_stats(stats):
    for class_name in stats:
        for phase in ['train', 'test']:
            if phase not in stats[class_name]:
                stats[class_name][phase] = {}
                stats[class_name][phase]['imgs'] = 0
                stats[class_name][phase]['phys'] = 0


def get_time_marking_stats(path, prefix):

    stats = {}
    for phase in ['train', 'test']:
        phase_marking = load_marking("{path}/{prefix}_{phase}.json".format(**locals()))
        marking = organize_by_classes(phase_marking)

        for class_name in marking:
            stats.setdefault(class_name, {})
            stats[class_name].setdefault(phase, {})
            stats[class_name][phase]['imgs'] = len(marking[class_name])
            stats[class_name]['filename'] = marking[class_name][0]['pict_name']
            stats[class_name]['bbox'] = marking[class_name][0]

            id_set = set()
            for image in marking[class_name]:
                id_set.add(image['sign_id'])
            stats[class_name][phase]['phys'] = len(id_set)

    extend_stats(stats)
    return stats

# autosave23_10_2012_10_01_14_0.jpg
#the first filename in old test marking (sorted)

def write_time_marking(marking, path, prefix):
    ind = sorted(marking).index("autosave23_10_2012_10_01_14_0.jpg")
    train = dict(sorted(marking.items())[:ind + 1])
    test = dict(sorted(marking.items())[ind + 1:])

    with open("{path}/{prefix}_train.json".format(**locals()), 'w') as out:
        json.dump(train, out, indent=4,sort_keys=True)

    with open("{path}/{prefix}_test.json".format(**locals()), 'w') as out:
        json.dump(test,out, indent=4,sort_keys=True)



if __name__ == '__main__':
    stats = get_stats("marking")
    print_stats(stats)
    write_marking_stats(stats, prefix='split', write_path='./')


    stats = get_stats('classmarking')
    print_stats(stats)
    write_marking_stats(stats, prefix='class_split', write_path='./')

    prefix = 'time_marking'
    path = '../global_data/Traffic_signs/RTSD/'
    raw_marking = load_marking('{}/new_marking.json'.format(path))
    write_time_marking(raw_marking, path, prefix)

    time_stats = get_time_marking_stats(path, prefix)
    print_stats(time_stats)
    write_marking_stats(time_stats, prefix='time_split', write_path='./')

    # with open("{}/time_marking_test.json".format(path)) as f:
    #     test = json.load(f)
    #     print(sorted(test)[0])

