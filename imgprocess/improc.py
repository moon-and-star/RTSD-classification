#!/usr/bin/env python

from .preprocess import process
from util.config import get_config
from imgprocess.marking import classification_marking, save_marking
from .crop import marking2cropped
from imgprocess.marking import load_classification_marking
from shutil import rmtree


def crop2proc(args):
    config = get_config(args.confpath)
    img = config['img']
    rootpath = img['cropped_path']
    outpath = img['processed_path']
    rmtree(outpath, ignore_errors=True)
    rmtree(outpath, ignore_errors=True)
    print("Removed old preprocecessed signs")


    phases = ['train']
    if config['img']['test_ratio'] > 0:
        phases.append('test')

    if config['img']['val_ratio'] > 0:
        phases.append('val')

    for phase in phases:
        process(rootpath, outpath, phase, 
            border=img['border'], 
            pad=img['padding'],
            size=img['img_size'])



def process_marking(args):
    config = get_config(args.confpath)
    img = config['img']
    path = img['marking_path']

    groupped = False
    if 'groupped' in img:
        groupped = img['groupped']


    marking = classification_marking(
                    path, 
                    seed=config['randseed'],
                    group=groupped,
                    # threshold=img['min_class_size'], #old parameter
                    test_ratio=img['test_ratio'],
                    val_ratio=img['val_ratio'])
    save_marking(marking, path, img['classmark_prefix'])



def process_imgs(args):
    img = get_config(args.confpath)['img']
    marking = load_classification_marking(img['marking_path'], img['classmark_prefix'])
    img_path, cropped_path = img['uncut_path'], img['cropped_path']
    rmtree(cropped_path, ignore_errors=True)
    print("Removed old cropped images")

    marking2cropped(marking=marking, img_path=img_path, cropped_path=cropped_path)



def uncut2cropped(args): #full images to classification set
    process_marking(args)
    process_imgs(args)
    


def improc(args):
    if args.uncut:
        uncut2cropped(args)
    if args.cropped:
        crop2proc(args)
    


def setupImprocParser(subparsers):
    improc_parser = subparsers.add_parser('improc', help='- Performs image processing \
                        (crops signs from large images by given marking)')
    improc_parser.add_argument('-u','--uncut',action='store_true', 
                        help='processing of uncut images which contain one or more traffic sign on it.')
    improc_parser.add_argument('-c','--cropped',action='store_true', 
                        help='makes cropped images fit the shape specified in config.')

    improc_parser.set_defaults(func=improc)

