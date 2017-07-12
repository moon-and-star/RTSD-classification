#!/usr/bin/env python
from pprint import pprint
from imgprocess.preprocess import process
from util.config import getConfig
from marking import classification_marking, save_marking


def crop2proc(args):
    config = getConfig(args.confpath)
    img = config['img']
    rootpath = img['cropped_path']
    outpath = img["processed_path"]

    for phase in ["train", "test"]:
        process(rootpath, outpath, phase, border=img['border'])


def process_marking(args):
    config = getConfig(args.confpath)
    img = config['img']
    path = img['marking_path']
    marking = classification_marking(
                    path, 
                    seed=config['randseed'],
                    threshold=img['min_class_size'],
                    test_ratio=img['test_ratio'],
                    val_ratio=img['val_ratio'])
    save_marking(marking, path, img['classmark_prefix'])


def uncut2cropped(args): #full images to classification set
    process_marking(args)
    # crop_and_save()
    pass


def improc(args):
    if args.uncut:
        uncut2cropped(args)
    if args.cropped:
        crop2proc(args)
    


def setupImprocParser(subparsers):
    improc_parser = subparsers.add_parser("improc", help='- Performs image processing \
                        (crops signs from large images by given marking)')
    improc_parser.add_argument('-u','--uncut',action="store_true", 
                        help='processing of uncut images which contain one or more traffic sign on it.')
    improc_parser.add_argument('-c','--cropped',action="store_true", 
                        help='makes cropped images fit the shape specified in config.')

    improc_parser.set_defaults(func=improc)

