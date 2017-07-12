#!/usr/bin/env python
import json
from pprint import pprint
import cv2
import matplotlib.pyplot as plt
from util.util import safe_mkdir
from marking import load_marking




def crop_sign(name, entry):
	img = cv2.imread('{}/imgs/{}'.format(rootpath, name))
	m, n, _ = img.shape
	x, y, w, h = entry['x'], entry['y'], entry['w'], entry['h']

	x1, x2 = max(0,x), min(x + w, n)
	y1, y2 = max(0,y), min(y + h, m)
	cropped = img[y1 : y2, x1 : x2]
	return cropped


def save_gt(gt, phase):
	path = '{}/{}/gt_{}.txt'.format(rootpath, data_folder, phase)
	with open(path, 'w') as f:
		for pair in sorted(gt):
			f.write('{},{}\n'.format(pair[0], pair[1]))


def save_l2n(lab2num):
	path = '{}/{}/numbers_to_classes.txt'.format(rootpath, data_folder)
	with open(path, 'w') as f:
		f.write('class_number,sign_class\n')
		for name in sorted(lab2num):
			f.write('{},{}\n'.format(lab2num[name], name))

import os.path as osp

def get_marking(marking_path, prefix):
	marking = {}

	for phase in ['train', 'test', 'val']:
		path = '{}/{}_{}.json'.format(marking_path, prefix, phase)
		if osp.exists(path):
			marking[phase] = load_marking(path)
		else:
			marking[phase] = None
			print("WARNING: no {} marking file found.".format(phase))
	
	return marking


def class2lab(marking):
	mapping = {}

	for class_id in sorted(marking):
		c2l[class_id] = sorted(marking).index(class_id)

	return mapping
	

# def crop_and_save():
	# for sign_entry in sorted_by_pict:	

def process(img_path, marking_path, prefix, cropped_path):
	marking = get_marking(marking_path, prefix)
	class_mapping = class2lab(marking['train'])

	for phase in sorted(marking):
		sorted_by_pict = sorted(marking[phase], key=lamda x: x['pict_name'])
		# crop_and_save(sorted_by_pict, input_path, output_path, class_mapping)

# def main():
# 	for phase in ["train", "test"]:
# 		mpath = '{}/classification_marking_{}.json'.format(rootpath, phase)
# 		marking = load_marking(mpath)
# 		labels = get_label_set(marking)
# 		lab2num = {key:value for key, value in zip(sorted(labels), range(len(labels)))}
# 		save_l2n(lab2num)
		
# 		gt = []
# 		img_id = 0
# 		for name in sorted(marking):
# 			for sign_entry in marking[name]:
# 				if img_id % rate == 0:
# 					print(phase, img_id)

# 				clid = lab2num[sign_entry['sign_class']]
# 				path = '{}/{}.png'.format(clid, img_id)
# 				gt += [(path, clid)]

# 				img = crop_sign(name, sign_entry)
# 				path = '{}/{}/{}/{}'.format(rootpath, data_folder, phase, clid)
# 				safe_mkdir(path)
# 				cv2.imwrite('{}/{}.png'.format(path, img_id) , img)
				
# 				img_id += 1
# 		save_gt(gt, phase)
		





if __name__ == '__main__':
	print("classification marking -> cropped(from detection images) signs")
	rootpath = '../global_data/Traffic_signs/RTSD'
	data_folder = 'classification'
	rate = 50
	main()
	