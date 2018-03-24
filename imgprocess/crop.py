# !/usr/bin/env python
import json
import cv2
from util.util import safe_mkdir



def crop_sign(path, entry):
	name = entry['pict_name']
	img = cv2.imread('{}/{}'.format(path, name))
	m, n, _ = img.shape
	x, y, w, h = entry['x'], entry['y'], entry['w'], entry['h']

	x1, x2 = max(0,x), min(x + w, n)
	y1, y2 = max(0,y), min(y + h, m)
	cropped = img[y1 : y2, x1 : x2]

	return cropped


def save_gt(gt, path, phase):
	path = '{}/gt_{}.txt'.format(path, phase)
	with open(path, 'w') as f:
		for pair in sorted(gt):
			f.write('{},{}\n'.format(pair[0], pair[1]))


def save_l2n(lab2num):
	path = '{}/{}/numbers_to_classes.txt'.format(rootpath, data_folder)
	with open(path, 'w') as f:
		f.write('class_number,sign_class\n')
		for name in sorted(lab2num):
			f.write('{},{}\n'.format(lab2num[name], name))


def class2lab(marking, path=None): # marking without phase 
	mapping = {}

	for class_id in marking:
		mapping[class_id] = sorted(marking).index(class_id)

	if path != None:
		with open(path, 'w') as f:
			json.dump(mapping, f, indent=4)
		print('classes-to-labels mapping saved')

	return mapping


def lab2class(marking, path=None):
	mapping = {y:x for (x,y) in class2lab(marking).items()}

	if path != None:
		with open(path, 'w') as f:
			json.dump(mapping, f, indent=4)

	return mapping
	

def flat_marking(marking): #marking without separating entries by class id
	new_marking = {}

	for phase in marking:
		for class_id in marking[phase]:
			new_marking.setdefault(phase, []).extend(marking[phase][class_id])

	return new_marking


def sign2id(marking, path=None):
	print('')
	marking = flat_marking(marking)
	mapping = {}

	for phase in marking:
		sorted_by_pict = sorted(marking[phase], key=lambda x: x['pict_name'])

		for i, sign in enumerate(sorted_by_pict):
			name = "{}.png".format(str(i).zfill(6))
			mapping.setdefault(phase, {})[name] = sign

		if path != None:
			with open(path.format(phase), 'w') as f:
				json.dump(mapping, f, indent=4)
			print('sign-to-id mapping for {} saved'.format(phase))

	return mapping





def crop_and_save(sign_mapping, class_mapping, input_path='', output_path=''):
	rate = 100
	for phase in sign_mapping:
		gt = []
		count = 0
		print('\ncropping signs: {}'.format(phase))
		for sign_name in sorted(sign_mapping[phase]):
			if count % rate == 0:
				print(sign_name)
			count += 1
				
			sign_entry = sign_mapping[phase][sign_name]
			cropped = crop_sign(input_path, sign_entry)

			label = class_mapping[sign_entry['sign_class']]
			directory = '{output_path}/{phase}/{label}'.format(**locals())
			safe_mkdir(directory)
			full_path = '{directory}/{sign_name}'.format(**locals())
			cv2.imwrite(full_path, cropped)

			short_path = '{label}/{sign_name}'.format(**locals())
			gt += [(short_path, label)]
		save_gt(gt, output_path, phase)



def marking2cropped(marking=None, img_path='', cropped_path=''):
	print('\nmarking -> cropped')
	safe_mkdir(cropped_path)
	class2lab_mapping = class2lab(marking['train'], cropped_path + '/classes-to-labels.json')
	lab2class_mapping = lab2class(marking['train'], cropped_path + '/labels-to-classes.json')
	sign_mapping = sign2id(marking, cropped_path + '/id-to-sign_{}.json') #note: {} is for phase insertion (see sign2id definition)

	crop_and_save(sign_mapping, class2lab_mapping, input_path=img_path, output_path=cropped_path)

