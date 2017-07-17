#!/usr/bin/env python

for phase in ['train', 'val', 'test']:
	with open('./local_data/RTSD/gt_{}.txt'.format(phase)) as f:
		lines = f.readlines()

	with open('./local_data/RTSD/gt_{}.txt'.format(phase), 'w') as f:
		for line in lines:
			f.write('./local_data/RTSD/{phase}/{line}'.format(**locals()))