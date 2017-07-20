#!/usr/bin/env python

import json
from pprint import pprint

def initial_config():
	config = {}
	config['randseed'] = 42

	im = {}
	im['uncut_path'] = '../global_data/Traffic_signs/RTSD/imgs'
	im['marking_path'] = '../global_data/Traffic_signs/RTSD'
	im['classmark_prefix'] = 'classmarking'
	im['cropped_path'] = '../global_data/Traffic_signs/RTSD/classification'
	im['processed_path'] = './local_data/RTSD'
	im['img_size'] = 32
	im['padding'] = 4
	im['border'] = 'replicate' # black, grey; replicate
	im['min_class_size'] = 300
	im['test_ratio'] = 0.2
	im['val_ratio'] = 0.1
	config['img'] = im

	exp = {}
	exp['exp_path'] = './Experiments'
	exp['group'] = None
	exp['group_description'] = ''
	exp['exp_num'] = None
	exp['exp_description'] = ''
	config['curr_exp'] = exp
	# config['exp_list'] = [[0]] #group 0 contains only 1 experiment - 0 
	config['exp_list'] = []


	wrn = {}
	wrn['width'] = 2
	wrn['metablock_depth'] = 2 #number of blocks in one block of blocks (watch article)
	config['wide_resnet'] = wrn


	tp = {}
	tp['batch_size'] = 128 #
	tp['epoch_num'] = 400 #
	tp['test_frequency'] = 1 #
	tp['lr_step'] = 80 #
	tp['snap_epoch'] = 80
	tp['gpu_num'] = 0
	config['train_params'] = tp

	sp = {}
	sp['base_lr'] = 0.1
	sp['type'] = 'SGD'
	sp['momentum'] = 0.9
	sp['weight_decay'] = 0.0005
	sp['lr_policy'] = 'step'
	sp['gamma'] = 0.2
	sp['solver_mode'] = 'CPU' 
	config['solver_exact'] = sp



	return config
	



def get_config(path):
	with open(path) as f:
		return json.load(f)
		

def set_config(path, config):
	with open(path, 'w') as out:
		json.dump(config, out, sort_keys=True, indent=4) 
	

def init_config(path):
	config = initial_config()
	set_config(path, config)

	print('\nConfiguration file path: {}\n'.format(path) )
	s = json.dumps(get_config(path), indent=4, sort_keys=True)
	print(s)

	return config



if __name__ == '__main__':
	initConfig('./config.json')