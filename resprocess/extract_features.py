#!/usr/bin/env python
import sys
import pathlib2 as pathlib
absolute_path = pathlib.Path('./').resolve()
sys.path.append(str(absolute_path))

# def load_model(config):
#     print('gpu_num = ', config['train_params']['gpu_num'])
#
#     if config['model'] == 'wresnet':
#         from .wresnet import prepare_model
#     elif config["model"] == 'densenet':
#         from .densenet import prepare_model
#     elif config["model"] == 'MCDNN':
#         from .mcdnn import prepare_model
#     else:
#         print("Unknown model ", config["model"])
#
#     print('preparing model')
#     model = prepare_model(config)
#     model.load_weights(best_checkpoint(config))



from util.config import get_config, load_exp_config
from netprocess.keras_scripts.test import  load_model

if __name__ == "__main__":
    confpath = "./config.json"
    config = get_config(confpath)

    group_num = 7
    exp_num = 0
    config = load_exp_config(config, group_num, exp_num)
    model = load_model(config)

    print(model.summary())