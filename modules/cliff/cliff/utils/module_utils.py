'''
 @FileName    : module_utils.py
 @EditTime    : 2022-09-27 14:38:28
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''
import numpy as np
import random
import torch
import pickle
import os
import json

def save_json(out_path, data):
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))
    with open(out_path, 'w') as f:
        json.dump(data, f)

def load_pkl(path):
    """"
    load pkl file
    """
    with open(path, 'rb') as f:
        param = pickle.load(f, encoding='iso-8859-1')
    return param

def seed_worker(worker_seed=7):
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_seed(seed):
    # Set a constant random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    g = torch.Generator()
    g.manual_seed(seed)
    return g
