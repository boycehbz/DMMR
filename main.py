'''
 @FileName: main.py
 @EditTime: 2021-07-07 15:04:44
 @Author  : Buzhen Huang
 @Email   : hbz@seu.edu.cn
'''
import sys
import time
import torch
import os
from core.cmd_parser import parse_config
from core.init import init, load_model
from core.utils.module_utils import save_results, load_camera
from core.utils.init_guess import init_guess, load_init, fix_params
from core.utils.non_linear_solver import non_linear_solver


def main(**args):

    dataset_obj, setting = init(**args)

    start = time.time()

    results = {}
    seq_count = -1
    for idx, data in enumerate(dataset_obj):
        if data['count'] < 1:
            continue
        print('Processing: {}'.format(data['img_path'][0][0]))
        
        setting = load_model(dataset_obj, setting, **args)
        setting = load_camera(data, setting, **args)

        # init guess
        # if seq_count != dataset_obj.serial_cnt:
        seq_count += 1
        setting['seq_start'] = True
        data = init_guess(setting, data, dataset_obj, frames_seq=dataset_obj.frames, use_torso=True, **args)

        fix_params(setting, scale=setting['fixed_scale'], shape=setting['fixed_shape'])
        # linear solve
        print("linear solve, to do...")
        # non-linear solve
        results = non_linear_solver(setting, data, dataset_obj, **args)
        # save results
        save_results(setting, data, results, dataset_obj, frames_seq=dataset_obj.frames, **args)

    elapsed = time.time() - start
    time_msg = time.strftime('%H hours, %M minutes, %S seconds',
                             time.gmtime(elapsed))
    print('Processing the data took: {}'.format(time_msg))

if __name__ == "__main__":
    torch.backends.cudnn.enabled=False
    sys.argv = ["", "--config=cfg_files\\fit_smpl.yaml"
    ]
    args = parse_config()
    main(**args)





