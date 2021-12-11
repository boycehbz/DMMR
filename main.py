'''
 @FileName: main.py
 @EditTime: 2021-07-07 15:04:44
 @Author  : Buzhen Huang
 @Email   : hbz@seu.edu.cn
'''
import sys
import time
import torch
from core.cmd_parser import parse_config
from core.init import init, load_model
from core.utils.module_utils import save_results, load_camera
from core.utils.init_guess import init_guess, fix_params
from core.utils.non_linear_solver import non_linear_solver
torch.backends.cudnn.enabled = False

def main(**args):

    dataset_obj, setting = init(**args)

    start = time.time()

    results = {}
    for idx, data in enumerate(dataset_obj):
        if data['count'] < 1: # empty files
            continue
        print('Processing: {}'.format(data['img_path'][0][0]))
        
        # Load SMPL models
        setting = load_model(dataset_obj, setting, **args)

        # Load initial cameras
        setting = load_camera(data, setting, **args)

        # Initialize the global rotations and translations
        data = init_guess(setting, data, dataset_obj, frames_seq=dataset_obj.frames, use_torso=True, **args)

        # Fix the shape parameters if the human shapes are known
        fix_params(setting, scale=setting['fixed_scale'], shape=setting['fixed_shape'])

        # Jointly optimize the cameras and motions
        results = non_linear_solver(setting, data, dataset_obj, **args)

        # Save results
        save_results(setting, data, results, dataset_obj, frames_seq=dataset_obj.frames, **args)

    elapsed = time.time() - start
    time_msg = time.strftime('%H hours, %M minutes, %S seconds',
                             time.gmtime(elapsed))
    print('Processing the data took: {}'.format(time_msg))

if __name__ == "__main__":
    # sys.argv = ["", "--config=cfg_files/fit_smpl.yaml"
    # ]
    args = parse_config()
    main(**args)





