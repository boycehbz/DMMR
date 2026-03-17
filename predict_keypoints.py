
import sys
import os
import torch
import argparse

sys.path.append('./')
from modules.CrossViewMatching import MultiViewReIDMatcher
from modules.utils.utils import *

# sys.argv = ["", "--images_root=data/Eval_DoubleB/images/doubleB"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Get last folder name from path")
    parser.add_argument(
        "--images_root",
        type=str,
        required=True,
        help="Path to images root directory"
    )
    args = parser.parse_args()


    images_root = Path(args.images_root)
    output_root = os.path.join('output/keypoints', os.path.basename(images_root)) 
    verbose = True
    save_process_results = False
    process_root = ''  

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    camera_dirs = list_camera_dirs(images_root)
    yolox, alphapose, vggt, samurai = build_models(device)

    matcher = MultiViewReIDMatcher(
        verbose=bool(verbose),
        yolox=yolox,
        alphapose=alphapose,
        vggt=vggt,
    )

    save_process = bool(save_process_results)
    process_root_path = get_process_root(output_root, process_root)
    match_root = process_root_path / 'match_result'
    tracking_root = process_root_path / 'tracking_result'
    pose_root = process_root_path / 'pose_result'

    print('[1/3] multi-view ReID...')
    view_gid_bbox, all_gids = run_first_timestamp_reid(camera_dirs, matcher)
    if False:
        print('  Saving match visualization...')
        save_match_visualization(camera_dirs, view_gid_bbox, match_root)

    print('[2/3] SAMURAI tracking per view...')
    tracked_bboxes = run_samurai_tracking(camera_dirs, view_gid_bbox, samurai)
    if False:
        print('  Saving tracking visualization...')
        save_tracking_visualization(camera_dirs, tracked_bboxes, all_gids, tracking_root)

    print('[3/3] AlphaPose per frame and save json...')
    run_alphapose_and_save(
        camera_dirs, tracked_bboxes, all_gids, alphapose, output_root,
        save_pose_vis=save_process, pose_root=pose_root
    )

    print(f'Done. Keypoints saved to: {output_root}')
    if save_process:
        print(f'Process results saved to: {process_root_path}')
