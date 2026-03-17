'''
 @FileName    : yolox.py
 @EditTime    : 2023-02-03 13:42:18
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''
import os
import gc
import sys
import cv2
import torch

SAM2_ROOT = os.path.join(os.path.dirname(__file__), "sam2")
if SAM2_ROOT not in sys.path:
    sys.path.append(SAM2_ROOT)

from sam2.build_sam import build_sam2_video_predictor
import numpy as np
from modules.utils.module_utils import vis_img, annToMask
from modules.utils.FileLoaders import save_pkl

DEFAULT_FONT = cv2.FONT_HERSHEY_SIMPLEX

color = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 50),
                      (0, 255, 102), (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                      (77, 191, 255), (204, 77, 255), (77, 222, 255), (255, 156, 127),
                      (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36), 
                      (0, 77, 255), (0, 77, 255), (0, 77, 255), (0, 77, 255), (255, 156, 127), (255, 156, 127)]

class SAMURAI_core(object):
    def __init__(
        self,
        model_path,
        detector,
    ):
        self.model_cfg = self.determine_model_cfg(model_path)

        self.predictor = build_sam2_video_predictor(self.model_cfg, model_path, device="cuda:0")

        self.bbox_detector = detector

    def determine_model_cfg(self, model_path):
        if "large" in model_path:
            return "configs/samurai/sam2.1_hiera_l.yaml"
        elif "base_plus" in model_path:
            return "configs/samurai/sam2.1_hiera_b+.yaml"
        elif "small" in model_path:
            return "configs/samurai/sam2.1_hiera_s.yaml"
        elif "tiny" in model_path:
            return "configs/samurai/sam2.1_hiera_t.yaml"
        else:
            raise ValueError("Unknown model size in path!")
        
    def inference(self, img_folder, bboxes=None, viz=False, save_results=False):
        imgs = sorted(os.listdir(img_folder))

        if bboxes is None:
            img = cv2.imread(os.path.join(img_folder, imgs[0]))
            results, result_img = self.bbox_detector.predict(img, viz=False)
            bboxes = results['bbox']

        output = self.predict(bboxes, img_folder, viz=viz, save_results=save_results)

        tracking_data = []
        for i in range(len(imgs)):
            bbox = {key:output[key]['bbox'][i] for key in output.keys()}
            mask = {key:output[key]['segmentation'][i] for key in output.keys()}
            tracking_data.append({'bbox':bbox, 'mask':mask})

        total_number = len(output)
        return tracking_data, total_number

    def predict(self, bboxes, img_folder, viz=False, save_results=False):

        imgs = sorted(os.listdir(img_folder))
        output = {}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            state = self.predictor.init_state(img_folder, offload_video_to_cpu=True)

            for id, bbox in enumerate(bboxes):
                self.predictor.reset_state(state)
                _, _, masks = self.predictor.add_new_points_or_box(state, box=bbox, frame_idx=0, obj_id=id)
                results = {'bbox':[], 'segmentation':[]}
                for frame_idx, object_ids, masks in self.predictor.propagate_in_video(state):
                    mask_to_vis = {}
                    bbox_to_vis = {}

                    for obj_id, mask in zip(object_ids, masks):
                        mask = mask[0].cpu().numpy()
                        mask = mask > 0.0
                        non_zero_indices = np.argwhere(mask)
                        if len(non_zero_indices) == 0:
                            bbox = [0, 0, 0, 0]
                        else:
                            y_min, x_min = non_zero_indices.min(axis=0).tolist()
                            y_max, x_max = non_zero_indices.max(axis=0).tolist()
                            bbox = [x_min, y_min, x_max, y_max]
                        bbox_to_vis[obj_id] = bbox
                        mask_to_vis[obj_id] = mask

                        # if viz:
                        #     img = cv2.imread(os.path.join(img_folder, imgs[frame_idx]))
                        #     height, width = img.shape[:2]
                        #     for obj_id, mask in mask_to_vis.items():
                        #         mask_img = np.zeros((height, width, 3), np.uint8)
                        #         mask_img[mask] = color[(obj_id + 1) % len(color)]
                        #         img = cv2.addWeighted(img, 1, mask_img, 0.2, 0)

                        #     for obj_id, bbox in bbox_to_vis.items():
                        #         cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color[obj_id % len(color)], 2)

                        #     vis_img('img', img)

                        binary_mask = (mask > 0).astype(np.uint8)

                        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        polygons = []
                        for contour in contours:
                            contour = contour.flatten().tolist()
                            if len(contour) >= 6:  # 至少要有 3 个点（x, y）
                                polygons.append(contour)

                        results['segmentation'].append(polygons)
                        results['bbox'].append(bbox)
                
                        # mask_new = annToMask(polygons, height, width)

                        # vis_img('mask_new', mask_new*255)

                output[str(id)] = results

        if viz:
            self.show_output(output, img_folder)

        if save_results:
            self.save_results(output, img_folder)

        gc.collect()
        torch.clear_autocast_cache()
        torch.cuda.empty_cache()

        return output
    
    def overlay_mask_on_image(self, img, masks, bboxes, alpha=0.5):
        height, width = img.shape[:2]
        for obj_id, mask in masks:
            mask = mask.astype(np.bool)
            mask_img = np.zeros((height, width, 3), np.uint8)
            mask_img[mask] = color[(int(obj_id) + 1) % len(color)]
            img = cv2.addWeighted(img, 1, mask_img, alpha, 0)

        for obj_id, bbox in bboxes:
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color[int(obj_id) + 1 % len(color)], 2)
            cv2.putText(img, obj_id, (int(bbox[0]), int((bbox[1] + 26))), DEFAULT_FONT, 1, color[int(obj_id) + 1 % len(color)], 2)
        return img

    def save_results(self, params, img_folder):
        name = os.path.basename(img_folder)
        out_folder = os.path.join('output/samurai', name) 
        os.makedirs(out_folder, exist_ok=True)
        render_folder = os.path.join(out_folder, 'images') 
        os.makedirs(render_folder, exist_ok=True)

        imgs = sorted(os.listdir(img_folder))
        for idx, img_path in enumerate(imgs):
            img = cv2.imread(os.path.join(img_folder, img_path))
            h, w = img.shape[:2]

            bboxes = [(key, params[key]['bbox'][idx]) for key in params]
            masks = [(key, annToMask(params[key]['segmentation'][idx], h, w)) for key in params]

            render = self.overlay_mask_on_image(img.copy(), masks, bboxes)
            img = np.concatenate((img, render), axis=1)
            
            cv2.imwrite(os.path.join(render_folder, img_path), img)

        save_pkl(os.path.join(out_folder, 'results.pkl'), params)

    def show_output(self, output, img_folder):
        imgs = sorted(os.listdir(img_folder))

        for idx, img in enumerate(imgs):
            img = cv2.imread(os.path.join(img_folder, img))
            h, w = img.shape[:2]

            bboxes = [(key, output[key]['bbox'][idx]) for key in output]
            masks = [(key, annToMask(output[key]['segmentation'][idx], h, w)) for key in output]

            img = self.overlay_mask_on_image(img, masks, bboxes)
            vis_img('img', img)
