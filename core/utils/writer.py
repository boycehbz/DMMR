import os
from os.path import join
import numpy as np
import cv2
# from mytools import save_json, merge
# from ..mytools import merge, plot_bbox, plot_keypoints
# from mytools.file_utils import read_json, save_json, read_annot, read_smpl, write_smpl, get_bbox_from_pose
# from .vis_base import plot_bbox, plot_keypoints, merge
# from .file_utils import write_keypoints3d, write_smpl, mkout, mkdir

from core.utils.module_utils import get_rgb

skeletons = {'coco17':[[0,1],[1,3],[0,2],[2,4],[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[11,13],[13,15],[6,12],[12,14],    [14,16],[11,12]],
            'MHHI':[[0,1],[1,2],[3,4],[4,5],[0,6],[3,6],[6,13],[13,7],[13,10],[7,8],[8,9],[10,11],[11,12]],
            }
colors = {'coco17':[(255,0,0), (255,82,0), (255,165,0), (255,210,0), (255,255,0), (127,255,0), (0,127,0), (0,255,0), (0,210,255), (0,127,255), (0,82,127), (0,210,127), (0,0,127), (0,0,255), (139,0,255), (139,0,127)],
            'MHHI':[(255,0,0), (255,82,0), (255,165,0), (255,210,0), (255,255,0), (127,255,0), (0,127,0), (0,255,0), (0,210,255), (0,127,255), (0,82,127), (0,210,127), (0,0,127)]}
row_col_ = {
    2: (2, 1),
    7: (2, 4),
    8: (2, 4),
    9: (3, 3),
    26: (4, 7)
}

class FileWriter:
    """
        This class provides:
                      |  write  | vis
        - keypoints2d |    x    |  o
        - keypoints3d |    x    |  o
        - smpl        |    x    |  o
    """
    def __init__(self, output_path, dataset_dir, config=None, basenames=[], cfg=None) -> None:
        self.out = output_path
        self.dataset_dir = dataset_dir
        keys = ['keypoints3d', 'match', 'smpl', 'skel', 'repro', 'keypoints']
        output_dict = {key:join(self.out, key) for key in keys}
        self.output_dict = output_dict
        
        # self.basenames = basenames
        # if cfg is not None:
        #     print(cfg, file=open(join(output_path, 'exp.yml'), 'w'))
        # self.save_origin = False
        # self.config = config
    
    def mkout(self, outname):
        dir_ = os.path.join(self.out, outname)
        if not os.path.exists(dir_):
            os.makedirs(dir_)

    def write_keypoints2d(self, ):
        pass

    def vis_keypoints2d_mv(self, images, lDetections, outname=None,
        vis_id=True):
        self.mkout(outname)
        n_frames = len(images[0])
        n_views = len(images)
        for nf in range(n_frames):
            images_vis = []
            for nv in range(n_views):
                img = cv2.imread(os.path.join(self.dataset_dir, 'images', images[nv][nf]))
                keypoints = lDetections[nv][nf]
                for pid, keyp in enumerate(keypoints):
                    if keyp is None or keyp[:,2].max() < 0.1:
                        continue
                    # if pid > 2:
                    #     continue
                    # keyp = keyp[:17]
                    bbox = self.calc_aabb(keyp)

                    img = self.plot_bbox(img, bbox, pid=pid, vis_id=vis_id)
                    img = self.draw_keyp(img, keyp, pid=pid)
                    # cv2.imwrite(os.path.join(self.out, outname, 'Camera%02d_%05d.jpg' %(nv, nf)), img)
                images_vis.append(img)
            if len(images_vis) > 1:
                images_vis = self.merge(images_vis, resize=False)
            else:
                images_vis = images_vis[0]
            if outname is not None:
                name = os.path.join(self.out, outname, '%05d.jpg' %nf)
                cv2.imwrite(name, images_vis)
        return images_vis
    
    def write_keypoints3d(self, results, outname):
        write_keypoints3d(outname, results)
    
    def vis_keypoints3d(self, result, outname):
        # visualize the repro of keypoints3d
        import ipdb; ipdb.set_trace()
    
    def vis_smpl(self, render_data, images, cameras, outname, add_back):
        mkout(outname)
        from ..visualize.renderer import Renderer
        render = Renderer(height=1024, width=1024, faces=None)
        render_results = render.render(render_data, cameras, images, add_back=add_back)
        image_vis = merge(render_results, resize=not self.save_origin)
        cv2.imwrite(outname, image_vis)
        return image_vis

    def _write_keypoints3d(self, results, nf=-1, base=None):
        os.makedirs(self.output_dict['keypoints3d'], exist_ok=True)
        if base is None:
            base = '{:06d}'.format(nf)
        savename = join(self.output_dict['keypoints3d'], '{}.json'.format(base))
        save_json(savename, results)
    
    def vis_detections(self, images, lDetections, to_img=True, vis_id=True):

        images_vis = []
        for nv, image in enumerate(images):
            img = image.copy()
            for det in lDetections[nv]:
                if key == 'match' and 'id_match' in det.keys():
                    pid = det['id_match']
                else:
                    pid = det['id']
                if key not in det.keys():
                    keypoints = det['keypoints']
                else:
                    keypoints = det[key]
                if 'bbox' not in det.keys():
                    bbox = self.calc_aabb(keypoints)
                else:
                    bbox = det['bbox']
                plot_bbox(img, bbox, pid=pid, vis_id=vis_id)
                plot_keypoints(img, keypoints, pid=pid, config=self.config, use_limb_color=False, lw=2)
            images_vis.append(img)
        image_vis = merge(images_vis, resize=not self.save_origin)
        if to_img:
            savename = join(self.output_dict[key], '{:06d}.jpg'.format(nf))
            cv2.imwrite(savename, image_vis)
        return image_vis
    
    def write_smpl(self, results, outname):
        write_smpl(outname, results)

    def vis_keypoints3d(self, infos, nf, images, cameras, mode='repro'):
        out = join(self.out, mode)
        os.makedirs(out, exist_ok=True)
        # cameras: (K, R, T)
        images_vis = []
        for nv, image in enumerate(images):
            img = image.copy()
            K, R, T = cameras['K'][nv], cameras['R'][nv], cameras['T'][nv]
            P = K @ np.hstack([R, T])
            for info in infos:
                pid = info['id']
                keypoints3d = info['keypoints3d']
                # 重投影
                kcam = np.hstack([keypoints3d[:, :3], np.ones((keypoints3d.shape[0], 1))]) @ P.T
                kcam = kcam[:, :2]/kcam[:, 2:]
                k2d = np.hstack((kcam, keypoints3d[:, -1:]))
                bbox = get_bbox_from_pose(k2d, img)
                plot_bbox(img, bbox, pid=pid, vis_id=pid)
                plot_keypoints(img, k2d, pid=pid, config=self.config, use_limb_color=False, lw=2)
            images_vis.append(img)
        savename = join(out, '{:06d}.jpg'.format(nf))
        image_vis = merge(images_vis, resize=False)
        cv2.imwrite(savename, image_vis)
        return image_vis

    def _vis_smpl(self, render_data_, nf, images, cameras, mode='smpl', base=None, add_back=False, extra_mesh=[]):
        out = join(self.out, mode)
        os.makedirs(out, exist_ok=True)
        from visualize.renderer import Renderer
        render = Renderer(height=1024, width=1024, faces=None, extra_mesh=extra_mesh)
        if isinstance(render_data_, list): # different view have different data
            for nv, render_data in enumerate(render_data_):
                render_results = render.render(render_data, cameras, images)
                image_vis = merge(render_results, resize=not self.save_origin)
                savename = join(out, '{:06d}_{:02d}.jpg'.format(nf, nv))
                cv2.imwrite(savename, image_vis)
        else:
            render_results = render.render(render_data_, cameras, images, add_back=add_back)
            image_vis = merge(render_results, resize=not self.save_origin)
            if nf != -1:
                if base is None:
                    base = '{:06d}'.format(nf)
                savename = join(out, '{}.jpg'.format(base))
                cv2.imwrite(savename, image_vis)
            return image_vis

    def get_row_col(self, l):
        if l in row_col_.keys():
            return row_col_[l]
        else:
            from math import sqrt
            row = int(sqrt(l) + 0.5)
            col = int(l/ row + 0.5)
            if row*col<l:
                col = col + 1
            if row > col:
                row, col = col, row
            return row, col


    def merge(self, images, row=-1, col=-1, resize=False, ret_range=False, **kwargs):
        if row == -1 and col == -1:
            row, col = self.get_row_col(len(images))
        height = images[0].shape[0]
        width = images[0].shape[1]
        ret_img = np.zeros((height * row, width * col, images[0].shape[2]), dtype=np.uint8) + 255
        ranges = []
        for i in range(row):
            for j in range(col):
                if i*col + j >= len(images):
                    break
                img = images[i * col + j]
                # resize the image size
                img = cv2.resize(img, (width, height))
                ret_img[height * i: height * (i+1), width * j: width * (j+1)] = img
                ranges.append((width*j, height*i, width*(j+1), height*(i+1)))
        if resize:
            min_height = 3000
            if ret_img.shape[0] > min_height:
                scale = min_height/ret_img.shape[0]
                ret_img = cv2.resize(ret_img, None, fx=scale, fy=scale)
        if ret_range:
            return ret_img, ranges
        return ret_img

    def calc_aabb(self, ptSets):
        lt = np.array([ptSets[0][0], ptSets[0][1]])
        rb = lt.copy()
        for pt in ptSets:
            if pt[0] == 0 and pt[1] == 0:
                continue
            lt[0] = min(lt[0], pt[0])
            lt[1] = min(lt[1], pt[1])
            rb[0] = max(rb[0], pt[0])
            rb[1] = max(rb[1], pt[1])

        return lt, rb

    def plot_bbox(self, img, bbox, pid, vis_id=True):
        # 画bbox: (l, t, r, b)
        x1, y1 = bbox[0]
        x2, y2 = bbox[1]
        x1 = int(round(x1))
        x2 = int(round(x2))
        y1 = int(round(y1))
        y2 = int(round(y2))
        color = get_rgb(pid)
        lw = max(img.shape[0]//300, 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, lw)
        if vis_id:
            font_scale = img.shape[0]/1000
            cv2.putText(img, '{}'.format(pid), (x1, y1+int(25*font_scale)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
        return img

    def draw_keyp(self, img, joints, pid=0, format='coco17'):
        color = tuple(get_rgb(pid))
        confidence = joints[:,2]
        joints = joints[:,:2].astype(np.int)
        for bone, c in zip(skeletons[format], colors[format]):
            if color is not None:
                c = color
            # c = (0,255,255)
            if confidence[bone[0]] > 0.1 and confidence[bone[1]] > 0.1:
                img = cv2.line(img, tuple(joints[bone[0]]), tuple(joints[bone[1]]), c, thickness=10)
        
        for p in joints:
            img = cv2.circle(img, tuple(p), 10, (0,0,255), -1)

        return img




