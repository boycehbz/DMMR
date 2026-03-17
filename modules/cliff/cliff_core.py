'''
 @FileName    : cliff_core.py
 @EditTime    : 2023-02-28 13:43:40
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''
import os
import cv2
import numpy as np
import torch
from modules.cliff.cliff.model.CLIFF import CLIFF
from modules.cliff.cliff.utils.smpl_torch_batch import SMPLModel
from utils.module_utils import draw_keyp, vis_img
import modules.cliff.cliff.utils.constants as constants
# from modules.cliff.cliff.utils.renderer_pyrd import Renderer
# from utils.renderer_moderngl import Renderer

from utils.module_utils import vis_img, save_camparam
from utils.FileLoaders import write_obj, save_pkl

class CLIFF_Predictor(object):
    def __init__(
        self,
        pose_checkpoint,
        type='res50',
        device=torch.device('cuda'),
        dtype=torch.float32
    ):
        self.device = device
        self.pose_checkpoint = pose_checkpoint

        # load smpl model 
        self.model_smpl_gpu = SMPLModel(
                            device=torch.device('cuda'),
                            model_path='./smpl/smpl/SMPL_NEUTRAL.pkl', 
                            data_type=dtype,
                        )

        # Load pose model
        self.model = CLIFF(self.model_smpl_gpu, type)
        print('load model')

        model_dict = self.model.state_dict()
        params = torch.load(self.pose_checkpoint)
        premodel_dict = params['model']
        premodel_dict = {k.replace('module.', ''): v for k ,v in premodel_dict.items() if k.replace('module.', '') in model_dict}
        # premodel_dict = {k: v for k ,v in premodel_dict.items() if k in model_dict}
        self.init_weights = premodel_dict
        model_dict.update(premodel_dict)
        self.model.load_state_dict(model_dict)
        self.model.to(self.device)
        self.model.eval()


    def bbox_from_detector(self, bbox, rescale=1.1):
        """
        Get center and scale of bounding box from bounding box.
        The expected format is [min_x, min_y, max_x, max_y].
        """
        # center
        center_x = (bbox[0] + bbox[2]) / 2.0
        center_y = (bbox[1] + bbox[3]) / 2.0
        center = torch.tensor([center_x, center_y])

        # scale
        bbox_w = bbox[2] - bbox[0]
        bbox_h = bbox[3] - bbox[1]
        bbox_size = max(bbox_w * 256 / float(192), bbox_h)
        scale = bbox_size / 200.0
        # adjust bounding box tightness
        scale *= rescale
        return center, scale


    def get_transform(self, center, scale, res, rot=0):
        """Generate transformation matrix."""
        # res: (height, width), (rows, cols)
        crop_aspect_ratio = res[0] / float(res[1])
        h = 200 * scale
        w = h / crop_aspect_ratio
        t = np.zeros((3, 3))
        t[0, 0] = float(res[1]) / w
        t[1, 1] = float(res[0]) / h
        t[0, 2] = res[1] * (-float(center[0]) / w + .5)
        t[1, 2] = res[0] * (-float(center[1]) / h + .5)
        t[2, 2] = 1
        if not rot == 0:
            rot = -rot  # To match direction of rotation from cropping
            rot_mat = np.zeros((3, 3))
            rot_rad = rot * np.pi / 180
            sn, cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0, :2] = [cs, -sn]
            rot_mat[1, :2] = [sn, cs]
            rot_mat[2, 2] = 1
            # Need to rotate around center
            t_mat = np.eye(3)
            t_mat[0, 2] = -res[1] / 2
            t_mat[1, 2] = -res[0] / 2
            t_inv = t_mat.copy()
            t_inv[:2, 2] *= -1
            t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
        return t


    def transform(self, pt, center, scale, res, invert=0, rot=0):
        """Transform pixel location to different reference."""
        t = self.get_transform(center, scale, res, rot=rot)
        if invert:
            t = np.linalg.inv(t)
        new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
        new_pt = np.dot(t, new_pt)
        return np.array([round(new_pt[0]), round(new_pt[1])], dtype=int) + 1



    def crop(self, img, center, scale, res):
        """
        Crop image according to the supplied bounding box.
        res: [rows, cols]
        """
        # Upper left point
        ul = np.array(self.transform([1, 1], center, scale, res, invert=1)) - 1
        # Bottom right point
        br = np.array(self.transform([res[1] + 1, res[0] + 1], center, scale, res, invert=1)) - 1

        # Padding so that when rotated proper amount of context is included
        pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)

        new_shape = [br[1] - ul[1], br[0] - ul[0]]
        if len(img.shape) > 2:
            new_shape += [img.shape[2]]
        new_img = np.zeros(new_shape, dtype=np.float32)

        # Range to fill new array
        new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
        new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
        # Range to sample from original image
        old_x = max(0, ul[0]), min(len(img[0]), br[0])
        old_y = max(0, ul[1]), min(len(img), br[1])
        try:
            new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]
        except Exception as e:
            print(e)

        new_img = cv2.resize(new_img, (res[1], res[0]))  # (cols, rows)

        return new_img, ul, br


    def process_image(self, orig_img_rgb, bbox,
                    crop_height=256,
                    crop_width=192):
        """
        Read image, do preprocessing and possibly crop it according to the bounding box.
        If there are bounding box annotations, use them to crop the image.
        If no bounding box is specified but openpose detections are available, use them to get the bounding box.
        """
        try:
            center, scale = self.bbox_from_detector(bbox)
        except Exception as e:
            print("Error occurs in person detection", e)
            # Assume that the person is centered in the image
            height = orig_img_rgb.shape[0]
            width = orig_img_rgb.shape[1]
            center = np.array([width // 2, height // 2])
            scale = max(height, width * crop_height / float(crop_width)) / 200.

        img, ul, br = self.crop(orig_img_rgb, center, scale, (crop_height, crop_width))
        crop_img = img.copy()

        img = img / 255.
        mean = np.array(constants.IMG_NORM_MEAN, dtype=np.float32)
        std = np.array(constants.IMG_NORM_STD, dtype=np.float32)
        norm_img = (img - mean) / std
        norm_img = np.transpose(norm_img, (2, 0, 1))

        return norm_img, center, scale, ul, br, crop_img

    def vis_input(self, image):
        image = image.detach().numpy()
        # Show image
        mean = np.array(constants.IMG_NORM_MEAN, dtype=np.float32)
        std = np.array(constants.IMG_NORM_STD, dtype=np.float32)
        image = image.transpose((0,2,3,1))
        image = image[:,:,:,::-1]
        image = image * std + mean
        for img in image:
            vis_img('img', img*255)

    def vis_results(self, imgname, results, viz=False):
        img_origin = cv2.imread(imgname)
        name = os.path.basename(imgname)
        img = img_origin.copy()
        renderer = Renderer(focal_length=results['focal_length'][0], img_w=img.shape[1], img_h=img.shape[0],
                            faces=self.model_smpl_gpu.faces,
                            same_mesh_color=True)
        front_view = renderer.render_front_view(results['pred_verts'],
                                                bg_img_rgb=img.copy())
        side_view = renderer.render_side_view(results['pred_verts'])

        # back_view = renderer.render_back_view(results['pred_verts'])

        # backside_view = renderer.render_backside_view(results['pred_verts'])

        top_view = renderer.render_top_view(results['pred_verts'])

        # img_origin = np.concatenate((img_origin, np.zeros_like(img_origin)), axis=1)

        rendered = np.concatenate((img_origin, front_view, side_view, top_view), axis=1)
        # rendered = np.concatenate((rendered, img_origin), axis=0)
        renderer.delete()
        if viz:
            vis_img('img', rendered)

        return rendered

        # cv2.imwrite('output/%s' %name, front_view)
        # print("save image to output/%s" %name)

        # mesh_folder = os.path.join('output/%s' %name.split('.')[0])
        # os.makedirs(mesh_folder, exist_ok=True)
        # for i, verts in enumerate(results['pred_verts']):
        #     self.model_smpl_gpu.write_obj(verts, os.path.join(mesh_folder, '%04d.obj' %i))

    # Data preprocess
    def predict(self, imgname, boxes, intris=None, viz=False, save_render=False):
        
        load_data = {}
        
        # Load image
        try:
            img = cv2.imread(imgname)[:,:,::-1].copy().astype(np.float32)
        except TypeError:
            print(imgname)
        img_h, img_w, _ = img.shape
        load_data["origin_img"] = imgname

        num_people = len(boxes)

        imgnames = ['empty'] * num_people
        norm_imgs = torch.zeros((num_people, 3, 256, 192)).float()
        centers = torch.zeros((num_people, 2)).float()
        scales = torch.zeros((num_people)).float()
        crop_uls = np.zeros((num_people, 2), dtype=np.float32)
        crop_brs = np.zeros((num_people, 2), dtype=np.float32)
        img_hs = np.zeros((num_people), dtype=np.float32)
        img_ws = np.zeros((num_people), dtype=np.float32)
        focal_lengthes = np.zeros((num_people), dtype=np.float32)

        for i in range(num_people):
            # if i > 10:
            #     break

            if intris is not None:
                focal_length = intris[i][0][0]
            else:
                focal_length = (img_h ** 2 + img_w ** 2) ** 0.5

            bbox = boxes[i]

            norm_img, center, scale, crop_ul, crop_br, _ = self.process_image(img.copy(), bbox)

            # Get 2D keypoints and apply augmentation transforms
            h = 200 * scale
            s = float(256) / h

            norm_imgs[i] = torch.from_numpy(norm_img)
            centers[i] = center
            scales[i] = scale
            crop_uls[i] = crop_ul
            crop_brs[i] = crop_br
            img_hs[i] = img_h
            img_ws[i] = img_w
            focal_lengthes[i] = focal_length

        load_data['instance'] = imgnames
        load_data["norm_img"] = norm_imgs.to(self.device)
        load_data["center"] = centers.to(self.device)
        load_data["scale"] = scales.to(self.device)
        load_data["crop_ul"] = torch.from_numpy(crop_uls).to(self.device)
        load_data["crop_br"] = torch.from_numpy(crop_brs).to(self.device)
        load_data["img_h"] = torch.from_numpy(img_hs).to(self.device)
        load_data["img_w"] = torch.from_numpy(img_ws).to(self.device)
        load_data["focal_length"] = torch.from_numpy(focal_lengthes).to(self.device)

        pred = self.model(load_data)
        # self.vis_input(load_data["norm_img"])

        results = {}
        results.update(pred_verts=pred['pred_verts'].detach().cpu().numpy().astype(np.float32))
        results.update(focal_length=load_data["focal_length"].detach().cpu().numpy().astype(np.float32))

        if save_render or viz:
            rendered = self.vis_results(imgname, results, viz)
        else:
            rendered = None

        num_agent = len(pred['pred_pose'])
        pose = pred['pred_pose'].reshape(num_agent, 72)
        pose = pose.detach().cpu().numpy().astype(np.float32)

        betas = pred['pred_shape']
        betas = betas.detach().cpu().numpy().astype(np.float32)

        trans = pred['pred_cam_t']
        trans = trans.detach().cpu().numpy().astype(np.float32)

        params = {}
        params.update(img_path=imgname)
        params.update(pose=pose)
        params.update(betas=betas)
        params.update(trans=trans)

        intri = np.eye(3)
        extri = np.eye(4)
        intri[0][0] = focal_lengthes[0]
        intri[1][1] = focal_lengthes[0]
        intri[0][2] = img_w / 2
        intri[1][2] = img_h / 2

        camparams = {'intri':[intri], 'extri':[extri]}

        return params, rendered, camparams, pred['pred_verts'].detach().cpu().numpy().astype(np.float32)


    def save_results(self, image_folder, params, rendered, camparams, verts):
        out_folder = os.path.join("output/CLIFF", os.path.basename(image_folder)) 
        os.makedirs(out_folder, exist_ok=True)

        name = os.path.basename(params['img_path'])

        render_folder = os.path.join(out_folder, 'images')
        os.makedirs(render_folder, exist_ok=True)
        cv2.imwrite(os.path.join(render_folder, name), rendered)
        print("save image to %s" %out_folder)

        mesh_folder = os.path.join(out_folder, 'meshes/%s' %name.split('.')[0])
        os.makedirs(mesh_folder, exist_ok=True)
        for i, verts in enumerate(verts):
            write_obj(verts, self.model.smpl.faces, os.path.join(mesh_folder, '%04d.obj' %i))

        params_folder = os.path.join(out_folder, 'params/%s' %name.split('.')[0])
        os.makedirs(params_folder, exist_ok=True)
        save_pkl(os.path.join(params_folder, '0000.pkl'), params)

        camparams_folder = os.path.join(out_folder, 'camparams/%s' %name.split('.')[0])
        os.makedirs(camparams_folder, exist_ok=True)
        save_camparam(os.path.join(camparams_folder, 'camparams.txt'), camparams['intri'], camparams['extri'])



    def inference_feature(self, data, viz=False):

        data["norm_img"] = data["norm_img"].to(self.device)
        data["center"] = data["center"].to(self.device)
        data["scale"] = data["scale"].to(self.device)
        data["img_h"] = data["img_h"].to(self.device)
        data["img_w"] = data["img_w"].to(self.device)
        data["focal_length"] = data["focal_length"].to(self.device)

        pred = self.model.feature(data)

        return pred

    def inference(self, data, viz=False):

        data["norm_img"] = data["norm_img"].to(self.device)[0]
        data["center"] = data["center"].to(self.device)[0]
        data["scale"] = data["scale"].to(self.device)[0]
        data["img_h"] = data["img_h"].to(self.device)[0]
        data["img_w"] = data["img_w"].to(self.device)[0]
        data["focal_length"] = data["focal_length"].to(self.device)[0]

        pred = self.model(data)

        return pred

    def visualize(self, img, poses, format='halpe', viz=False):
        for person in poses:
            img = draw_keyp(img, person, format=format)

        if viz:
            vis_img('image', img)

        return img

    @property
    def stopped(self):
        if self.opt.sp:
            return self._stopped
        else:
            return self._stopped.value
    @property
    def length(self):
        return len(self.all_imgs)

    @property
    def joint_pairs(self):
        """Joint pairs which defines the pairs of joint to be swapped
        when the image is flipped horizontally."""
        return [[1, 2], [3, 4], [5, 6], [7, 8],
                [9, 10], [11, 12], [13, 14], [15, 16]]
