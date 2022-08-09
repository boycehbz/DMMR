import torch
import torch.nn as nn

from .hmr import hmr
from .smpl import SMPL
import cv2
import numpy as np
# import scipy.misc
from torchvision.transforms import Normalize

IMG_NORM_MEAN = [0.485, 0.456, 0.406]
IMG_NORM_STD = [0.229, 0.224, 0.225]

class SPIN(nn.Module):
    def __init__(self, device=torch.device('cpu'), **kwargs):
        super(SPIN, self).__init__()
        # Load pretrained model
        self.model = hmr('data/smpl_mean_params.npz').to(device)
        checkpoint = torch.load('data/spin_checkpoint.pt', map_location=device)
        self.model.load_state_dict(checkpoint['model'], strict=False)

        # Load SMPL model
        self.smpl = SMPL('models/smpl/SMPL_NEUTRAL.pkl',
                    batch_size=1,
                    create_transl=False).to(device)
        self.model.eval()
        self.device = device
        self.normalize_img = Normalize(mean=IMG_NORM_MEAN, std=IMG_NORM_STD)

    def forward(self, img):

        pred_rotmat, pred_betas, pred_camera = self.model(img.to(self.device))
        pred_output, verts = self.smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)

        return pred_output, verts

    def vis_img(self, name, im):
        ratiox = 800/int(im.shape[0])
        ratioy = 800/int(im.shape[1])
        if ratiox < ratioy:
            ratio = ratiox
        else:
            ratio = ratioy

        cv2.namedWindow(name,0)
        cv2.resizeWindow(name,int(im.shape[1]*ratio),int(im.shape[0]*ratio))
        #cv2.moveWindow(name,0,0)
        if im.max() > 1:
            im = im/255.
        cv2.imshow(name,im)
        if name != 'mask':
            cv2.waitKey()


    def get_transform(self, center, scale, res, rot=0):
        """Generate transformation matrix."""
        h = 200 * scale
        t = np.zeros((3, 3))
        t[0, 0] = float(res[1]) / h
        t[1, 1] = float(res[0]) / h
        t[0, 2] = res[1] * (-float(center[0]) / h + .5)
        t[1, 2] = res[0] * (-float(center[1]) / h + .5)
        t[2, 2] = 1
        if not rot == 0:
            rot = -rot # To match direction of rotation from cropping
            rot_mat = np.zeros((3,3))
            rot_rad = rot * np.pi / 180
            sn,cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0,:2] = [cs, -sn]
            rot_mat[1,:2] = [sn, cs]
            rot_mat[2,2] = 1
            # Need to rotate around center
            t_mat = np.eye(3)
            t_mat[0,2] = -res[1]/2
            t_mat[1,2] = -res[0]/2
            t_inv = t_mat.copy()
            t_inv[:2,2] *= -1
            t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
        return t

    def transform(self, pt, center, scale, res, invert=0, rot=0):
        """Transform pixel location to different reference."""
        t = self.get_transform(center, scale, res, rot=rot)
        if invert:
            t = np.linalg.inv(t)
        new_pt = np.array([pt[0]-1, pt[1]-1, 1.]).T
        new_pt = np.dot(t, new_pt)
        return new_pt[:2].astype(int)+1


    def crop(self, img, center, scale, res, rot=0):
        """Crop image according to the supplied bounding box."""
        # Upper left point
        ul = np.array(self.transform([1, 1], center, scale, res, invert=1))-1
        # Bottom right point
        br = np.array(self.transform([res[0]+1, 
                                res[1]+1], center, scale, res, invert=1))-1
        
        # Padding so that when rotated proper amount of context is included
        pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
        if not rot == 0:
            ul -= pad
            br += pad

        new_shape = [br[1] - ul[1], br[0] - ul[0]]
        if len(img.shape) > 2:
            new_shape += [img.shape[2]]
        new_img = np.zeros(new_shape)

        # Range to fill new array
        new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
        new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
        # Range to sample from original image
        old_x = max(0, ul[0]), min(len(img[0]), br[0])
        old_y = max(0, ul[1]), min(len(img), br[1])
        new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], 
                                                            old_x[0]:old_x[1]]

        if not rot == 0:
            # Remove padding
            new_img = scipy.misc.imrotate(new_img, rot)
            new_img = new_img[pad:-pad, pad:-pad]

        new_img = cv2.resize(new_img, res) #scipy.misc.imresize(new_img, res)
        return new_img


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

    def process_image(self, img_file, keypoints, input_res=224):
        """Read image, do preprocessing and possibly crop it according to the bounding box.
        If there are bounding box annotations, use them to crop the image.
        If no bounding box is specified but openpose detections are available, use them to get the bounding box.
        """
        
        img = cv2.imread(img_file)[:,:,::-1].copy() # PyTorch does not support negative stride at the moment
        lt, rb = self.calc_aabb(keypoints)
        center = (lt + rb) / 2
        width = max(rb[0]-lt[0], rb[1]-lt[1]) * 1.2
        scale = width / 200.0

        img = self.crop(img, center, scale, (input_res, input_res))
        img = img.astype(np.float32) / 255.
        
        # self.vis_img('img', img)

        img = torch.from_numpy(img).permute(2,0,1)
        norm_img = self.normalize_img(img.clone())[None]
        return img, norm_img


