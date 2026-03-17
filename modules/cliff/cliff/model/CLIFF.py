# 2022.07.19 - Changed for CLIFF
#              Huawei Technologies Co., Ltd.

# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2019, University of Pennsylvania, Max Planck Institute for Intelligent Systems

# This program is free software; you can redistribute it and/or modify it
# under the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

# This script is borrowed and extended from SPIN

import torch
import torch.nn as nn
import numpy as np
import math
import os.path as osp

from torch.nn import functional as F
# from common.imutils import rot6d_to_rotmat
from cliff.cliff.model.backbones.hrnet.cls_hrnet import HighResolutionNet
from cliff.cliff.model.backbones.hrnet.hrnet_config import cfg
from cliff.cliff.model.backbones.hrnet.hrnet_config import update_config
from cliff.cliff.utils.imutils import cam_crop2full
from cliff.cliff.utils.rotation_conversions import matrix_to_axis_angle
from cliff.cliff.utils.geometry import perspective_projection, rot6d_to_rotmat
from cliff.cliff.model.backbones.resnet import ResNet

class CLIFF(nn.Module):
    """ SMPL Iterative Regressor with ResNet50 backbone"""

    def __init__(self, smpl, type='hrnet', img_feat_num=2048):
        super(CLIFF, self).__init__()
        curr_dir = osp.dirname(osp.abspath(__file__))

        if type == 'res50':
            self.encoder = ResNet(layers=[3, 4, 6, 3])
        else:
            config_file = "cliff/cliff/model/backbones/hrnet/models/cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100.yaml"
            update_config(cfg, config_file)
            self.encoder = HighResolutionNet(cfg)

        self.smpl = smpl
        self.init_trans = None

        npose = 24 * 6
        nshape = 10
        ncam = 3
        nbbox = 3

        fc1_feat_num = 1024
        fc2_feat_num = 1024
        final_feat_num = fc2_feat_num
        reg_in_feat_num = img_feat_num + nbbox + npose + nshape + ncam
        # CUDA Error: an illegal memory access was encountered
        # the above error will occur, if use mobilenet v3 with BN, so don't use BN
        self.fc1 = nn.Linear(reg_in_feat_num, fc1_feat_num)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(fc1_feat_num, fc2_feat_num)
        self.drop2 = nn.Dropout()

        self.decpose = nn.Linear(final_feat_num, npose)
        self.decshape = nn.Linear(final_feat_num, nshape)
        self.deccam = nn.Linear(final_feat_num, ncam)

        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        smpl_mean_params = 'smpl/smpl_mean_params.npz'
        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

    def forward(self, data, init_pose=None, init_shape=None, init_cam=None, n_iter=3):

        x = data["norm_img"]
        center = data["center"]
        scale = data["scale"]
        img_h = data["img_h"]
        img_w = data["img_w"]
        focal_length = data["focal_length"]

        cx, cy, b = center[:, 0], center[:, 1], scale * 200
        bbox_info = torch.stack([cx - img_w / 2., cy - img_h / 2., b], dim=-1)
        # The constants below are used for normalization, and calculated from H36M data.
        # It should be fine if you use the plain Equation (5) in the paper.
        bbox_info[:, :2] = bbox_info[:, :2] / focal_length.unsqueeze(-1) * 2.8  # [-1, 1]
        bbox_info[:, 2] = (bbox_info[:, 2] - 0.24 * focal_length) / (0.06 * focal_length)  # [-1, 1]


        batch_size = x.shape[0]
        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        xf = self.encoder(x)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        for i in range(n_iter):
            xc = torch.cat([xf, bbox_info, pred_pose, pred_shape, pred_cam], 1)

            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)

            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)
        pred_pose =  matrix_to_axis_angle(pred_rotmat.view(-1, 3, 3)).view(batch_size, 72)

        # convert the camera parameters from the crop camera to the full camera
        full_img_shape = torch.stack((img_h, img_w), dim=-1)
        pred_trans = cam_crop2full(pred_cam, center, scale, full_img_shape, focal_length)
        if self.init_trans is not None:
            pred_trans = self.init_trans.detach()

        pred_verts, pred_joints = self.smpl(pred_shape, pred_pose, pred_trans, halpe=True)

        camera_center = torch.stack([img_w/2, img_h/2], dim=-1)
        pred_keypoints_2d = perspective_projection(pred_joints,
                                                   rotation=torch.eye(3, device=pred_pose.device).unsqueeze(0).expand(batch_size, -1, -1),
                                                   translation=torch.zeros(3, device=pred_pose.device).unsqueeze(0).expand(batch_size, -1),
                                                   focal_length=focal_length,
                                                   camera_center=camera_center)
        # pred_keypoints_2d = pred_keypoints_2d / (self.img_res / 2.)

        h = 200 * scale
        s = float(256) / h
        pred_keypoints_2d[:,:,:2] = pred_keypoints_2d[:,:,:2] - center[:,None,:]
        pred_keypoints_2d[:,:,:2] = (pred_keypoints_2d[:,:,:2] * s[:,None,None]) / 256

        pred = {'pred_pose':pred_pose,\
                'pred_shape':pred_shape,\
                'pred_cam_t':pred_trans,\
                'pred_rotmat':pred_rotmat,\
                'pred_verts':pred_verts,\
                'pred_joints':pred_joints,\
                'focal_length':focal_length,\
                'pred_keypoints_2d':pred_keypoints_2d,\
                 }


        return pred

    def feature(self, data, init_pose=None, init_shape=None, init_cam=None, n_iter=3):

        x = data["norm_img"].squeeze(0)

        xf = self.encoder(x)

        return xf
