'''
 @FileName    : data_parser.py
 @EditTime    : 2021-11-29 13:59:47
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import os.path as osp
import platform
import json

from collections import namedtuple

import cv2
import numpy as np

import torch
from torch.utils.data import Dataset

Keypoints = namedtuple('Keypoints',
                       ['keypoints', 'gender_gt', 'gender_pd'])

Keypoints.__new__.__defaults__ = (None,) * len(Keypoints._fields)


def read_keypoints(keypoint_fn, num_people, num_joint):
    if not os.path.exists(keypoint_fn):
        keypoints = [np.zeros((num_joint, 3))] * num_people # keypoints may not exist
        flags = np.zeros((num_people,))
        valid = 0
        return keypoints, flags, valid

    with open(keypoint_fn) as keypoint_file:
        data = json.load(keypoint_file)
    valid = 1
    keypoints = []
    flags = np.zeros((len(data['people'])))
    for idx, person_data in enumerate(data['people']):
        if person_data is None:
            body_keypoints = np.zeros((num_joint, 3),
                                    dtype=np.float32)
        else:
            flags[idx] = 1
            body_keypoints = np.array(person_data['pose_keypoints_2d'],
                                    dtype=np.float32)
            body_keypoints = body_keypoints.reshape([-1, 3])

        keypoints.append(body_keypoints)

    return keypoints[:num_people], flags[:num_people], valid

def read_joints(keypoint_fn, use_hands=True, use_face=True,
                   use_face_contour=False):
    """
    load 3D annotation
    """
    with open(keypoint_fn) as keypoint_file:
        data = json.load(keypoint_file)

    keypoints = []

    gender_pd = []
    gender_gt = []
    for idx, person_data in enumerate(data['people']):
        try:
            body_keypoints = np.array(person_data['pose_keypoints_3d'],
                                    dtype=np.float32)
            body_keypoints = body_keypoints.reshape([-1, 4])
            if use_hands:
                left_hand_keyp = np.array(
                    person_data['hand_left_keypoints_3d'],
                    dtype=np.float32).reshape([-1, 4])
                right_hand_keyp = np.array(
                    person_data['hand_right_keypoints_3d'],
                    dtype=np.float32).reshape([-1, 4])

                body_keypoints = np.concatenate(
                    [body_keypoints, left_hand_keyp, right_hand_keyp], axis=0)
            if use_face:
                # TODO: Make parameters, 17 is the offset for the eye brows,
                # etc. 51 is the total number of FLAME compatible landmarks
                face_keypoints = np.array(
                    person_data['face_keypoints_3d'],
                    dtype=np.float32).reshape([-1, 4])[17: 17 + 51, :]

                contour_keyps = np.array(
                    [], dtype=body_keypoints.dtype).reshape(0, 4)
                if use_face_contour:
                    contour_keyps = np.array(
                        person_data['face_keypoints_3d'],
                        dtype=np.float32).reshape([-1, 4])[:17, :]

                body_keypoints = np.concatenate(
                    [body_keypoints, face_keypoints, contour_keyps], axis=0)
            keypoints.append(body_keypoints)
        except:
            keypoints = None

        if 'gender_pd' in person_data:
            gender_pd.append(person_data['gender_pd'])
        if 'gender_gt' in person_data:
            gender_gt.append(person_data['gender_gt'])

    return Keypoints(keypoints=keypoints, gender_pd=gender_pd,
                     gender_gt=gender_gt)

def smpl_to_annotation(model_type='smpl', use_hands=False, use_face=False,
                     use_face_contour=False, pose_format='coco17'):

    if pose_format == 'halpe':
        if model_type == 'smplhalpe':
            # Halpe to SMPL
            return np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],
                            dtype=np.int32)
        else:
            raise ValueError('Unknown model type: {}'.format(model_type))

class FittingData(Dataset):

    NUM_BODY_JOINTS = 17
    NUM_HAND_JOINTS = 20

    def __init__(self, data_folder, img_folder='images',
                    keyp_folder='keypoints',
                    use_hands=False,
                    use_face=False,
                    dtype=torch.float32,
                    model_type='smplx',
                    joints_to_ign=None,
                    use_face_contour=False,
                    pose_format='coco17',
                    use_3d=False,
                    use_hip=True,
                    frames=1,
                    num_people=1,
                    **kwargs):
        super(FittingData, self).__init__()

        self.use_hands = use_hands
        self.use_face = use_face
        self.model_type = model_type
        self.dtype = dtype
        self.use_3d = use_3d
        self.use_hip = use_hip
        self.joints_to_ign = joints_to_ign
        self.use_face_contour = use_face_contour

        self.pose_format = pose_format
        if self.pose_format == 'halpe':
            self.NUM_BODY_JOINTS = 26

        self.num_joints = (self.NUM_BODY_JOINTS +
                            2 * self.NUM_HAND_JOINTS * use_hands)
        self.data_folder = data_folder
        self.img_folder = osp.join(data_folder, img_folder)
        self.keyp_folder = osp.join(data_folder, keyp_folder)

        img_serials = sorted(os.listdir(self.img_folder))
        self.img_paths = []
        for i_s in img_serials:
            i_s_dir = osp.join(self.img_folder, i_s)
            img_cameras = sorted(os.listdir(i_s_dir))
            this_serials = []
            for i_cam in img_cameras:
                i_c_dir = osp.join(i_s_dir, i_cam)
                cam_imgs = [osp.join(i_s, i_cam, img_fn)
                            for img_fn in os.listdir(i_c_dir)
                            if img_fn.endswith('.png') or
                            img_fn.endswith('.jpg') and
                            not img_fn.startswith('.')]
                cam_imgs = sorted(cam_imgs)
                this_serials.append(cam_imgs)
            self.img_paths.append(this_serials)

        self.cnt = 0
        self.serial_cnt = 0
        self.max_frames = frames
        self.min_frames = 13
        self.num_people = num_people
        # if len(cam_imgs) < frames:
        #     self.frames = len(cam_imgs)
        # else:
        self.frames = frames

    def get_model2data(self):
        # Map SMPL to Halpe
        return np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],
                            dtype=np.int32)

    def get_left_shoulder(self):
        return 2

    def get_right_shoulder(self):
        return 5

    def get_joint_weights(self):
        # The weights for the joint terms in the optimization
        optim_weights = np.ones(self.num_joints + 2 * self.use_hands +
                                self.use_face * 51 +
                                17 * self.use_face_contour,
                                dtype=np.float32)

        # Neck, Left and right hip
        # These joints are ignored because SMPL has no neck joint and the
        # annotation of the hips is ambiguous.
        
        # if self.joints_to_ign is not None and -1 not in self.joints_to_ign:
        #     optim_weights[self.joints_to_ign] = 0.
        # return torch.tensor(optim_weights, dtype=self.dtype)
        if (self.pose_format != 'lsp14' and self.pose_format != 'halpe') or not self.use_hip:
            optim_weights[11] = 0.
            optim_weights[12] = 0.
        return torch.tensor(optim_weights, dtype=self.dtype)


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        return self.read_item(img_path)

    
    def read_item(self, img_paths):
        """Load keypoints according to img name"""
        keypoints = []
        total_flags = []
        count = 0
        for imgs in img_paths:
            cam_keps = []
            cam_flag = []
            for img in imgs:
                if platform.system() == 'Windows':
                    seq_name, cam_name, f_name = img.split('\\')
                else:
                    seq_name, cam_name, f_name = img.split('/')
                index = f_name.split('.')[0]
                keypoint_fn = osp.join(self.keyp_folder, seq_name, cam_name, '%s_keypoints.json' %index)
                keypoints_, flags, valid = read_keypoints(keypoint_fn, self.num_people, self.NUM_BODY_JOINTS)
                count += valid

                cam_flag.append(flags)
                cam_keps.append(keypoints_)

            keypoints.append(cam_keps)
            total_flags.append(cam_flag)

        total_flags = np.array(total_flags, dtype=np.int)
        total_flags = np.max(total_flags, axis=0)

        camparam = os.path.join(self.data_folder, 'camparams', seq_name, 'camparams.txt')
        output_dict = { 'camparam': camparam,
                        'img_path': img_paths,
                        'keypoints': keypoints,
                        'flags':total_flags,
                        'count':count}

        return output_dict

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.serial_cnt >= len(self.img_paths):
            raise StopIteration

        img_path = self.img_paths[self.serial_cnt]

        img_paths = []
        for cam in img_path:
            if self.cnt+self.max_frames > len(cam):
                if len(cam) - self.cnt < self.min_frames:
                    img_paths.append(cam[-self.min_frames:])
                else:
                    img_paths.append(cam[self.cnt:])
            else:
                img_paths.append(cam[self.cnt:self.cnt+self.max_frames]) # 
        self.frames = len(img_paths[0])
        if self.cnt + self.max_frames >= len(cam):
            self.cnt = 0
            self.serial_cnt += 1
        else:
            self.cnt += self.frames

        return self.read_item(img_paths)
