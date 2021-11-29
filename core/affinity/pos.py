'''
 @FileName    : pos.py
 @EditTime    : 2021-07-12 15:39:34
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''

import numpy as np
from numpy.core.defchararray import array
from .plucker import computeRay, dist_ll_pointwise_conf, dist_pl_pointwise_conf

def dist_pos_conf(p0, p1, gtp0, gtp1):
    if p0[..., :2].max() < 0:
        conf0 = p0[...,-1]
    else:
        conf0 = np.sqrt(gtp0[...,-1] * p0[...,-1])
    if p1[..., :2].max() < 0:
        conf1 = p1[...,-1]
    else:
        conf1 = np.sqrt(gtp1[...,-1] * p1[...,-1])
    dist0 = np.sum(np.linalg.norm(gtp0[...,:2] - p0[...,:2], axis=-1) * conf0, axis=-1) / (1e-5 + conf0.sum(axis=-1))
    dist1 = np.sum(np.linalg.norm(gtp1[...,:2] - p1[...,:2], axis=-1) * conf1, axis=-1) / (1e-5 + conf1.sum(axis=-1))

    dist = (dist0[:,None] + dist1[None,:]) / 2

    return dist

class Pos_Affinity:
    def __init__(self, cameras, MAX_DIST) -> None:
        self.MAX_DIST = MAX_DIST
        self.torso_joints = [5,6,11,12]

    def __call__(self, annots, appes, last_2d, dimGroups, joints):

        # calculate the ray
        nViews = len(annots)
        distance = np.zeros((dimGroups[-1], dimGroups[-1])) + self.MAX_DIST*2

        joints_array = []
        for i, (t, p) in enumerate(zip(dimGroups[:-1], dimGroups[1:])):
            arrays = np.zeros((p-t, len(self.torso_joints), 3))
            if last_2d[i] is not None:
                arrays[:] = last_2d[i][self.torso_joints]
            joints_array.append(arrays)

        pose2ds = []
        for nv, annot in enumerate(annots):
            poses = []
            for det in annot:
                if det is not None:
                    lines = det[self.torso_joints]
                else:
                    lines = np.ones((len(self.torso_joints),3)) * -1000
                    lines[:,-1] = 1
                poses.append(lines)
            if len(poses) > 0:
                poses = np.stack(poses)
            pose2ds.append(poses)

        for nv0 in range(nViews-1):
            for nv1 in range(nv0+1, nViews):
                if dimGroups[nv0]==dimGroups[nv0+1] or dimGroups[nv1]==dimGroups[nv1+1]:
                    continue
                gtp0 = joints_array[nv0]
                gtp1 = joints_array[nv1]
                p0 = pose2ds[nv0]
                p1 = pose2ds[nv1]
                dist = dist_pos_conf(p0, p1, gtp0, gtp1)
                distance[dimGroups[nv0]:dimGroups[nv0+1], dimGroups[nv1]:dimGroups[nv1+1]] = dist
                distance[dimGroups[nv1]:dimGroups[nv1+1], dimGroups[nv0]:dimGroups[nv0+1]] = dist.T
        distance[distance > self.MAX_DIST] = self.MAX_DIST
        affinity = 1 - distance / self.MAX_DIST
        return affinity