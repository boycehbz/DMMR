'''
  @ Date: 2021-06-04 21:34:19
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-06-05 16:26:06
  @ FilePath: /EasyMocapRelease/easymocap/affinity/ray.py
'''
import numpy as np
from .plucker import computeRay, dist_ll_pointwise_conf

class Geo_Affinity:
    def __init__(self, cameras, MAX_DIST) -> None:
        self.cameras = []
        for i, (extri, intri) in enumerate(zip(cameras[0], cameras[1])):
            invK = np.linalg.inv(intri)
            R = extri[:3,:3]
            t = extri[:3,3].reshape(3,1)
            self.cameras.append({'invK':invK, 'R':R, 'T':t})
        self.MAX_DIST = MAX_DIST
    
    def __call__(self, annots, appes, last_2d, dimGroups, joints):
        # calculate the ray
        nViews = len(annots)
        distance = np.zeros((dimGroups[-1], dimGroups[-1])) + self.MAX_DIST*2

        lPluckers = []
        for nv, annot in enumerate(annots):
            cam = self.cameras[nv]
            pluckers = []
            for det in annot:
                if det is not None:
                    lines = computeRay(det[None, 5:17, :], 
                    cam['invK'], cam['R'], cam['T'])[0]
                else:
                    lines = np.zeros((12,7))
                pluckers.append(lines)
            if len(pluckers) > 0:
                pluckers = np.stack(pluckers)
            lPluckers.append(pluckers)
        for nv0 in range(nViews-1):
            for nv1 in range(nv0+1, nViews):
                if dimGroups[nv0]==dimGroups[nv0+1] or dimGroups[nv1]==dimGroups[nv1+1]:
                    continue
                p0 = lPluckers[nv0][:, None]
                p1 = lPluckers[nv1][None, :]
                dist = dist_ll_pointwise_conf(p0, p1)
                distance[dimGroups[nv0]:dimGroups[nv0+1], dimGroups[nv1]:dimGroups[nv1+1]] = dist
                distance[dimGroups[nv1]:dimGroups[nv1+1], dimGroups[nv0]:dimGroups[nv0+1]] = dist.T
        distance[distance > self.MAX_DIST] = self.MAX_DIST
        affinity = 1 - distance / self.MAX_DIST
        return affinity