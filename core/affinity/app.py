'''
 @FileName    : app.py
 @EditTime    : 2021-07-12 12:15:48
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''

import numpy as np
from .plucker import computeRay, dist_ll_pointwise_conf, dist_pl_pointwise_conf

def dist_appearances(p0, p1):
    # p0 = p0 / (np.linalg.norm(p0, axis=-1)[:,None] + 1e-5)
    # p1 = p1 / (np.linalg.norm(p1, axis=-1)[:,None] + 1e-5)

    dist = 1 - np.dot(p0, p1.T)

    return dist

class App_Affinity:
    def __init__(self, cameras, MAX_DIST) -> None:
        self.MAX_DIST = MAX_DIST

    def __call__(self, annots, appes, last_2d, dimGroups, joints):
        # calculate the ray
        nViews = len(appes)
        distance = np.zeros((dimGroups[-1], dimGroups[-1])) + self.MAX_DIST*2

        appearances = []
        for nv, appea in enumerate(appes):
            appeas = []
            for det in appea:
                if det is not None:
                    app = np.array(det)
                else:
                    app = np.zeros((512,))
                appeas.append(app)
            if len(appeas) > 0:
                appeas = np.stack(appeas)
            appearances.append(appeas)

        for nv0 in range(nViews-1):
            for nv1 in range(nv0+1, nViews):
                if dimGroups[nv0]==dimGroups[nv0+1] or dimGroups[nv1]==dimGroups[nv1+1]:
                    continue
                p0 = appearances[nv0]
                p1 = appearances[nv1]
                dist = dist_appearances(p0, p1)
                distance[dimGroups[nv0]:dimGroups[nv0+1], dimGroups[nv1]:dimGroups[nv1+1]] = dist
                distance[dimGroups[nv1]:dimGroups[nv1+1], dimGroups[nv0]:dimGroups[nv0+1]] = dist.T
        distance[distance > self.MAX_DIST] = self.MAX_DIST
        affinity = 1 - distance / self.MAX_DIST
        return affinity