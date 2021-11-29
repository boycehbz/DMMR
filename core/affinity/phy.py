'''
 @FileName    : phy.py
 @EditTime    : 2021-07-09 13:51:24
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''
from time import sleep
import numpy as np
from .plucker import computeRay, dist_ll_pointwise_conf, dist_pl_pointwise_conf

class Phy_Affinity:
    def __init__(self, cameras, MAX_DIST) -> None:
        self.cameras = []
        for i, (extri, intri) in enumerate(zip(cameras[0], cameras[1])):
            invK = np.linalg.inv(intri)
            R = extri[:3,:3]
            t = extri[:3,3].reshape(3,1)
            self.cameras.append({'invK':invK, 'R':R, 'T':t})
        self.MAX_DIST = MAX_DIST
        self.line_joints = [5,6,7,8,9,10,11,12,13,14,15,16]

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
                    lines = computeRay(det[None, self.line_joints, :], 
                    cam['invK'], cam['R'], cam['T'])[0]
                else:
                    lines = np.zeros((12,7))
                pluckers.append(lines)
            if len(pluckers) > 0:
                pluckers = np.stack(pluckers)
            lPluckers.append(pluckers)

        # for nv0 in range(nViews):
        #     if dimGroups[nv0]==dimGroups[nv0+1]:
        #         continue
        #     p0 = lPluckers[nv0]
        #     joint = self.joints[None, :, :]
        #     dist = dist_pl_pointwise_conf(p0, joint)
        #     distance[dimGroups[nv0]:dimGroups[nv0+1], dimGroups[nv1]:dimGroups[nv1+1]] = dist
        #     distance[dimGroups[nv1]:dimGroups[nv1+1], dimGroups[nv0]:dimGroups[nv0+1]] = dist.T
        # from core.utils.visualization3d import Visualization
        # viz = Visualization()

        for nv0 in range(nViews-1):
            for nv1 in range(nv0+1, nViews):
                if dimGroups[nv0]==dimGroups[nv0+1] or dimGroups[nv1]==dimGroups[nv1+1]:
                    continue
                p0 = lPluckers[nv0][:, None]
                p1 = lPluckers[nv1][None, :]
                joint = joints.copy()[None, :, :]


                # viz.visualize_points(joint[0], [0,0,1])
                # viz.visualize_Pluckers(p1[0][0], -4, [1,0,0])
                # viz.visualize_Pluckers(p1[0][1], -4, [0,1,0])
                # viz.visualize_Pluckers(p1[0][2], -4, [0,1,1])
                # viz.visualize_Pluckers(p1[0][3], -4, [1,1,0])

                # viz.visualize_Pluckers(p0[0][0], -4, [1,0,0])
                # viz.visualize_Pluckers(p0[1][0], -4, [0,1,0])
                # viz.visualize_Pluckers(p0[2][0], -4, [0,1,1])
                # viz.visualize_Pluckers(p0[3][0], -4, [1,1,0])

                # count = 0
                # while True:
                #     count += 1
                #     if count > 1e5:
                #         break
                #     viz.show()

                dist = dist_pl_pointwise_conf(p0, p1, joint)
                distance[dimGroups[nv0]:dimGroups[nv0+1], dimGroups[nv1]:dimGroups[nv1+1]] = dist
                distance[dimGroups[nv1]:dimGroups[nv1+1], dimGroups[nv0]:dimGroups[nv0+1]] = dist.T
        distance[distance > self.MAX_DIST] = self.MAX_DIST
        affinity = 1 - distance / self.MAX_DIST
        return affinity
