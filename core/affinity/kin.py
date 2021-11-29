'''
 @FileName    : kin.py
 @EditTime    : 2021-07-14 12:14:46
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''

from time import sleep
import numpy as np

class Kin_Affinity:
    def __init__(self, cameras, MAX_DIST) -> None:
        self.cameras = []
        for i, (extri, intri) in enumerate(zip(cameras[0], cameras[1])):
            invK = np.linalg.inv(intri)
            R = extri[:3,:3]
            t = extri[:3,3].reshape(3,)
            self.cameras.append({'invK':invK, 'R':R, 'T':t})
        self.MAX_DIST = MAX_DIST
        self.line_joints = [5,6,7,8,9,10,11,12,13,14,15,16,17]

    def __call__(self, annots, appes, last_2d, dimGroups, joints):
        # calculate the ray
        nViews = len(annots)
        distance = np.zeros((dimGroups[-1], dimGroups[-1])) + self.MAX_DIST*2
        num_joints = 26
        last_pos_mat = np.zeros((dimGroups[-1], dimGroups[-1], num_joints, 4))
        for nv0 in range(nViews-1):
            for nv1 in range(nv0+1, nViews):
                if dimGroups[nv0]==dimGroups[nv0+1] or dimGroups[nv1]==dimGroups[nv1+1]:
                    continue
                cameras = [self.cameras[nv0], self.cameras[nv1]]
                if last_2d[nv0] is None or last_2d[nv1] is None:
                    pos = joints
                    confs = np.ones((num_joints, 1)) * 0.9
                    pos = np.hstack((pos, confs))
                else:
                    keyps = np.array([last_2d[nv0], last_2d[nv1]])
                    confs = np.sqrt(last_2d[nv0][:,-1] * last_2d[nv1][:,-1])[:,None]
                    pos = self.recompute3D(cameras, keyps)
                    pos = np.hstack((pos, confs))
                last_pos_mat[dimGroups[nv0]:dimGroups[nv0+1], dimGroups[nv1]:dimGroups[nv1+1]] = pos
                last_pos_mat[dimGroups[nv1]:dimGroups[nv1+1], dimGroups[nv0]:dimGroups[nv0+1]] = pos

        for nv0 in range(nViews-1):
            for nv1 in range(nv0+1, nViews):
                if dimGroups[nv0]==dimGroups[nv0+1] or dimGroups[nv1]==dimGroups[nv1+1]:
                    continue
                cameras = [self.cameras[nv0], self.cameras[nv1]]
                pos_mat = np.ones((len(annots[nv0]), len(annots[nv1]), num_joints, 4)) * 1000
                for v0 in range(len(annots[nv0])):
                    for v1 in range(len(annots[nv1])):
                        if annots[nv0][v0] is None or annots[nv1][v1] is None:
                            continue
                        else:
                            keyps = np.array([annots[nv0][v0], annots[nv1][v1]])
                            confs = np.sqrt(annots[nv0][v0][:,-1] * annots[nv1][v1][:,-1])[:,None]
                            pos = self.recompute3D(cameras, keyps)
                            pos = np.hstack((pos, confs))
                            pos_mat[v0, v1] = pos

                last_pos = last_pos_mat[dimGroups[nv0]:dimGroups[nv0+1], dimGroups[nv1]:dimGroups[nv1+1]]
                confs_last = last_pos[...,-1]
                confs = pos_mat[...,-1]
                confs = np.sqrt(confs_last * confs)
                dist = (np.linalg.norm(last_pos[...,:3] - pos_mat[...,:3], axis=-1) * confs).sum(axis=-1) / np.sum(confs, axis=-1)
                distance[dimGroups[nv0]:dimGroups[nv0+1], dimGroups[nv1]:dimGroups[nv1+1]] = dist
                distance[dimGroups[nv1]:dimGroups[nv1+1], dimGroups[nv0]:dimGroups[nv0+1]] = dist.T

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

        distance[distance > self.MAX_DIST] = self.MAX_DIST
        affinity = 1 - distance / self.MAX_DIST
        return affinity

    def recompute3D(self, cameras, keypoints):

        assert len(cameras) == len(keypoints)

        # homegeneous
        joint_conf = keypoints[:,:,2].copy()
        keypoints[:,:,2] = 1.
        # (keypoints, 2, 1, axis=2)

        num_joint = keypoints.shape[1]

        AtA = np.zeros((num_joint,3,3))
        Atb = np.zeros((num_joint,3))
        skelPos = np.zeros((num_joint, 3))

        for v in range(len(cameras)):
            conf = joint_conf[v]
            intri = cameras[v]['invK']
            R = cameras[v]['R']
            t = cameras[v]['T']
            keps = keypoints[v]
            ns = np.dot(intri, keps.T).T
            ns = self.nomalized(ns)
            nMats = self.fill_nMat(ns)
            nMats = np.matmul(R.T, nMats)
            AtA += np.matmul(nMats, R) * (conf[:,None,None] + 1e-6)
            Atb += np.matmul(-nMats, t) * (conf[:,None] + 1e-6)
        
        AtA = AtA.astype(np.float32)
        skelPos = np.linalg.solve(AtA, Atb)
        return skelPos

    def nomalized(self, z):
        norm = np.linalg.norm(z, axis=-1)[:,None]
        z = z / norm
        return z

    def fill_nMat(self, n):
        nMat = np.matmul(n[:,:,None], n[:,None,:])
        nMat = np.eye(3) - nMat
        return nMat
