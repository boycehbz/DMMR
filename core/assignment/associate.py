'''
  @ Date: 2021-06-04 21:58:37
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-06-25 11:50:10
  @ FilePath: /EasyMocapRelease/easymocap/assignment/associate.py
'''
import numpy as np
from core.assignment.criterion import *
# from ..mytools.reconstruction import batch_triangulate, projectN3
# from ..config import load_object

def batch_triangulate(keypoints_, Pall, keypoints_pre=None, lamb=1e3):
    # keypoints: (nViews, nJoints, 3)
    # Pall: (nViews, 3, 4)
    # A: (nJoints, nViewsx2, 4), x: (nJoints, 4, 1); b: (nJoints, nViewsx2, 1)
    v = (keypoints_[:, :, -1]>0).sum(axis=0)
    valid_joint = np.where(v > 1)[0]
    keypoints = keypoints_[:, valid_joint]
    conf3d = keypoints[:, :, -1].sum(axis=0)/v[valid_joint]
    # P2: P矩阵的最后一行：(1, nViews, 1, 4)
    P0 = Pall[None, :, 0, :]
    P1 = Pall[None, :, 1, :]
    P2 = Pall[None, :, 2, :]
    # uP2: x坐标乘上P2: (nJoints, nViews, 1, 4)
    uP2 = keypoints[:, :, 0].T[:, :, None] * P2
    vP2 = keypoints[:, :, 1].T[:, :, None] * P2
    conf = keypoints[:, :, 2].T[:, :, None]
    Au = conf * (uP2 - P0)
    Av = conf * (vP2 - P1)
    A = np.hstack([Au, Av])
    if keypoints_pre is not None:
        # keypoints_pre: (nJoints, 4)
        B = np.eye(4)[None, :, :].repeat(A.shape[0], axis=0)
        B[:, :3, 3] = -keypoints_pre[valid_joint, :3]
        confpre = lamb * keypoints_pre[valid_joint, 3]
        # 1, 0, 0, -x0
        # 0, 1, 0, -y0
        # 0, 0, 1, -z0
        # 0, 0, 0,   0
        B[:, 3, 3] = 0
        B = B * confpre[:, None, None]
        A = np.hstack((A, B))
    u, s, v = np.linalg.svd(A)
    X = v[:, -1, :]
    X = X / X[:, 3:]
    # out: (nJoints, 4)
    result = np.zeros((keypoints_.shape[1], 4))
    result[valid_joint, :3] = X[:, :3]
    result[valid_joint, 3] = conf3d
    return result

def projectN3(kpts3d, Pall):
    # kpts3d: (N, 3)
    nViews = len(Pall)
    kp3d = np.hstack((kpts3d[:, :3], np.ones((kpts3d.shape[0], 1))))
    kp2ds = []
    for nv in range(nViews):
        kp2d = Pall[nv] @ kp3d.T
        kp2d[:2, :] /= kp2d[2:, :]
        kp2ds.append(kp2d.T[None, :, :])
    kp2ds = np.vstack(kp2ds)
    kp2ds[..., -1] = kp2ds[..., -1] * (kpts3d[None, :, -1] > 0.)
    return kp2ds

def views_from_dimGroups(dimGroups):
    views = np.zeros(dimGroups[-1], dtype=np.int)
    for nv in range(len(dimGroups) - 1):
        views[dimGroups[nv]:dimGroups[nv+1]] = nv
    return views

def calc_aabb(ptSets):
    lt = np.array([ptSets[0][0], ptSets[0][1]])
    rb = lt.copy()
    for pt in ptSets:
        if pt[0] == 0 and pt[1] == 0:
            continue
        lt[0] = min(lt[0], pt[0])
        lt[1] = min(lt[1], pt[1])
        rb[0] = max(rb[0], pt[0])
        rb[1] = max(rb[1], pt[1])

    return np.array([lt, rb])

def set_keypoints2d(indices, annots, Pall, dimGroups):
    Vused = np.where(indices!=-1)[0]
    if len(Vused) < 1:
        return [], [], []
    keypoints2d = np.stack([annots[nv][indices[nv]-dimGroups[nv]].copy() for nv in Vused])
    bboxes = np.stack([calc_aabb(annots[nv][indices[nv]-dimGroups[nv]].copy()) for nv in Vused])
    Pused = Pall[Vused]
    return keypoints2d, bboxes, Pused, Vused

def load_criterions():
    criterions = []
    criterions.append(BaseCrit(0.1, 10))
    criterions.append(CritLenTorso(1, 8, 0.1, 0.8, 0.3))
    criterions.append(CritMinMax(2.2, 0.001))
    criterions.append(CritRange([-10,-10,-0.2], [10,10,2.5], 0.8, 0.001))
    criterions.append(CritWithTorso([18,19,6,5,12,11], 0.3))
    # criterions.append(CritLimbLength('halpe', 0.5, 0.1)
    return criterions

def simple_associate(annots, affinity, dimGroups, Pall, person_id, cfg={}):
    group = []
    group_results = []
    nViews = len(annots)
    nPeople = len(annots[0])
    criterions = load_criterions()
    n2D = dimGroups[-1]
    views = views_from_dimGroups(dimGroups)

    output = [None  for v in range(nViews)]

    views_cnt = np.zeros((affinity.shape[0], nViews))
    for nv in range(nViews):
        views_cnt[:, nv] = affinity[:, dimGroups[nv]:dimGroups[nv+1]].sum(axis=1)
    views_cnt = (views_cnt>0.5).sum(axis=1)
    # sortidx = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
    sortidx = np.argsort(-views_cnt)
    p2dAssigned = np.zeros(n2D, dtype=np.int) - 1
    indices_zero = np.zeros((nViews), dtype=np.int) - 1
    person_count = 0
    for n, idx in enumerate(sortidx):
        if n >= 1:
            break
        if p2dAssigned[idx] != -1:
            continue
        proposals = [indices_zero.copy()]
        for nv in range(nViews):
            match = np.where( 
                (affinity[idx, dimGroups[nv]:dimGroups[nv+1]] > 0.) 
              & (p2dAssigned[dimGroups[nv]:dimGroups[nv+1]] == -1) )[0]
            if len(match) > 0:
                match = match + dimGroups[nv]
                for proposal in proposals:
                    proposal[nv] = match[0]
            if len(match) > 1:
                proposals_new = []
                for proposal in proposals:
                    for col in match[1:]:
                        p = proposal.copy()
                        p[nv] = col
                        proposals_new.append(p)
                proposals = proposals_new
        results = []
        while len(proposals) > 0:
            proposal = proposals.pop()
            # less than two views
            if (proposal != -1).sum() < 2:
                continue
            # print('[associate] pop proposal: {}'.format(proposal))
            keypoints2d, bboxes, Pused, Vused = set_keypoints2d(proposal, annots, Pall, dimGroups)
            keypoints3d = batch_triangulate(keypoints2d, Pused)
            kptsRepro = projectN3(keypoints3d, Pused)
            err = ((kptsRepro[:, :, 2]*keypoints2d[:, :, 2]) > 0.) * np.linalg.norm(kptsRepro[:, :, :2] - keypoints2d[:, :, :2], axis=2)
            size = (bboxes[:, 1] - bboxes[:, 0]).max(axis=1, keepdims=True)
            err = err / size
            err_view = err.sum(axis=1)/((err>0.).sum(axis=1)+1e-5)
            flag = (err_view < 0.1).all() #cfg.max_repro_error
            err = err.sum()/(err>0).sum()
            # err_view = err.sum(axis=1)/((err>0.).sum(axis=1))
            # err = err.sum()/(err>0.).sum()
            # flag = err_view.max() < err_view.mean() * 2
            flag = True
            # for crit in criterions:
            #     if not crit(keypoints3d):
            #         flag = False
            #         break
            if flag:
                # print('[associate]: view {}'.format(Vused))
                results.append({
                    'indices': proposal,
                    'keypoints2d': keypoints2d,
                    'keypoints3d': keypoints3d,
                    'Vused': Vused,
                    'error': err
                })
            else:
                # make new proposals
                outlier_view = Vused[err_view.argmax()]
                proposal[outlier_view] = -1
                proposals.append(proposal)
        if len(results) == 0:
            continue
        if len(results) > 1:
            # print('[associate] More than one avalible results')
            results.sort(key=lambda x:x['error'])
        
        result = results[0]
        group_results.append(result)
        proposal = result['indices']
        Vused = result['Vused']
        # proposal中有-1的，所以需要使用Vused进行赋值
        p2dAssigned[proposal[Vused]] = 1

        indices = [id for id in proposal if id > -1]
        assert len(indices) == len(Vused)
        for v, id, kp2d in zip(Vused, indices, result['keypoints2d']):
            output[v] = kp2d
            annots[v][id-dimGroups[v]] = None
        person_count += 1
        # group.append(result['keypoints3d'])
        # #group.add(result)
    
    # group.dimGroups = dimGroups
    # assert len(group) == nPeople
    return annots, output