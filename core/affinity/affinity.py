'''
  @ Date: 2021-06-04 20:40:12
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-06-04 21:43:45
  @ FilePath: /EasyMocapRelease/easymocap/affinity/affinity.py
'''
import numpy as np
# from ..config import load_object
from .matchSVT import matchSVT
from .ray import Geo_Affinity
from .phy import Phy_Affinity
from .app import App_Affinity
from .pos import Pos_Affinity
from .kin import Kin_Affinity

def getDimGroups(lDetections):
    dimGroups = [0]
    for data in lDetections:
        dimGroups.append(dimGroups[-1] + len(data))
    views = np.zeros(dimGroups[-1], dtype=np.int)
    for nv in range(len(dimGroups) - 1):
        views[dimGroups[nv]:dimGroups[nv+1]] = nv
    return dimGroups, views

def composeAff(out, vis=False):
    names = list(out.keys())
    N = len(names)
    aff = out[names[0]].copy()
    for i in range(1, N):
        aff = aff * out[names[i]]
    aff = np.power(aff, 1/N)
    return aff

def SimpleConstrain(dimGroups):
    constrain = np.ones((dimGroups[-1], dimGroups[-1]))
    for i in range(len(dimGroups)-1):
        start, end = dimGroups[i], dimGroups[i+1]
        constrain[start:end, start:end] = 0
    N = constrain.shape[0]
    constrain[range(N), range(N)] = 1
    return constrain

class ComposedAffinity:
    def __init__(self, cameras):
        affinity = {}
        # It may be sensitive to the size of the scene
        # More robust implementation, TODO...
        affinity['ray'] = Geo_Affinity(cameras, 0.3)  # 0.1
        affinity['phy'] = Phy_Affinity(cameras, 0.5)  # 0.2

        # affinity['kin'] = Kin_Affinity(cameras, 0.5)  # 0.2
        # affinity['pos'] = Pos_Affinity(cameras, 30)  # 0.2
        # affinity['app'] = App_Affinity(cameras, 0.5)  # 0.2
        self.cameras = cameras
        self.affinity = affinity

    def __call__(self, annots, appes, last_2d, joints, images=None):
        dimGroups, maptoview = getDimGroups(annots)
        out = {}
        for key, model in self.affinity.items():
            out[key] = model(annots, appes, last_2d, dimGroups, joints)
        aff = composeAff(out, False)
        constrain = SimpleConstrain(dimGroups)
        observe = np.ones_like(aff)
        aff = constrain * aff
        if True:
            aff = matchSVT(aff, dimGroups, constrain, observe)
        aff[aff<0.2] = 0
        return aff, dimGroups