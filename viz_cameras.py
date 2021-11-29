'''
 @FileName    : viz_cameras.py
 @EditTime    : 2021-11-29 15:30:36
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''
from core.utils.visualization3d import Visualization
from core.utils.module_utils import load_camera_para
import numpy as np

def add_camera_mesh(extrinsic, camerascale=1):

    # 12 points camera
    r = np.zeros((3,4,3))
    r[0][0] = np.array([-0.5, 0.5, 0]) * camerascale
    r[0][1] = np.array([0.5, 0.5, 0]) * camerascale
    r[0][2] = np.array([0.5, -0.5, 0]) * camerascale
    r[0][3] = np.array([-0.5, -0.5, 0]) * camerascale

    r[1][0] = np.array([-1, 1, 1]) * camerascale
    r[1][1] = np.array([1, 1, 1]) * camerascale
    r[1][2] = np.array([1, -1, 1]) * camerascale
    r[1][3] = np.array([-1, -1, 1]) * camerascale

    r[2][0] = np.array([-0.5, 0.5, -2]) * camerascale
    r[2][1] = np.array([0.5, 0.5, -2]) * camerascale
    r[2][2] = np.array([0.5, -0.5, -2]) * camerascale
    r[2][3] = np.array([-0.5, -0.5, -2]) * camerascale

    P = np.zeros((3, 40))
    for i in range(3):
        P[:,i * 8 + 0] = r[i][0] 
        P[:,i * 8 + 1] = r[i][1]
        P[:,i * 8 + 2] = r[i][1] 
        P[:,i * 8 + 3] = r[i][2]
        P[:,i * 8 + 4] = r[i][2] 
        P[:,i * 8 + 5] = r[i][3]
        P[:,i * 8 + 6] = r[i][3] 
        P[:,i * 8 + 7] = r[i][0]

    for i in range(2):
        P[:,24 + i * 8 + 0] = r[0][0] 
        P[:,24 + i * 8 + 1] = r[i + 1][0]
        P[:,24 + i * 8 + 2] = r[0][1] 
        P[:,24 + i * 8 + 3] = r[i + 1][1]
        P[:,24 + i * 8 + 4] = r[0][2] 
        P[:,24 + i * 8 + 5] = r[i + 1][2]
        P[:,24 + i * 8 + 6] = r[0][3] 
        P[:,24 + i * 8 + 7] = r[i + 1][3]

    # // transform from camera space to object space
    # // this step is critical for visualizing the cameras since our viewpoint is in the object space
    M = np.linalg.inv(extrinsic)
    for i in range(P.shape[1]):
        t = np.ones((4,))
        t[:3] = P[:,i]
        p = np.dot(M, t)
        P[:,i] = p[:3] / p[3]

    return P

if __name__ == '__main__':

    visualizer = Visualization()
    extris, intris = load_camera_para(R'D:\BuzhenHuang_Programs\DMMR\output\cameras\doubleB\00064.txt')
    for cam in extris:
        # if cam[0][0] == 1:
        #     continue
        cam = add_camera_mesh(cam, camerascale=0.1)
        visualizer.visualize_cameras(cam.T, [0,0,1])
    while True:
        visualizer.show()
