'''
 @FileName    : viz_cameras.py
 @EditTime    : 2021-11-29 15:30:36
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''
from core.utils.visualization3d import Visualization
from core.utils.module_utils import load_camera_para, add_camera_mesh
import numpy as np


if __name__ == '__main__':

    visualizer = Visualization()
    visualizer.visualize_meshes(['output/meshes/doubleB/00024_00.obj','output/meshes/doubleB/00024_01.obj'])
    extris, intris = load_camera_para('output/camparams/doubleB/00064.txt')
    for cam in extris:
        cam = add_camera_mesh(cam, camerascale=0.1)
        visualizer.visualize_cameras(cam.T, [0,0,1])
    
    while True:
        visualizer.show()
