'''
 @FileName    : visualization3d.py
 @EditTime    : 2021-12-13 15:44:57
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''

import open3d as o3d
import numpy as np
import time

class Visualization(object):
    def __init__(self) -> None:
        # set viewer
        self.viewer = o3d.visualization.Visualizer() #O3DVisualizer Visualizer
        window_size = 1200
        self.viewer.create_window(
            width=window_size + 1, height=window_size + 1,
            window_name='result'
        )
        self.count = 0

    def visualize_cameras(self, points, color, viz=True):
        point_cloud1 = o3d.geometry.PointCloud()
        point_cloud2 = o3d.geometry.PointCloud()
        lineset = o3d.geometry.LineSet()
        for i in range(len(points)//2):
            point_cloud1.points = o3d.utility.Vector3dVector(points[i*2].reshape(-1,3))
            point_cloud2.points = o3d.utility.Vector3dVector(points[i*2+1].reshape(-1,3))

            lineset_one = lineset.create_from_point_cloud_correspondences(point_cloud1, point_cloud2, [(0, 0)])
            self.viewer.add_geometry(lineset_one.paint_uniform_color(color))
            if viz:
                self.viewer.poll_events()

        # self.viewer.add_geometry(point_cloud)
        # for i in range(len(points)-1):
        #     point_cloud.points = o3d.utility.Vector3dVector(points[:i+1])
        #     point_cloud.paint_uniform_color([0.9, 0.7, 0.7])
        #     # o3d.visualization.draw_geometries([point_cloud])
        #     self.viewer.update_geometry(point_cloud)
        #     # self.viewer.update_geometry(mesh_gt)
        #     self.viewer.poll_events()

    def visualize_fitting(self, points, cameras):
        self.viewer.clear_geometries()
        points = points.reshape(-1, 3)
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        self.viewer.add_geometry(point_cloud)

        point_cloud.points = o3d.utility.Vector3dVector(points)
        # point_cloud.paint_uniform_color([1,0,0])
        # o3d.visualization.draw_geometries([point_cloud])
        self.viewer.update_geometry(point_cloud)
        # self.viewer.update_geometry(mesh_gt)
        # time.sleep(1)
        for cam in cameras:
            self.visualize_cameras(cam.T, [0,0,1], viz=False)
        self.viewer.poll_events()
        # import cv2
        # image = self.viewer.capture_screen_float_buffer()
        # # o3d.io.write_image('data/render_img_%05d.png' % self.count, image,quality=-1)
        # # cv2.imshow("img", np.asarray(image))
        # cv2.imwrite('data/render_img_%05d.png' % self.count, np.asarray(image)[:,:,::-1]*255.)
        # # self.viewer.capture_screen_image('data/render_img_%05d.jpg' % self.count, do_render=False)
        # self.count += 1

    def visualize_points(self, points, color):
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        self.viewer.add_geometry(point_cloud)
        for i in range(len(points)):
            point_cloud.points = o3d.utility.Vector3dVector(points[:i+1])
            point_cloud.paint_uniform_color(color)
            # o3d.visualization.draw_geometries([point_cloud])
            self.viewer.update_geometry(point_cloud)
            # self.viewer.update_geometry(mesh_gt)
            # time.sleep(1)
            self.viewer.poll_events()
            
    def visualize_lines(self, points, filted):
        point_cloud1 = o3d.geometry.PointCloud()
        point_cloud2 = o3d.geometry.PointCloud()
        point_cloud3 = o3d.geometry.PointCloud()
        point_cloud4 = o3d.geometry.PointCloud()

        # self.viewer.add_geometry(point_cloud1)
        # self.viewer.add_geometry(point_cloud2)

        lineset = o3d.geometry.LineSet()
        

        for i in range(len(points)-1):
            point_cloud1.points = o3d.utility.Vector3dVector(points[i:i+1])
            point_cloud2.points = o3d.utility.Vector3dVector(points[i+1:i+2])
            lineset_one = lineset.create_from_point_cloud_correspondences(point_cloud1, point_cloud2, [(0, 0)])
            self.viewer.add_geometry(lineset_one.paint_uniform_color([1, 0.5, 0.5]))

            point_cloud3.points = o3d.utility.Vector3dVector(filted[i:i+1])
            point_cloud4.points = o3d.utility.Vector3dVector(filted[i+1:i+2])
            lineset_one = lineset.create_from_point_cloud_correspondences(point_cloud3, point_cloud4, [(0, 0)])
            self.viewer.add_geometry(lineset_one.paint_uniform_color([0.5, 0.5, 1]))

            # point_cloud.points = o3d.utility.Vector3dVector(points[:i+1])
            # point_cloud.paint_uniform_color([0.9, 0.7, 0.7])
            # # o3d.visualization.draw_geometries([point_cloud])
            # self.viewer.update_geometry(point_cloud)
            # self.viewer.update_geometry(lineset_one)
            self.viewer.poll_events()

    def visualize_Pluckers(self, points, length, color):
        point_cloud1 = o3d.geometry.PointCloud()
        point_cloud2 = o3d.geometry.PointCloud()
        lineset = o3d.geometry.LineSet()
        # self.viewer.add_geometry(point_cloud1)
        # self.viewer.add_geometry(point_cloud2)

        for i in range(len(points)):
            point1 = np.cross(points[i][:3], points[i][3:6]) 
            point2 = point1 + length * points[i][:3]
            point_cloud1.points = o3d.utility.Vector3dVector([point1])
            point_cloud2.points = o3d.utility.Vector3dVector([point2])
            lineset_one = lineset.create_from_point_cloud_correspondences(point_cloud1, point_cloud2, [(0, 0)])
            self.viewer.add_geometry(lineset_one.paint_uniform_color(color))

            self.viewer.poll_events()

    def show(self):
        self.viewer.poll_events()

    def visualize_joints_trajectory(self, points):
        f_length, n_joints, _ = points.shape
        point_cloud1 = o3d.geometry.PointCloud()
        point_cloud2 = o3d.geometry.PointCloud()
        lineset = o3d.geometry.LineSet()

        color_s = np.array([0,0,1])
        color_e = np.array([1,0,1])
        color_t = (color_s - color_e) / f_length

        num_joints = points.shape[1]
    
        for i in range(f_length-1):
            for n in range(num_joints):
                point_cloud1.points = o3d.utility.Vector3dVector(points[i][n:n+1])
                point_cloud2.points = o3d.utility.Vector3dVector(points[i+1][n:n+1])
                lineset_one = lineset.create_from_point_cloud_correspondences(point_cloud1, point_cloud2, [(0, 0)])
                self.viewer.add_geometry(lineset_one.paint_uniform_color(color_e + color_t * i))

                # point_cloud3.points = o3d.utility.Vector3dVector(filted[i:i+1])
                # point_cloud4.points = o3d.utility.Vector3dVector(filted[i+1:i+2])
                # lineset_one = lineset.create_from_point_cloud_correspondences(point_cloud3, point_cloud4, [(0, 0)])
                # self.viewer.add_geometry(lineset_one.paint_uniform_color([0.5, 0.5, 1]))

                # point_cloud.points = o3d.utility.Vector3dVector(points[:i+1])
                # point_cloud.paint_uniform_color([0.9, 0.7, 0.7])
                # # o3d.visualization.draw_geometries([point_cloud])
                # self.viewer.update_geometry(point_cloud)
                # self.viewer.update_geometry(lineset_one)
                self.viewer.poll_events()

    def visualize_joints_test(self, points):
        f_length, n_joints, _ = points.shape
        point_cloud1 = o3d.geometry.PointCloud()
        point_cloud2 = o3d.geometry.PointCloud()
        lineset = o3d.geometry.LineSet()

        color_s = np.array([0,0,1])
        color_e = np.array([1,0,1])
        color_t = (color_s - color_e) / f_length

        num_joints = points.shape[1]
        t = np.array([0.1,0.1,0.1])
        for i in range(f_length-1):
            point_cloud1.points = o3d.utility.Vector3dVector(points[i])
            point_cloud2.points = o3d.utility.Vector3dVector(points[i+1])
            lineset_one = lineset.create_from_point_cloud_correspondences(point_cloud1, point_cloud2, [(i,i) for i in range(26)])
            self.viewer.add_geometry(lineset_one.paint_uniform_color(color_e + color_t * i))

            # point_cloud3.points = o3d.utility.Vector3dVector(filted[i:i+1])
            # point_cloud4.points = o3d.utility.Vector3dVector(filted[i+1:i+2])
            # lineset_one = lineset.create_from_point_cloud_correspondences(point_cloud3, point_cloud4, [(0, 0)])
            # self.viewer.add_geometry(lineset_one.paint_uniform_color([0.5, 0.5, 1]))

            # point_cloud.points = o3d.utility.Vector3dVector(points[:i+1])
            # point_cloud.paint_uniform_color([0.9, 0.7, 0.7])
            # # o3d.visualization.draw_geometries([point_cloud])
            # self.viewer.update_geometry(point_cloud)
            # self.viewer.update_geometry(lineset_one)
            self.viewer.poll_events()

