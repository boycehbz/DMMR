'''
 @FileName    : render.py
 @EditTime    : 2021-12-13 15:44:45
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''

import numpy as np
from numpy.lib.type_check import imag
import trimesh
import pyrender
from pyrender.constants import RenderFlags
import cv2

class WeakPerspectiveCamera(pyrender.Camera):
    def __init__(self,
                 scale,
                 translation,
                 znear=pyrender.camera.DEFAULT_Z_NEAR,
                 zfar=None,
                 name=None):
        super(WeakPerspectiveCamera, self).__init__(
            znear=znear,
            zfar=zfar,
            name=name,
        )
        self.scale = scale
        self.translation = translation

    def get_projection_matrix(self, width=None, height=None):
        P = np.eye(4)
        P[0, 0] = self.scale[0]
        P[1, 1] = self.scale[1]
        P[0, 3] = self.translation[0] * self.scale[0]
        P[1, 3] = -self.translation[1] * self.scale[1]
        P[2, 2] = -1
        return P

class Renderer:
    def __init__(self, resolution=(256, 256, 3), wireframe=False):

        self.resolution = resolution
        self.wireframe = wireframe
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=self.resolution[0],
            viewport_height=self.resolution[1],
            point_size=1.0)

        # set the scene
        self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.1, 0.1, 0.1))
        self.colors = [
            [.8, .1, .1], #'red': 
            [.1, .1, .8], #'bule': 
            [.1, .8, .1], #'green': 

            [.7, .7, .9], #'pink':
            [.9, .9, .8], #'neutral': 
            [.7, .75, .5], #'capsule': 
            [.5, .7, .75], #'yellow': 
        ]
        # self.renderer = pyrender.Viewer(self.scene, use_raymond_lighting=True, viewport_size=(1000,1000), cull_faces=False, run_in_thread=True)

    def Extrinsic_to_ModelViewMatrix(self, extri):
        extri[1] = -extri[1]
        extri[2] = -extri[2]
        return extri

    def vis_img(self, name, im):
        ratiox = 800/int(im.shape[0])
        ratioy = 800/int(im.shape[1])
        if ratiox < ratioy:
            ratio = ratiox
        else:
            ratio = ratioy

        cv2.namedWindow(name,0)
        cv2.resizeWindow(name,int(im.shape[1]*ratio),int(im.shape[0]*ratio))
        #cv2.moveWindow(name,0,0)
        if im.max() > 1:
            im = im/255.
        cv2.imshow(name,im)
        if name != 'mask':
            cv2.waitKey(1)

    def add_points_light(self, intensity=1.0, bbox=None):
        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=intensity)
        # Use 3 directional lights
        light_pose = np.eye(4)
        light_pose[:3, 3] = np.array([0, -2, 2]) + bbox[0]
        self.scene.add(light, pose=light_pose)
        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=intensity)
        light_pose[:3, 3] = np.array([0, 2, 2]) + bbox[0]
        self.scene.add(light, pose=light_pose)
        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=intensity)
        light_pose[:3, 3] = np.array([2, 2, 2]) + bbox[0]
        self.scene.add(light, pose=light_pose)
        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=intensity)

        # Use 3 directional lights
        light_pose = np.eye(4)
        light_pose[:3, 3] = np.array([0, -2, 2]) + bbox[1]
        self.scene.add(light, pose=light_pose)
        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=intensity)
        light_pose[:3, 3] = np.array([0, 2, 2]) + bbox[1]
        self.scene.add(light, pose=light_pose)
        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=intensity)
        light_pose[:3, 3] = np.array([2, 2, 2]) + bbox[1]
        self.scene.add(light, pose=light_pose)

    def __call__(self, verts, faces, rotation, trans, intri, img=None, color=[0.5,0.5,0.5], viz=False):
        
        # Add mesh
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0,
            alphaMode='OPAQUE',
            baseColorFactor=(color[0], color[1], color[2], 1.0)
        )
        
        rot = np.eye(4)
        rot[:3,:3] = rotation
        mesh.apply_transform(rot)
        # rot = trimesh.transformations.rotation_matrix(
        #     np.radians(180), [1, 0, 0])
        # mesh.apply_transform(rot)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
        mesh_node = self.scene.add(mesh, 'mesh')

        # Add cameras
        camera = pyrender.IntrinsicsCamera(fx=intri[0][0], fy=intri[1][1], cx=intri[0][2], cy=intri[1][2], zfar=8000)
        camera_pose = np.eye(4)
        trans = trans.reshape(-1,)
        trans[0] = -trans[0]
        camera_pose[:3,3] = trans
        camera_pose = self.Extrinsic_to_ModelViewMatrix(camera_pose)
        cam_node = self.scene.add(camera, pose=camera_pose)

        # Add light
        # self.add_points_light(10, bbox=mesh.bounds)
        light_nodes = self.use_raymond_lighting(15, trans=mesh.centroid - (mesh.centroid-camera_pose[:3,3]) * 0.6)

        if self.wireframe:
            render_flags = RenderFlags.RGBA | RenderFlags.ALL_WIREFRAME
        else:
            render_flags = RenderFlags.RGBA 

        image, _ = self.renderer.render(self.scene, flags=render_flags)

        visible_weight = 1
        if img is not None:
            valid_mask = (image[:, :, -1] > 0)[:, :,np.newaxis]
            if image.shape[-1] == 4:
                image = image[:,:,:-1]
            
            image = image * valid_mask * visible_weight + img * valid_mask * (1-visible_weight) + (1 - valid_mask) * img

        if viz:
            self.vis_img('img', image)
            
        self.scene.remove_node(mesh_node)
        self.scene.remove_node(cam_node)
        for n in light_nodes:
            self.scene.remove_node(n)
        return image

    def render_multiperson(self, verts, faces, rotation, trans, intri, img=None, viz=False):
        # Add mesh
        mesh_nodes = []
        mesh_bounds = []
        for i, vert in enumerate(verts):
            color = i % len(self.colors)
            color = self.colors[color]

            if vert is None:
                continue
            else:
                vert = vert.detach().cpu().numpy()
            mesh = trimesh.Trimesh(vertices=vert, faces=faces, process=False)
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0,
                alphaMode='OPAQUE',
                baseColorFactor=(color[0], color[1], color[2], 1.0)
            )

            rot = np.eye(4)
            rot[:3,:3] = rotation
            mesh.apply_transform(rot)

            mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
            mesh_node = self.scene.add(mesh, 'mesh')
            mesh_nodes.append(mesh_node)
            mesh_bounds.append(mesh.bounds)

        if len(mesh_bounds) < 1:
            return img

        mesh_bounds = np.array(mesh_bounds)
        top = np.mean(mesh_bounds[:,0,:], axis=0)
        bottom = np.mean(mesh_bounds[:,1,:], axis=0)
        pos = (top + bottom) / 2

        # Add light
        # self.add_points_light(10, bbox=mesh.bounds)
        # light_nodes = self.use_raymond_lighting(15, trans=pos-np.array([0,0,3]))

        # Add cameras
        camera = pyrender.IntrinsicsCamera(fx=intri[0][0], fy=intri[1][1], cx=intri[0][2], cy=intri[1][2], zfar=8000)
        camera_pose = np.eye(4)
        trans = trans.reshape(-1,)
        trans[0] = -trans[0]
        camera_pose[:3,3] = trans
        camera_pose = self.Extrinsic_to_ModelViewMatrix(camera_pose)
        cam_node = self.scene.add(camera, pose=camera_pose)

        # Create light source
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        # for DirectionalLight, only rotation matters
        light_pose = trimesh.transformations.rotation_matrix(np.radians(-45), [1, 0, 0])
        self.scene.add(light, pose=np.dot(camera_pose, light_pose))
        light_pose = trimesh.transformations.rotation_matrix(np.radians(45), [0, 1, 0])
        self.scene.add(light, pose=np.dot(camera_pose, light_pose))

        if self.wireframe:
            render_flags = RenderFlags.RGBA | RenderFlags.ALL_WIREFRAME
        else:
            render_flags = RenderFlags.RGBA 

        image, _ = self.renderer.render(self.scene, flags=render_flags)

        visible_weight = 1
        if img is not None:
            valid_mask = (image[:, :, -1] > 0)[:, :,np.newaxis]
            if image.shape[-1] == 4:
                image = image[:,:,:-1]
            
            image = image * valid_mask * visible_weight + img * valid_mask * (1-visible_weight) + (1 - valid_mask) * img

        if viz:
            self.vis_img('img', image)
        
        for n in mesh_nodes:
            self.scene.remove_node(n)
        self.scene.remove_node(cam_node)
        # for n in light_nodes:
        #     self.scene.remove_node(n)
        return image

    def _add_raymond_light(self, trans):
        from pyrender.light import DirectionalLight
        from pyrender.light import PointLight
        from pyrender.node import Node

        thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

        nodes = []

        for phi, theta in zip(phis, thetas):
            xp = np.sin(theta) * np.cos(phi)
            yp = np.sin(theta) * np.sin(phi)
            zp = np.cos(theta)

            z = np.array([xp, yp, zp])
            z = z / np.linalg.norm(z)
            x = np.array([-z[1], z[0], 0.0])
            if np.linalg.norm(x) == 0:
                x = np.array([1.0, 0.0, 0.0])
            x = x / np.linalg.norm(x)
            y = np.cross(z, x)

            matrix = np.eye(4)
            matrix[:3,:3] = np.c_[x,y,z]
            matrix[:3,3] = trans
            nodes.append(Node(
                light=PointLight(color=np.ones(3), intensity=1.0),
                matrix=matrix
            ))
        return nodes

    def use_raymond_lighting(self, intensity=1.0, trans=np.array([0,0,0])):
        # if not self.use_offscreen:
        #     sys.stderr.write('Interactive viewer already uses raymond lighting!\n')
        #     return
        nodes = []
        for n in self._add_raymond_light(trans):
            n.light.intensity = intensity / 3.0
            if not self.scene.has_node(n):
                self.scene.add_node(n)#, parent_node=pc)
            nodes.append(n)
        return nodes