'''
 @FileName    : utils.py
 @EditTime    : 2021-11-24 19:52:36
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import platform

import numpy as np

import torch
import torch.nn as nn
import os.path as osp
import os
try:
    import cPickle as pickle
except ImportError:
    import pickle
import cv2
from copy import deepcopy
from core.utils.camera import create_camera
from core.SPIN.spin import SPIN

def generate_colorbar(N = 20, cmap = 'jet'):
    bar = ((np.arange(N)/(N-1))*255).astype(np.uint8).reshape(-1, 1)
    colorbar = cv2.applyColorMap(bar, cv2.COLORMAP_JET).squeeze()
    if False:
        colorbar = np.clip(colorbar + 64, 0, 255)
    import random
    random.seed(666)
    index = [i for i in range(N)]
    random.shuffle(index)
    rgb = colorbar[index, :]
    rgb = rgb.tolist()
    return rgb

colors_bar_rgb = generate_colorbar(cmap='hsv')
# colors_bar_rgb = [(18,18,196),(97,184,27),(175,103,42),(97,155,213),(127,75,152)]


colors_table = {
    'b': [0.65098039, 0.74117647, 0.85882353],
    '_pink': [.9, .7, .7],
    '_mint': [ 166/255.,  229/255.,  204/255.],
    '_mint2': [ 202/255.,  229/255.,  223/255.],
    '_green': [ 153/255.,  216/255.,  201/255.],
    '_green2': [ 171/255.,  221/255.,  164/255.],
    'r': [ 251/255.,  128/255.,  114/255.],
    '_orange': [ 253/255.,  174/255.,  97/255.],
    'y': [ 250/255.,  230/255.,  154/255.],
    '_r':[255/255,0,0],
    'g':[0,255/255,0],
    '_b':[0,0,255/255],
    'k':[0,0,0],
    '_y':[255/255,255/255,0],
    'purple':[128/255,0,128/255],
    'smap_b':[51/255,153/255,255/255],
    'smap_r':[255/255,51/255,153/255],
    'smap_b':[51/255,255/255,153/255],
}

def get_rgb(index):
    if isinstance(index, int):
        if index == -1:
            return (255, 255, 255)
        if index < -1:
            return (0, 0, 0)
        col = colors_bar_rgb[index%len(colors_bar_rgb)]
    else:
        col = colors_table.get(index, (1, 0, 0))
        col = tuple([int(c*255) for c in col[::-1]])
    return col


def estimate_translation_from_intri(S, joints_2d, joints_conf, fx=5000., fy=5000., cx=128., cy=128.):
    num_joints = S.shape[0]
    # focal length
    f = np.array([fx, fy])
    # optical center
   # center = np.array([img_size/2., img_size/2.])
    center = np.array([cx, cy])
    # transformations
    Z = np.reshape(np.tile(S[:,2],(2,1)).T,-1)
    XY = np.reshape(S[:,0:2],-1)
    O = np.tile(center,num_joints)
    F = np.tile(f,num_joints)
    weight2 = np.reshape(np.tile(np.sqrt(joints_conf),(2,1)).T,-1)

    # least squares
    Q = np.array([F*np.tile(np.array([1,0]),num_joints), F*np.tile(np.array([0,1]),num_joints), O-np.reshape(joints_2d,-1)]).T
    c = (np.reshape(joints_2d,-1)-O)*Z - F*XY

    # weighted least squares
    W = np.diagflat(weight2)
    Q = np.dot(W,Q)
    c = np.dot(W,c)

    # square matrix
    A = np.dot(Q.T,Q)
    b = np.dot(Q.T,c)

    # test
    A += np.eye(A.shape[0]) * 1e-6

    # solution
    trans = np.linalg.solve(A, b)
    return trans

def cal_trans(J3d, J2d, intri):
    fx = intri[0][0]
    fy = intri[1][1]
    cx = intri[0][2]
    cy = intri[1][2]
    j_conf = J2d[:,2] 
    gt_cam_t = estimate_translation_from_intri(J3d, J2d[:,:2], j_conf, cx=cx, cy=cy, fx=fx, fy=fy)
    return gt_cam_t

def load_pkl(path):
    """"
    load pkl file
    """
    param = pickle.load(open(path, 'rb'),encoding='iso-8859-1')
    return param


def vis_img(name, im):
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
        cv2.waitKey()


def surface_projection(vertices, faces, joint, exter, intri, image, op):
    im = deepcopy(image)

    intri_ = np.insert(intri,3,values=0.,axis=1)
    temp_v = np.insert(vertices,3,values=1.,axis=1).transpose((1,0))

    out_point = np.dot(exter, temp_v)
    dis = out_point[2]
    out_point = (np.dot(intri_, out_point) / dis)[:-1]
    out_point = (out_point.astype(np.int32)).transpose(1,0)
    max = dis.max()
    min = dis.min()
    t = 255./(max-min)
    
    img_faces = []
    color = (255, 255, 255)
    for f in faces:
        point = out_point[f]
        im = cv2.polylines(im, [point], True, color, 1)
        
    temp_joint = np.insert(joint,3,values=1.,axis=1).transpose((1,0))
    out_point = np.dot(exter, temp_joint)
    dis = out_point[2]
    out_point = (np.dot(intri_, out_point) / dis)[:-1].astype(np.int32)
    out_point = out_point.transpose(1,0)
    for i in range(len(out_point)):
        if i == op:
            im = cv2.circle(im, tuple(out_point[i]), 5, (0,0,255),-1)
        else:
            im = cv2.circle(im, tuple(out_point[i]), 5, (255,0,0),-1)

    ratiox = 800/int(im.shape[0])
    ratioy = 800/int(im.shape[1])
    if ratiox < ratioy:
        ratio = ratiox
    else:
        ratio = ratioy

    cv2.namedWindow("mesh",0)
    cv2.resizeWindow("mesh",int(im.shape[1]*ratio),int(im.shape[0]*ratio))
    cv2.moveWindow("mesh",0,0)
    cv2.imshow('mesh',im)
    cv2.waitKey()

    return out_point, im

def joint_projection(joint, extri, intri, image, viz=False):

    im = deepcopy(image)

    intri_ = np.insert(intri,3,values=0.,axis=1)
    temp_joint = np.insert(joint,3,values=1.,axis=1).transpose((1,0))
    out_point = np.dot(extri, temp_joint)
    dis = out_point[2]
    out_point = (np.dot(intri_, out_point) / dis)[:-1].astype(np.int32)
    out_point = out_point.transpose(1,0)

    if viz:
        for i in range(len(out_point)):
            im = cv2.circle(im, tuple(out_point[i]), 10, (0,0,255),-1)


        ratiox = 800/int(im.shape[0])
        ratioy = 800/int(im.shape[1])
        if ratiox < ratioy:
            ratio = ratiox
        else:
            ratio = ratioy

        cv2.namedWindow("mesh",0)
        cv2.resizeWindow("mesh",int(im.shape[1]*ratio),int(im.shape[0]*ratio))
        cv2.moveWindow("mesh",0,0)
        cv2.imshow('mesh',im)
        cv2.waitKey()

    return out_point, im


def rel_change(prev_val, curr_val):
    return (prev_val - curr_val) / max([np.abs(prev_val), np.abs(curr_val), 1])

def load_camera_para(file):
    """"
    load camera parameters
    """
    campose = []
    intra = []
    campose_ = []
    intra_ = []
    f = open(file,'r')
    for line in f:
        line = line.strip('\n')
        line = line.rstrip()
        words = line.split()
        if len(words) == 3:
            intra_.append([float(words[0]),float(words[1]),float(words[2])])
        elif len(words) == 4:
            campose_.append([float(words[0]),float(words[1]),float(words[2]),float(words[3])])
        else:
            pass

    index = 0
    intra_t = []
    for i in intra_:
        index+=1
        intra_t.append(i)
        if index == 3:
            index = 0
            intra.append(intra_t)
            intra_t = []

    index = 0
    campose_t = []
    for i in campose_:
        index+=1
        campose_t.append(i)
        if index == 3:
            index = 0
            campose_t.append([0.,0.,0.,1.])
            campose.append(campose_t)
            campose_t = []
    
    return np.array(campose), np.array(intra)

def get_rot_trans(campose, photoscan=False):
    trans = []
    rot = []
    for cam in campose:
        # for photoscan parameters
        if photoscan:
            cam = np.linalg.inv(cam)  
        trans.append(cam[:3,3])
        rot.append(cam[:3,:3])
        # rot.append(cv2.Rodrigues(cam[:3,:3])[0])

    return trans, rot

class JointMapper(nn.Module):
    def __init__(self, joint_maps=None):
        super(JointMapper, self).__init__()
        if joint_maps is None:
            self.joint_maps = joint_maps
        else:
            self.register_buffer('joint_maps',
                                 torch.tensor(joint_maps, dtype=torch.long))

    def forward(self, joints, **kwargs):
        if self.joint_maps is None:
            return joints
        else:
            return torch.index_select(joints, 1, self.joint_maps)


class GMoF(nn.Module):
    def __init__(self, rho=1):
        super(GMoF, self).__init__()
        self.rho = rho

    def extra_repr(self):
        return 'rho = {}'.format(self.rho)

    def forward(self, residual):
        squared_res = residual ** 2
        dist = torch.div(squared_res, squared_res + self.rho ** 2)
        return self.rho ** 2 * dist



def project_to_img(joints, verts, faces, gt_joints, camera, image_path, img_folder, viz=False, path=None):
    exp = 1
    if len(verts) < 1:
        return
    if True:
        from core.utils.render import Renderer
        for v, (cam, gt_joint_ids, img_path) in enumerate(zip(camera, gt_joints, image_path)):
            if v > 0 and exp:
                break
            intri = np.eye(3)
            rot = cam.rotation.detach().cpu().numpy()
            trans = cam.translation.detach().cpu().numpy()
            intri[0][0] = cam.focal_length_x.detach().cpu().numpy()
            intri[1][1] = cam.focal_length_y.detach().cpu().numpy()
            intri[:2,2] = cam.center.detach().cpu().numpy()
            rot_mat = cv2.Rodrigues(rot)[0]
            
            img = cv2.imread(os.path.join(img_folder, img_path))
            render = Renderer(resolution=(img.shape[1], img.shape[0]))
            img = render.render_multiperson(verts, faces, rot_mat.copy(), trans.copy(), intri.copy(), img.copy(), viz=False)

            # for i, gt_joint in enumerate(gt_joint_ids):
            #     # if i != 0 and i != 1:
            #     #     continue
            #     color = [(0,0,255),(0,255,255),(255,0,0),(255,255,0),(255,0,255),(148,148,255)]
            #     if gt_joint is not None and True:
            #         for p in gt_joint:
            #             cv2.circle(img, (int(p[0]),int(p[1])), 3, color[i], 10)
            img_out_file = os.path.join(path, img_path)
            if not os.path.exists(os.path.dirname(img_out_file)):
                os.makedirs(os.path.dirname(img_out_file))
            cv2.imwrite(img_out_file, img)
            render.renderer.delete()
            del render
    else:
        for v, cam in enumerate(camera):
            img_dir = image_path[v]
            serial, cam_, fn = img_dir.split('\\')[-3:]
            img = cv2.imread(os.path.join(img_folder, img_dir))
            for idx, (joint, vert, gt_joint) in enumerate(zip(joints, verts, gt_joints[v])):
                if joint is None or vert is None:
                    continue
                d2j = cam(joint.unsqueeze(0)).detach().cpu().numpy().astype(np.int32)
                vert = cam(vert.unsqueeze(0)).detach().cpu().numpy().astype(np.int32)

                for f in faces:
                    color = 255
                    point = vert[0][f]
                    img = cv2.polylines(img,[point],True,(color,color,color),1)
                for p in d2j[0]:
                    cv2.circle(img, (int(p[0]),int(p[1])), 3, (0,0,255), 10)
                if gt_joint is not None:
                    for p in gt_joint:
                        cv2.circle(img, (int(p[0]),int(p[1])), 3, (0,255,0), 10)

            img_out_folder = os.path.join(path, serial, cam_)
            if not os.path.exists(img_out_folder):
                os.makedirs(img_out_folder)
            cv2.imwrite(os.path.join(img_out_folder, '%s' %fn), img)
            


def save_results(setting, data, result, dataset_obj,
                use_vposer=True,
                use_motionprior=True, 
                save_meshes=False, save_images=False, frames_seq=1,
                **kwargs):
    vposer = setting['vposer']
    models = setting['model']
    camera = setting['cameras']
    img_path = data['img_path']
    keypoints = data['keypoints']
    flags = data['flags'].T
    names = data['img_path'][0]
    model_outputs = []
    faces = models[0].faces
    # Extract model output for each person
    for idx in range(dataset_obj.num_people):
        model_results = result['person%02d' %idx]
        if use_vposer:
            pose_embedding = model_results['pose_embedding']
            body_pose = vposer.decode(
                    pose_embedding, output_type='aa').view(frames_seq, -1) if use_vposer else None
        elif use_motionprior:
            pose_embedding = model_results['pose_embedding']
            body_pose = vposer.decode(
                        pose_embedding, t=frames_seq).view(
                            frames_seq, -1)
        # the parameters of foot and hand are from vposer
        # we do not use this inaccurate results
        body_pose[:,-6:] = 0.
        model_results['body_pose'] = body_pose.detach().cpu().numpy()
        orient = np.array(models[idx].global_orient.detach().cpu().numpy())
        pose = np.hstack((orient, model_results['body_pose']))
        model_results['pose'] = pose
        model_results['pose_embedding'] = pose_embedding.detach().cpu().numpy()

        result['person%02d' %idx] = model_results

        if save_meshes or save_images:
            model_output = models[idx](return_verts=True, body_pose=body_pose)
            model_outputs.append(model_output)

    for i, name in enumerate(names):
        # name of output
        if platform.system() == 'Windows':
            serial, cam, fn = name.split('\\')[-3:]
        else:
            serial, cam, fn = name.split('/')[-3:]
        fn = fn.split('.')[0]
        frame_results = {}
        meshes, joints = [], []
        for idx in range(dataset_obj.num_people):
            model_results = result['person%02d' %idx]
            if use_vposer or use_motionprior:
                frame_result = dict(betas=model_results['betas'], scale=model_results['scale'], loss=model_results['loss'], global_orient=model_results['global_orient'][i], transl=model_results['transl'][i], pose_embedding=model_results['pose_embedding'][i], body_pose=model_results['body_pose'][i], pose=model_results['pose'][i], )
                frame_results['person%02d' %idx] = frame_result

            if save_meshes:
                import trimesh
                curr_mesh_fn = osp.join(setting['mesh_folder'], serial)
                if not osp.exists(curr_mesh_fn):
                    os.makedirs(curr_mesh_fn)
                mesh_fn = osp.join(curr_mesh_fn, '%s_%02d.obj' %(fn, idx))
                mesh = model_outputs[idx].vertices[i].detach().cpu().numpy()
                out_mesh = trimesh.Trimesh(mesh, faces, process=False)
                out_mesh.export(mesh_fn)

            if save_images:
                if flags[idx][i] < 0:
                    meshes.append(None)
                    joints.append(None)
                else:
                    meshes.append(model_outputs[idx].vertices[i])
                    joints.append(model_outputs[idx].joints[i])

        # save results
        curr_result_fn = osp.join(setting['result_folder'], serial)
        if not osp.exists(curr_result_fn):
            os.makedirs(curr_result_fn)
        result_fn = osp.join(curr_result_fn, '%s.pkl' %fn)
        with open(result_fn, 'wb') as result_file:
            pickle.dump(frame_results, result_file, protocol=2)

        # save image
        if save_images:
            img_p = [img_path[v][i] for v in range(len(data['img_path']))]
            keyp_p = [keypoints[v][i] for v in range(len(data['img_path']))]
            project_to_img(joints, meshes, faces, keyp_p, camera, img_p, dataset_obj.img_folder, viz=True, path=setting['img_folder'])


    if kwargs.get('opt_cam'):
        cam_out = os.path.join(setting['cam_folder'], serial)
        if not osp.exists(cam_out):
            os.makedirs(cam_out)
        cam_out = osp.join(cam_out, '%s.txt' %fn)
        cam_out = open(cam_out, 'w')
        for idx, (rot, trans, intri) in enumerate(zip(result['cam_rots'], result['cam_trans'], result['intris'])):
            cam_out.write(str(idx)+'\n')
            for i in intri:
                cam_out.write('%s %s %s \n' %(str(i[0]), str(i[1]), str(i[2]), ))
            cam_out.write('0 0 \n')
            extri = np.eye(4)
            rot_mat = cv2.Rodrigues(rot)[0]
            extri[:3,:3] = rot_mat
            extri[:3,3] = trans
            for i in extri[:3]:
                cam_out.write('%s %s %s %s \n' %(str(i[0]), str(i[1]), str(i[2]), str(i[3])))
            cam_out.write('\n')
        cam_out.close()

def rot_mesh(mesh, J3d, gt3d):
    G3d = gt3d.copy()
    J = J3d.copy()
    cent_J = np.mean(J, axis=0, keepdims=True)
    J -= cent_J
    cent_G = np.mean(G3d, axis=0, keepdims=True)
    G3d -= cent_G
    M = np.dot(J.T, G3d)
    U, D, V = np.linalg.svd(M) 
    R = np.dot(V.T, U.T)
    out_mesh = np.dot(mesh, R)
    out_joint = np.dot(J3d, R)
    return out_mesh, out_joint, R

def estimate_translation_np(S, joints_2d, joints_conf, fx=5000, fy=5000, cx=128., cy=128.):
    num_joints = S.shape[0]
    # focal length
    f = np.array([fx, fy])
    # optical center
   # center = np.array([img_size/2., img_size/2.])
    center = np.array([cx, cy])
    # transformations
    Z = np.reshape(np.tile(S[:,2],(2,1)).T,-1)
    XY = np.reshape(S[:,0:2],-1)
    O = np.tile(center,num_joints)
    F = np.tile(f,num_joints)
    weight2 = np.reshape(np.tile(np.sqrt(joints_conf),(2,1)).T,-1)

    # least squares
    Q = np.array([F*np.tile(np.array([1,0]),num_joints), F*np.tile(np.array([0,1]),num_joints), O-np.reshape(joints_2d,-1)]).T
    c = (np.reshape(joints_2d,-1)-O)*Z - F*XY

    # weighted least squares
    W = np.diagflat(weight2)
    Q = np.dot(W,Q)
    c = np.dot(W,c)

    # square matrix
    A = np.dot(Q.T,Q)
    b = np.dot(Q.T,c)

    # solution
    trans = np.linalg.solve(A, b)
    return trans

def IOU(A, B):
    # A[x1,y1,x2,y2]

    H = min(A[1][1], B[1][1]) - max(A[0][1], B[0][1])
    W = min(A[1][0], B[1][0]) - max(A[0][0], B[0][0])
    if W <= 0 or H <= 0:
        return 0
    SA = (A[1][1] - A[0][1]) * (A[1][0] - A[0][0])
    SB = (B[1][1] - B[0][1]) * (B[1][0] - B[0][0])
    cross = W * H
    return cross/(SA + SB - cross)

def extris_est(spin, data, data_folder, intris):
    """
    Estimating the initial camera poses via SPIN
    """
    from core.utils.module_utils import joint_projection
    img_path = data['img_path']
    keypoints = data['keypoints']
    frame_length = len(img_path[0])
    idx = 0
    threshold = 0.4
    for f in range(frame_length):
        ious = []
        extris = []
        joints = []
        joints2d = []
        verts = []
        keyps = [keypoints[v][f][idx] for v in range(len(keypoints)) if keypoints[v][f][idx][:,2].max() > 0]
        if len(keyps) < len(keypoints):
            continue
        for v, (img, keyp) in enumerate(zip(img_path, keypoints)):
            if keyp[f][idx][:,2].max() < 1e-3:
                break
            img = os.path.join(data_folder, 'images', img[f])
            keyp = keyp[f][idx]
            img, norm_img = spin.process_image(img, keyp)
            output, vert = spin(norm_img)
            joints.append(output[0].detach().cpu().numpy())
            verts.append(vert[0].detach().cpu().numpy())
            joints2d.append(keyp)
        
        if len(joints) < 1:
            continue
        ref_joints = joints[0][:-2]
        ref_mesh = verts[0]

        for i, (j3d, j2d, intri) in enumerate(zip(joints, joints2d, intris)):
            j2d = j2d[[16,14,12,11,13,15,10,8,6,5,7,9]]
            j3d = j3d[:-2]
            extri = np.eye(4)
            mesh, joints_, R = rot_mesh(ref_mesh, ref_joints, j3d)

            # temp = np.dot(np.linalg.inv(R), ref_joints.T).T
            extri[:3,:3] = R
            trans = estimate_translation_np(j3d, j2d[:,:2], j2d[:,2], intri[0][0], intri[1][1], intri[0][2], intri[1][2])
            
            trans[2] = abs(trans[2]) # prevent negative depth
            extri[:3,3] = trans
            extris.append(extri)

            proj_joints, _ = joint_projection(ref_joints, extri, intri, np.zeros((1,1)), False)

            gt_lt, gt_rb = spin.calc_aabb(j2d)
            pr_lt, pr_rb = spin.calc_aabb(proj_joints)
            ious.append(IOU([gt_lt, gt_rb], [pr_lt, pr_rb]))

            if False:
                from core.utils.render import Renderer
                # from utils.utils import joint_projection, surface_projection
                img = cv2.imread(os.path.join(data_folder, 'images', data['img_path'][i][f]))
                render = Renderer(resolution=(img.shape[1], img.shape[0]))
                img = render(ref_mesh, spin.smpl.faces, extri[:3,:3].copy(), extri[:3,3].copy(), intri.copy(), img.copy(), color=[1,1,0.9], viz=False)
                render.vis_img("img", img)
                render.renderer.delete()
                del render

        ious = np.array(ious)
        if ious.min() > threshold:
            break

    assert len(ious) > 0, "Improper threshold, turn down the threshold"
    
    if False:
        from core.utils.render import Renderer
        for i in range(len(extris)):
            # from utils.utils import joint_projection, surface_projection
            img = cv2.imread(os.path.join(data_folder, 'images', data['img_path'][i][f]))
            render = Renderer(resolution=(img.shape[1], img.shape[0]))
            img = render(ref_mesh, spin.smpl.faces, extris[i][:3,:3].copy(), extris[i][:3,3].copy(), intris[i].copy(), img.copy(), color=[1,1,0.9], viz=False)
            render.vis_img("img", img)
            render.renderer.delete()
            del render
        # joint_projection(ref_joints, extri, intri, img, True)
        # surface_projection(ref_mesh, spin.smpl.faces, ref_joints, extri, intri, img, 5)

    return extris

def get_optical_line(c_points, intra):
    if c_points.shape[1] == 3:
        c_points = c_points[:,:2]
    zero = np.array([0,0,0])
    lines = []

    f = (intra[0][0] + intra[1][1]) / 2
    trans_x = intra[0][2]
    trans_y = intra[1][2]

    for point in c_points:
        point = point - np.array([trans_x, trans_y])
        point = np.array([point[0], point[1], f])
        d = point/np.linalg.norm(point,2)
        m = np.cross(zero,d)
        lines.append([d,m])
    
    return lines

def get_joints_from_Pluecker(depth, lines):
    est_joints = []
    for d, line in zip(depth, lines):
        t = (d - line[1][2]) / line[0][2]
        x = t * line[0][0] + line[1][0]
        y = t * line[0][1] + line[1][1]
        est_joints.append([x,y,d])
    return np.array(est_joints)

def save_camparam(path, intris, extris):
    f = open(path, 'w')
    for ind, (intri, extri) in enumerate(zip(intris, extris)):
        f.write(str(ind)+'\n')
        for i in intri:
            f.write(str(i[0])+' '+str(i[1])+' '+str(i[2])+'\n')
        f.write('0 0 \n')
        for i in extri[:3]:
            f.write(str(i[0])+' '+str(i[1])+' '+str(i[2])+' '+str(i[3])+'\n')
        f.write('\n')
    f.close()

def load_camera(data, setting, data_folder, **kwarg):

    intri_path = data['camparam']
    dtype = setting['dtype']
    device = setting['device']
    # load intrinsic parameters
    if os.path.exists(intri_path):
        extris, intris = load_camera_para(intri_path)
        trans, rot = get_rot_trans(extris, photoscan=False)

    views = len(intris)
    camera = []
    for v in range(views):
        focal_length_x = float(intris[v][0][0])
        focal_length_y = float(intris[v][1][1])
        center = torch.tensor(intris[v][:2,2],dtype=dtype).unsqueeze(0)
        camera_t = create_camera(focal_length_x=focal_length_x,
                            focal_length_y=focal_length_y,
                            center=center,
                            dtype=dtype,
                            **kwarg)
        camera.append(camera_t.to(device))

    if kwarg.get('opt_cam'):
        spin = SPIN(device=device)
        extris = extris_est(spin, data, data_folder, intris)

        # save_camparam('init_cam.txt', intris, extris)
        trans, rot = get_rot_trans(extris, photoscan=False)
        for cam, R, t in zip(camera, rot, trans):
            R = torch.tensor(R, dtype=dtype).unsqueeze(0)
            R = cam.matrot2aa(R) + 1e-6
            t = torch.tensor(t, dtype=dtype).unsqueeze(0)
            cam.load_extrinsic(R, t)
            cam.translation.requires_grad = True
            cam.rotation.requires_grad = True
    else:
        for cam, R, t in zip(camera, rot, trans):
            R = torch.tensor(R, dtype=dtype).unsqueeze(0)
            R = cam.matrot2aa(R)
            t = torch.tensor(t, dtype=dtype).unsqueeze(0)
            cam.load_extrinsic(R, t)

    # from core.utils.visualization3d import Visualization
    # viz = Visualization()
    # for cam in extris:
    #     t = add_camera_mesh(cam)
    #     viz.visualize_cameras(t.T, [0.5, 0.5, 1])
    # while True:
    #     viz.show()

    setting['extrinsics'] = extris
    setting['intrinsics'] = intris
    setting['cameras'] = camera

    return setting

def add_camera_mesh(extrinsic, camerascale=1):
    # 8 points camera
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
