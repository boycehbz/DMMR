
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


import time
try:
    import cPickle as pickle
except ImportError:
    import pickle

import sys
import os

import numpy as np
import torch

from tqdm import tqdm

from collections import defaultdict

import cv2
import PIL.Image as pil_img

from core.optimizers import optim_factory

from core.utils import fitting

def non_linear_solver(
                    setting,
                    data,
                    dataset_obj,
                    batch_size=1,
                    data_weights=None,
                    body_pose_prior_weights=None,
                    kinetic_weights=None,
                    shape_weights=None,
                    coll_loss_weights=None,
                    use_joints_conf=False,
                    use_3d=False,
                    rho=100,
                    interpenetration=False,
                    loss_type='smplify',
                    visualize=False,
                    use_vposer=True,
                    use_motionprior=False,
                    interactive=True,
                    use_cuda=True,
                    is_seq=False,
                    **kwargs):
    # assert batch_size == 1, 'PyTorch L-BFGS only supports batch_size == 1'

    # views = setting['views']
    lr = kwargs.pop('lr')
    device = setting['device']
    dtype = setting['dtype']
    vposer = setting['vposer']
    keypoints = data['keypoints']
    flags = data['flags']
    joint_weights = setting['joints_weight']
    models = setting['model']
    cameras = setting['cameras']
    pose_embeddings = setting['pose_embedding']
    seq_start = setting['seq_start']
    if data['3d_joint'] is None:
        use_3d = False

    assert (len(data_weights) ==
            len(body_pose_prior_weights) and len(shape_weights) ==
            len(body_pose_prior_weights) and len(coll_loss_weights) ==
            len(body_pose_prior_weights)), "Number of weight must match"
    
    # process keypoints
    gt_joints = []
    confidence = []
    for idx in range(dataset_obj.num_people):
        gt_joints.append([[f[idx][:,:2] if f[idx] is not None else np.zeros((dataset_obj.num_joints,2)) for f in v] for v in keypoints])
        confidence.append([[f[idx][:,2] if f[idx] is not None else np.zeros((dataset_obj.num_joints,)) for f in v] for v in keypoints])
    
    gt_joints = torch.tensor(gt_joints, dtype=dtype, device=device)
    joints_conf = torch.tensor(confidence, dtype=dtype, device=device)
    flags = torch.tensor(flags, dtype=dtype, device=device)

    if use_3d: #:
        joints3d = data['3d_joint'][0]
        joints_data = torch.tensor(joints3d, dtype=dtype)
        gt_joints3d = joints_data[:, :3]
        if use_joints_conf:
            joints3d_conf = joints_data[:, 3].reshape(1, -1).to(device=device, dtype=dtype)
            if not kwargs.get('use_hip'):
                joints3d_conf[0][11] = 0
                joints3d_conf[0][12] = 0

        gt_joints3d = gt_joints3d.to(device=device, dtype=dtype)
    else:
        gt_joints3d = None
        joints3d_conf = None

    # Create the search tree
    search_tree = None
    pen_distance = None
    filter_faces = None

    # Weights used for the pose prior and the shape prior
    opt_weights_dict = {'data_weight': data_weights,
                        'body_pose_weight': body_pose_prior_weights,
                        'shape_weight': shape_weights,
                        'kinetic_weight': kinetic_weights}
    if interpenetration:
        opt_weights_dict['coll_loss_weight'] = coll_loss_weights

    # get weights for each stage
    keys = opt_weights_dict.keys()
    opt_weights = [dict(zip(keys, vals)) for vals in
                   zip(*(opt_weights_dict[k] for k in keys
                         if opt_weights_dict[k] is not None))]
    for weight_list in opt_weights:
        for key in weight_list:
            weight_list[key] = torch.tensor(weight_list[key],
                                            device=device,
                                            dtype=dtype)

    # create fitting loss
    loss = fitting.create_loss(loss_type=loss_type,
                               joint_weights=joint_weights,
                               rho=rho,
                               use_joints_conf=use_joints_conf,
                               vposer=vposer,
                               pose_embedding=pose_embeddings,
                               body_pose_prior=setting['body_pose_prior'],
                               shape_prior=setting['shape_prior'],
                               angle_prior=setting['angle_prior'],
                               interpenetration=interpenetration,
                               pen_distance=pen_distance,
                               search_tree=search_tree,
                               tri_filtering_module=filter_faces,
                               dtype=dtype,
                               use_3d=use_3d,
                               frame_length=dataset_obj.frames,
                               **kwargs)
    loss = loss.to(device=device)

    monitor = fitting.FittingMonitor(
            batch_size=batch_size, visualize=visualize, **kwargs)
    # with fitting.FittingMonitor(
    #         batch_size=batch_size, visualize=visualize, **kwargs) as monitor:


    data_weight = 500 / 1080

    # Step 1: Optimize the full model
    final_loss_val = 0
    opt_start = time.time()

    # # initial value for non-linear solve
    # new_params = defaultdict(global_orient=model.global_orient,
    #                             # body_pose=body_mean_pose,
    #                             transl=model.transl,
    #                             scale=model.scale,
    #                             betas=model.betas,
    #                             )
    # if vposer is not None:
    #     with torch.no_grad():
    #         pose_embedding.fill_(0)
    # model.reset_params(**new_params)

    for opt_idx, curr_weights in enumerate(tqdm(opt_weights, desc='Stage')):
        # pass stage1 and stage2 if it is a sequence
        if not seq_start and is_seq:
            if opt_idx < 2:
                continue
            elif opt_idx == 2:
                curr_weights['body_pose_weight'] *= 0.15

        body_params = []
        for model, pose_embedding in zip(models, pose_embeddings):
            body_param = list(model.parameters())
            body_params += list(
            filter(lambda x: x.requires_grad, body_param))
            if vposer is not None:
                body_params.append(pose_embedding)

        final_params = list(
            filter(lambda x: x.requires_grad, body_params))

        for cam in cameras:
            if cam.translation.requires_grad:
                final_params.append(cam.translation)
            if cam.rotation.requires_grad:
                final_params.append(cam.rotation)

        # if opt_idx >= 2 and use_motionprior:
        #     lr = lr * 0.1

        body_optimizer, body_create_graph = optim_factory.create_optimizer(
            final_params, lr=lr,
            **kwargs)
        body_optimizer.zero_grad()

        curr_weights['data_weight'] = data_weight
        curr_weights['bending_prior_weight'] = (
            3.17 * curr_weights['body_pose_weight'])
        loss.reset_loss_weights(curr_weights)

        closure = monitor.create_fitting_closure(
            body_optimizer, models,
            camera=cameras, gt_joints=gt_joints,
            joints_conf=joints_conf,
            flags=flags,
            gt_joints3d=gt_joints3d,
            joints3d_conf=joints3d_conf,
            joint_weights=joint_weights,
            loss=loss, create_graph=body_create_graph,
            use_vposer=use_vposer, vposer=vposer,
            use_motionprior=use_motionprior,
            pose_embeddings=pose_embeddings,
            return_verts=True, return_full_pose=True, use_3d=use_3d)

        if interactive:
            if use_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            stage_start = time.time()
        final_loss_val = monitor.run_fitting(
            body_optimizer,
            closure, final_params,
            models,
            pose_embeddings=pose_embeddings, vposer=vposer,
            use_vposer=use_vposer, use_motionprior=use_motionprior)

        if interactive:
            if use_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.time() - stage_start
            if interactive:
                tqdm.write('Stage {:03d} done after {:.4f} seconds'.format(
                    opt_idx, elapsed))

    if interactive:
        if use_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - opt_start
        tqdm.write(
            'Body fitting done after {:.4f} seconds'.format(elapsed))
        tqdm.write('Body final loss val = {:.5f}'.format(
            final_loss_val))

        result = {}
        for idx, (model, pose_embedding) in enumerate(zip(models, pose_embeddings)):
            # Get the result of the fitting process
            model_result = {key: val.detach().cpu().numpy()
                            for key, val in model.named_parameters()}
            model_result['loss'] = final_loss_val
            model_result['pose_embedding'] = pose_embedding
            result['person%02d' %idx] = model_result

        rots, trans, intris = [], [], []
        for cam in cameras:
            rots.append(cam.rotation.detach().cpu().numpy())
            trans.append(cam.translation.detach().cpu().numpy())
            intri = np.eye(3)
            intri[0][0] = cam.focal_length_x.detach().cpu().numpy()
            intri[1][1] = cam.focal_length_y.detach().cpu().numpy()
            intri[:2,2] = cam.center.detach().cpu().numpy()
            intris.append(intri)
        result['cam_rots'] = rots
        result['cam_trans'] = trans
        result['intris'] = intris 
    return result
