'''
 @FileName    : init_guess.py
 @EditTime    : 2021-12-13 13:37:50
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''
from core.utils.recompute3D import recompute3D
import torch
import numpy as np
from core.utils.umeyama import umeyama
import cv2
from core.utils.visualization3d import Visualization
from core.affinity.affinity import ComposedAffinity
from core.assignment.associate import simple_associate
from scipy import signal

def joint_interpolation(poses, n_joints):
    """ Interpolate poses to a complete motion, the empty frame is None """
    start = 0
    # If the first frame is None
    if poses[start] is None:
        poses[start] = np.zeros((n_joints, 3))
    
    for n, joint in enumerate(poses):
        if joint is not None:
            if n >= len(poses)-1:
                break
            if poses[start+1] is not None:
                start += 1
            if n != start:
                j1 = poses[start]
                start_t = start
                det = (joint - j1) / (n - start_t)
                for i in range(n - start_t):
                    poses[start] = j1 + det * i
                    start += 1
    t = poses[start]
    # If the last frame is None
    while(start<n+1):
        poses[start] = t
        start += 1

    return poses

def rec_3D_joints(keypoints, extris, intris, idx, filter_joints_idx, first_frame=False):
    keypoints = np.array(keypoints, dtype=np.float32)
    keypoints = keypoints[:,:,idx,filter_joints_idx]
    n_views, n_frames, n_joints = keypoints.shape[:3]

    joints = []
    for f in range(n_frames):
        if first_frame and f > 0:
            break
        # Filter out unreliable detection
        pack = [[keypoints[v][f], extris[v], intris[v]] for v in range(n_views) if keypoints[v][f][:,2].max() > 0.2]
        if len(pack) < 2: # Do not process single view case
            joints.append(None)
            continue
        keps = np.array([p[0] for p in pack])
        cam_extris = np.array([p[1] for p in pack])
        cam_intris = np.array([p[2] for p in pack])
        rec_joints3d = recompute3D(cam_extris, cam_intris, keps.copy())
        joints.append(rec_joints3d)
    
    # Interpolation
    joints = joint_interpolation(joints, n_joints)

    return joints

def physics_geometry_filter(keypoints, extris, intris, frames_seq, flags, img_paths, dataset_obj, filter_joints_idx):
    """
    Filter out the noisy detection and recompute 3D joints using the filtered keypoints
    """
    # Calculate the joints in first frame
    last_js = []
    for idx in range(dataset_obj.num_people):
        rec_joints3d = rec_3D_joints(keypoints, extris, intris, idx, filter_joints_idx, first_frame=True)
        last_js.append(rec_joints3d[0])
    
    joints = []
    affinity_model = ComposedAffinity(cameras=[extris, intris])
    n_views = len(extris)
    n_people = dataset_obj.num_people
    n_joints = len(filter_joints_idx)
    total_n_joints = dataset_obj.num_joints
    Pall = np.array([intri @ extri[:3]  for extri, intri in zip(extris, intris)])
    # Container to save filtered results
    filterd_keypoints = [[[np.zeros((total_n_joints,3)) for n in range(n_people)] for f in range(frames_seq)] for v in range(n_views)]

    last_2d = [[keypoints[v][0][idx] for v in range(n_views)] for idx in range(n_people)]

    total_joints = []
    for i in range(frames_seq):
        keyps = [keypoints[v][i] for v in range(n_views)]
        joint = []
        for idx, last_j in enumerate(last_js):
            # Filter
            affinity, dimGroups = affinity_model(keyps, None, last_2d[idx], last_j, images=img_paths)
            keyps, output = simple_associate(keyps, affinity, dimGroups, Pall, idx)

            # Recompute 3D joints from the filtered keypoints and the initial cameras
            pack = [[k, extris[i], intris[i]] for i, k in enumerate(output) if k is not None]
            if len(pack) < 2: # do not process single view case
                joint.append(None)
                continue
            keps = np.array([p[0][filter_joints_idx] for p in pack])
            cam_extris = np.array([p[1] for p in pack])
            cam_intris = np.array([p[2] for p in pack])
            rec_joints3d = recompute3D(cam_extris, cam_intris, keps.copy())
            joint.append(rec_joints3d)

            # Save the filtered keypoints
            for v in range(n_views):
                filterd_keypoints[v][i][idx] = output[v] if output[v] is not None else np.zeros((total_n_joints,3))
                if output[v] is not None:
                    last_2d[idx][v] = output[v]
            if len(cam_extris) > 2:
                last_js[idx] = rec_joints3d
        total_joints.append(joint)

    # Interpolation
    interpolated_joints = []
    for idx in range(n_people):
        joints = [j[idx] for j in total_joints]
        joints = joint_interpolation(joints, n_joints)
        interpolated_joints.append(np.array(joints))

    return filterd_keypoints, flags, interpolated_joints


def init_guess(setting, data, dataset_obj, frames_seq=1, use_torso=False, **kwargs):
    models = setting['model']
    dtype = setting['dtype']
    keypoints = data['keypoints']
    flags = data['flags']
    device = setting['device']
    est_scale = not setting['fix_scale']
    fixed_scale = 1. if setting['fixed_scale'] is None else setting['fixed_scale']
    extris = setting['extrinsics']
    intris = setting['intrinsics']
    # The joints that are used for calculating consistency
    # (LS,RS,LE,RE,LW,RW,LH,RH,LK,RK,LA,RA)
    filter_joints_idx = [5,6,7,8,9,10,11,12,13,14,15,16]

    # Step1: Get initial joint positions for SMPL model
    init_SMPL_joints = []
    for idx in range(dataset_obj.num_people):
        # Reset the SMPL model with initial parameters 
        init_t = torch.zeros((frames_seq, 3), dtype=dtype)
        init_r = torch.zeros((frames_seq, 3), dtype=dtype)
        init_s = torch.tensor(fixed_scale, dtype=dtype)
        init_shape = torch.zeros((1, 10), dtype=dtype)
        models[idx].reset_params(transl=init_t, global_orient=init_r, scale=init_s, betas=init_shape)
        init_pose = torch.zeros((frames_seq, 69), dtype=dtype, device=device)
        with torch.no_grad():
            model_output = models[idx](return_verts=False, return_full_pose=False, body_pose=init_pose)
            output_joints = model_output.joints.cpu().numpy()
        init_SMPL_joints.append(output_joints)
    
    # Step2: Get reconstructed joint positions from 2D detections and inital cameras
    use_filter = False
    if use_filter:
        keypoints, flags, init_rec_joints = physics_geometry_filter(keypoints, extris, intris, frames_seq, flags, data['img_path'], dataset_obj, filter_joints_idx)
    else:
        init_rec_joints = []
        for idx in range(dataset_obj.num_people):
            # Recompute 3D joints with the initial cameras
            rec_joints3ds = rec_3D_joints(keypoints, extris, intris, idx, filter_joints_idx)
            init_rec_joints.append(np.array(rec_joints3ds))

    # Step3: Align the SMPL models to the reconstructed joints
    for idx in range(dataset_obj.num_people):
        rec_joints3ds = init_rec_joints[idx]
        model_joint3ds = init_SMPL_joints[idx]
        rotations, translations = [], []

        # Filter out noisy reconstrction with Butterworth Filter
        b, a = signal.butter(3, 0.05, 'lowpass')
        filtered_joints = signal.filtfilt(b, a, rec_joints3ds.T).T.copy()

        # Calculate the global rotations and translations for SMPL models
        for joints, joints3d in zip(model_joint3ds, filtered_joints):
            # We align the SMPL to the joints in torso (L_Shoulder, R_Shoulder, L_Hip, R_Hip) 
            joints = joints[[5,6,11,12]]
            joints3d = joints3d[[0,1,6,7]]

            if abs(joints3d).max() < 0.1:
                rotations.append(np.zeros((3,)))
                translations.append(np.zeros((3,)))
                continue
                
            # get transformation
            rot, trans, scale = umeyama(joints, joints3d, est_scale)
            rot = cv2.Rodrigues(rot)[0].reshape(3,)
            rotations.append(rot)
            translations.append(trans)

        # Apply the estimated results
        if est_scale:
            init_s = torch.tensor(scale, dtype=dtype)
        else:
            init_s = torch.tensor(fixed_scale, dtype=dtype)
        init_t = torch.tensor(translations, dtype=dtype)
        init_r = torch.tensor(rotations, dtype=dtype)
        models[idx].reset_params(transl=init_t, global_orient=init_r, scale=init_s)

        if kwargs.get('use_vposer') or kwargs.get('use_motionprior'):
            with torch.no_grad():   
                setting['pose_embedding'][idx].fill_(0)

        # Visualize the initialized results
        if False:
            import os
            from core.utils.render import Renderer

            if kwargs.get('use_vposer'):
                vposer = setting['vposer']
                init_pose = vposer.decode(
                    setting['pose_embedding'][idx], output_type='aa').view(
                        frames_seq, -1)
            elif kwargs.get('use_motionprior'):
                vposer = setting['vposer']
                init_pose = vposer.decode(
                    setting['pose_embedding'][idx], t=setting['pose_embedding'][idx].shape[0]).view(
                        setting['pose_embedding'][idx].shape[0], -1)
            else:
                init_pose = torch.zeros((frames_seq, 69), dtype=dtype, device=device)

            model_output = models[idx](return_verts=True, return_full_pose=True, body_pose=init_pose)
            for i, (joints, verts) in enumerate(zip(model_output.joints.detach().cpu().numpy(), model_output.vertices.detach().cpu().numpy())):
                for v in range(1):
                    img = cv2.imread(os.path.join(dataset_obj.img_folder, data['img_path'][v][i]))
                    render = Renderer(resolution=(img.shape[1], img.shape[0]))
                    img = render(verts, models[idx].faces, extris[v][:3,:3].copy(), extris[v][:3,3].copy(), intris[v].copy(), img.copy(), color=[1,1,0.9], viz=False)
                    render.vis_img("img", img)
                    render.renderer.delete()
                    del render

    del model_output
    torch.cuda.empty_cache()
    data['keypoints'] = keypoints
    data['flags'] = flags
    return data

def fix_params(setting, scale=None, shape=None):
    """
    Use the fixed shape and scale parameters.
    """
    dtype = setting['dtype']
    models = setting['model']
    for model in models:
        init_t = model.transl
        init_r = model.global_orient
        init_s = model.scale
        init_shape = model.betas
        if scale is not None:
            init_s = torch.tensor(scale, dtype=dtype)
            model.scale.requires_grad = False
        if shape is not None:
            init_shape = torch.tensor(shape, dtype=dtype)
            model.betas.requires_grad = False

        model.reset_params(transl=init_t, global_orient=init_r, scale=init_s, betas=init_shape)


def appearance_align(data, dataset_obj):
    '''
    We assume the first frame is aligned.
    '''
    torso = [i for i in range(26)] #[5,6,11,12,18,19]
    n_views = len(data['img_path'])
    appearance_buffer = []
    appearances = data['appearances']
    keypoints = data['keypoints']
    aligned_keypoints = [[[None for n in range(dataset_obj.num_people)] for f in range(dataset_obj.frames)] for v in range(n_views)]
    aligned_appearances= [[[None for n in range(dataset_obj.num_people)] for f in range(dataset_obj.frames)] for v in range(n_views)]

    # Load initial appearance for each person based on the first frame.
    for idx in range(dataset_obj.num_people):
        app_t = []
        for v in range(n_views):
            if appearances[v][0][idx] is not None:
                app_t.append(appearances[v][0][idx])
        app_t = np.array(app_t)
        app_t = np.mean(app_t, axis=0)
        appearance_buffer.append(app_t)
    appearance_buffer = np.array(appearance_buffer)

    # Align each frame
    for f in range(dataset_obj.frames):
        if f == 105:
            print(1)
        for v in range(n_views):
            app_t = [[i, app] for i, app in enumerate(appearances[v][f]) if app is not None]
            if len(app_t) < 1:
                continue
            apps = np.array([ap[1] for ap in app_t])[None,:] * 100
            temp_buffer = appearance_buffer.copy()[:,None] * 100
            loss = np.linalg.norm(apps - temp_buffer, axis=-1)

            if f > 0:
                kep_lt = np.ones((dataset_obj.num_people, len(torso), 3)) * 100
                non_idx = []
                for i, kep in enumerate(aligned_keypoints[v][f-1]):
                    if kep is not None:
                        kep_lt[i] = kep[torso]
                    else:
                        non_idx.append(i)
                #kep_lt = [[i, kep] for i, kep in enumerate(keypoints[v][f-1]) if kep is not None]
                kep_t = [[i, kep[torso]] for i, kep in enumerate(keypoints[v][f]) if kep is not None]

                keplt = kep_lt[:,None]
                kept = np.array([kep[1] for kep in kep_t])[None,:]

                loss_kp = np.linalg.norm(keplt[...,:2] - kept[...,:2], axis=-1)
                conf = np.sqrt(keplt[...,-1] * kept[...,-1])
                loss_kp = (loss_kp * conf).sum(axis=-1)/conf.sum(axis=-1)
                # if len(non_idx) > 0:
                #     loss_kp[non_idx] = 0
                # loss = loss + loss_kp

            for igt, gt in enumerate(appearance_buffer):
                if loss[igt].min() > 100:
                    continue
                bestid = np.argmin(loss[igt])
                loss[:,bestid] = 1000
                if f > 0 and aligned_keypoints[v][f-1][igt] is not None:
                    dis = np.linalg.norm(aligned_keypoints[v][f-1][igt][:,:2] - keypoints[v][f][app_t[bestid][0]][:,:2], axis=-1)
                    conf = np.sqrt(aligned_keypoints[v][f-1][igt][:,-1] * keypoints[v][f][app_t[bestid][0]][:,-1])
                    dis = (dis * conf).sum(axis=-1)/conf.sum(axis=-1)
                    if dis > 100:
                        continue
                    aligned_keypoints[v][f][igt] = keypoints[v][f][app_t[bestid][0]]
                    aligned_appearances[v][f][igt] = app_t[bestid][1]
                else:
                    aligned_keypoints[v][f][igt] = keypoints[v][f][app_t[bestid][0]]
                    aligned_appearances[v][f][igt] = app_t[bestid][1]

        # Update appearance
        for idx in range(dataset_obj.num_people):
            app_t = []
            for v in range(n_views):
                if aligned_appearances[v][f][idx] is not None:
                    app_t.append(aligned_appearances[v][f][idx])
            if len(app_t) < 1:
                continue
            app_t = np.array(app_t)
            app_t = np.mean(app_t, axis=0)
            appearance_buffer[idx] = appearance_buffer[idx] * 0.9 + app_t * 0.1
    
    data['keypoints'] = aligned_keypoints
    data['appearances'] = aligned_appearances

    return data