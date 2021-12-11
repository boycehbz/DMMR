from networkx.algorithms.traversal import beamsearch
from core.utils.recompute3D import recompute3D
import torch
import numpy as np
from core.utils.umeyama import umeyama
import cv2
from collections import defaultdict
from core.utils.module_utils import cal_trans, get_optical_line, get_joints_from_Pluecker
from core.utils.visualization3d import Visualization
from core.affinity.affinity import ComposedAffinity
from core.assignment.associate import simple_associate

def physical_based_filter1(keypoints, extris, intris, idx, frames_seq, flags, img_paths):
    '''
    Remove low-frequency identity error via reprojection
    '''
    from core.utils.utils import joint_projection
    import os
    # vis = Visualization()
    joints = []
    losses = []
    threshold = 1000
    view = 0
    for f in range(frames_seq):
        pack = [[k[f][idx], extris[i], intris[i], img_paths[i][f], i] for i, k in enumerate(keypoints) if (k[f][idx] is not None and k[f][idx][:,2].max() > 0.1)]
        if len(pack) < 2: # do not process single view case
            for v in range(len(extris)):
                keypoints[v][f][idx] = None
            joints.append(None)
            continue
        keps = np.array([p[0] for p in pack])
        cam_extris = np.array([p[1] for p in pack])
        cam_intris = np.array([p[2] for p in pack])
        rec_joints3d = recompute3D(cam_extris, cam_intris, keps)
        
        # Calculate reprojection loss
        loss = 0
        for p in pack:
            proj_joints, _ = joint_projection(rec_joints3d, p[1], p[2], np.zeros((1,1)), viz=False)
            loss += (np.linalg.norm(proj_joints - p[0][:,:2], axis=1) * p[0][:,2]).sum()
        loss = loss/len(pack)
        # print(loss)

        # Greater than threshold
        if loss > threshold:
            if len(pack) > 2:
                flag = 0
                t = []
                for i in range(len(pack)):
                    # Remove a view that may have identity error and calculate reprojection error
                    new_pack = [pack[j] for j in range(len(pack)) if j != i]
                    keps = np.array([p[0] for p in new_pack])
                    cam_extris = np.array([p[1] for p in new_pack])
                    cam_intris = np.array([p[2] for p in new_pack])
                    new_joints3d = recompute3D(cam_extris, cam_intris, keps)
                    loss_t = 0
                    for p in new_pack:
                        proj_joints, _ = joint_projection(new_joints3d, p[1], p[2], np.zeros((1,1)), viz=False)
                        loss_t += (np.linalg.norm(proj_joints - p[0][:,:2], axis=1) * p[0][:,2]).sum()
                    # loss_t = loss_t/len(pack)
                    t.append([new_pack, loss_t])
                t = sorted(t, key=lambda x:x[1])
                if t[0][1] < threshold:
                    flag = 1
                if flag: # if the loss less than threshold, set incorrect view to None
                    losses.append(loss_t)
                    joints.append(new_joints3d)
                    index = [p[4] for p in new_pack]
                    for v in range(len(extris)):
                        if v not in index:
                            keypoints[v][f][idx] = None
                    # img = cv2.imread(os.path.join('E:/Experiments_3DV2021/test_filter/images', img_paths[view][f]))
                    # joint_projection(new_joints3d, extris[view], intris[view], img, viz=True)

                else:
                    # set all views to None
                    for v in range(len(extris)):
                        keypoints[v][f][idx] = None
                    flags[f][idx] = 0
                    joints.append(None)
                continue
            else:
                for v in range(len(extris)):
                    keypoints[v][f][idx] = None
                flags[f][idx] = 0
                joints.append(None)
                continue

        losses.append(loss)
        # print(losses[-1])
        joints.append(rec_joints3d)
        
        # img = cv2.imread(os.path.join('E:/Experiments_3DV2021/test_filter/images', img_paths[view][f]))
        # joint_projection(rec_joints3d, extris[view], intris[view], img, viz=True)

    # interpolation
    start = 0
    end = 0
    if joints[start] is None:
        joints[start] = np.zeros((17,3))
    for n, joint in enumerate(joints):
        if joint is not None:
            if n >= len(joints)-1:
                break
            if joints[start+1] is not None:
                start += 1
            if n != start:
                j1 = joints[start]
                start_t = start
                det = (joint - j1) / (n - start_t)
                for i in range(n - start_t):
                    joints[start] = j1 + det * i
                    start += 1

        #     else:
        #         end += 1
        #         # start += 1
        # else:
        #     end += 1
        #     pass
    # vis.visualize_joints(np.array(joints))
    return keypoints, flags, joints

def align_by_pelvis(joints, get_pelvis=False, format='lsp'):
    """
    Assumes joints is 14 x 3 in LSP order.
    Then hips are: [3, 2]
    Takes mid point of these points, then subtracts it.
    """
    if format == 'lsp':
        left_id = 3
        right_id = 2

        pelvis = (joints[left_id, :] + joints[right_id, :]) / 2.
    elif format in ['smpl', 'h36m']:
        pelvis_id = 0
        pelvis = joints[pelvis_id, :]
    elif format in ['mpi']:
        pelvis_id = 14
        pelvis = joints[pelvis_id, :]
    elif format == 'coco17':
        left_id = 11
        right_id = 12

        pelvis = (joints[left_id, :] + joints[right_id, :]) / 2.

    if get_pelvis:
        return joints - np.expand_dims(pelvis, axis=0), pelvis
    else:
        return joints - np.expand_dims(pelvis, axis=0)

def compute_similarity_transform(S1, S2):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))

    # 7. Error:
    S1_hat = scale*R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat

def combinations(iterable, r):
    # combinations('ABCD', 2) --> AB AC AD BC BD CD
    # combinations(range(4), 3) --> 012 013 023 123
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        yield tuple(pool[i] for i in indices)


def physics_geometry_loss(joints3d, joints_last, extris, intris, keps, f, last_t):
    '''
    @joints3d: reconstructed joints
    @joints_last: joints in last time
    @extris: camera extrinsic paeameters
    @intris: camera intrinsic paeameters
    @keps: 2D keypoints
    @f: frame id
    @last_t: id of last frame
    '''
    loss = 0.
    # geometry loss
    geo_loss = 0.
    homogeneous_j = np.insert(joints3d, 3, 1, axis=1)
    confs = np.ones((joints3d.shape[0],))
    for extri, intri, kep in zip(extris, intris, keps):
        conf = kep[:,2]
        confs = confs * conf
        lines = np.array(get_optical_line(kep, intri))
        cam_j = (np.dot(extri, homogeneous_j.T).T)[:,:3]
        dis = np.cross(cam_j, lines[:,0]) - lines[:,1]
        geo_loss += (np.linalg.norm(dis, axis=1) * conf).sum()
    geo_loss = geo_loss / len(extris)
    # kinetic energy
    kin_loss = 0.
    confs = confs ** (1/len(extris))
    if f - last_t < 3 and joints_last[1] > 2:
        kin_loss = (((np.linalg.norm(joints3d - joints_last[0], axis=1)**2) * confs).sum() / confs.sum()) * 30
    loss = geo_loss + kin_loss

    return loss, geo_loss, kin_loss

def physical_based_filter(keypoints, extris, intris, idx, frames_seq, flags, img_paths, filter_joints_idx):
    '''
    Remove low-frequency identity error via physics-geometry consistency
    '''
    from core.utils.utils import joint_projection
    import os
    # vis = Visualization()
    # filter_joints_idx = [5,6,7,8,9,10,11,12,13,14,15,16]
    joints = []
    losses = []
    joints_last = [None, 0] # joints and num view
    last_t = 0
    threshold = 3  #Shelf GT: 3
    for f in range(frames_seq):
        if f == 392:
            print(1)
        pack = [[k[f][idx][filter_joints_idx], extris[i], intris[i], img_paths[i][f], i] for i, k in enumerate(keypoints) if (k[f][idx] is not None and k[f][idx][:,2].max() > 0.2)]
        if len(pack) < 2: # do not process single view case
            # for v in range(len(extris)):
            #     keypoints[v][f][idx] = None
            joints.append(None)
            continue
        keps = np.array([p[0] for p in pack])
        cam_extris = np.array([p[1] for p in pack])
        cam_intris = np.array([p[2] for p in pack])
        rec_joints3d = recompute3D(cam_extris, cam_intris, keps.copy())
        
        if joints_last[0] is None:
            last_t = 0
            joints_last[0] = rec_joints3d
            joints_last[1] = len(cam_extris)

        loss, geo_loss, kin_loss = physics_geometry_loss(rec_joints3d, joints_last, cam_extris, cam_intris, keps, f, last_t)

        # if f == 125:
        #     vis.visualize_points(rec_joints3d, [0,0,1])
        #     vis.visualize_points(joints_last[0], [1,0,0])
        #     while True:
        #         vis.show()

        # Greater than threshold
        if loss > threshold:
            if len(pack) > 2:
                flag = 0
                t = []
                # remove 1 view
                for i in range(len(pack)):
                    # Remove a view that may have identity error and calculate reprojection error
                    new_pack = [pack[j] for j in range(len(pack)) if j != i]
                    keps = np.array([p[0] for p in new_pack])
                    cam_extris = np.array([p[1] for p in new_pack])
                    cam_intris = np.array([p[2] for p in new_pack])
                    new_joints3d = recompute3D(cam_extris, cam_intris, keps.copy())

                    loss_t, geo_loss, kin_loss = physics_geometry_loss(new_joints3d, joints_last, cam_extris, cam_intris, keps, f, last_t)

                    t.append([new_pack, loss_t, new_joints3d])
                t = sorted(t, key=lambda x:x[1])

                if t[0][1] < threshold:
                    flag = 1
                else: # Can not find a satisfied result when remove one view
                    t = []
                    # combination all pairs
                    for i in range(2, len(pack)):
                        new_packs = combinations(pack, i)
                        for new_pack in new_packs:
                            keps = np.array([p[0] for p in new_pack])
                            cam_extris = np.array([p[1] for p in new_pack])
                            cam_intris = np.array([p[2] for p in new_pack])
                            new_joints3d = recompute3D(cam_extris, cam_intris, keps.copy())

                            loss_t, geo_loss, kin_loss = physics_geometry_loss(new_joints3d, joints_last, cam_extris, cam_intris, keps, f, last_t)

                            # loss_t = 0.
                            # # distance loss
                            # homogeneous_j = np.insert(new_joints3d, 3, 1, axis=1)
                            # for extri, intri, kep in zip(cam_extris, cam_intris, keps):
                            #     conf = kep[:,2]
                            #     lines = np.array(get_optical_line(kep, intri))
                            #     cam_j = (np.dot(extri, homogeneous_j.T).T)[:,:3]
                            #     dis = np.cross(cam_j, lines[:,0]) - lines[:,1]
                            #     loss_t += (np.linalg.norm(dis, axis=1) * conf).sum()
                            # loss_t = loss_t / len(extris)
                            # # kinetic energy
                            # if f - last_t < 5:
                            #     loss_t += (np.linalg.norm(new_joints3d - joints_last, axis=1)**2).sum() * 100

                            t.append([new_pack, loss_t, new_joints3d])
                    t = sorted(t, key=lambda x:x[1])

                if t[0][1] < threshold:
                    flag = 1
                if flag: # if the loss less than threshold, set incorrect view to None
                    # select the results that have maximum view and minimum loss
                    m_view = len(t[0][0])
                    t_final = t[:1]
                    for t_ in t:
                        if t_[1] > threshold:
                            break
                        if len(t_[0]) > m_view and t_[1] < threshold:
                            t_final = [t_]
                    t = t_final
                    losses.append(t[0][1])
                    joints.append(t[0][2])
                    index = [p[4] for p in t[0][0]]
                    joints_last[0] = t[0][2]
                    joints_last[1] = len(t[0][0])
                    last_t = f
                    print(t[0][1])
                    for v in range(len(extris)):
                        if v not in index:
                            keypoints[v][f][idx] = None

                    # img = cv2.imread(os.path.join('E:/Evaluations_3DV2021/Eval_Shelf/images', img_paths[view][f]))
                    # joint_projection(t[0][2], extris[view], intris[view], img, viz=True)

                # Can not find satisfied results
                else:
                    print(t[0][1])
                    # set all views to None
                    for v in range(len(extris)):
                        keypoints[v][f][idx] = None
                    # flags[f][idx] = 0
                    joints.append(None)
                continue
            else:
                for v in range(len(extris)):
                    keypoints[v][f][idx] = None
                # flags[f][idx] = 0
                joints.append(None)
                continue

        print(loss)

        joints_last[0] = rec_joints3d
        joints_last[1] = len(cam_extris)
        last_t = f
        joints.append(rec_joints3d)
        
        # img = cv2.imread(os.path.join('E:/Evaluations_3DV2021/Eval_Shelf/images', img_paths[view][f]))
        # joint_projection(rec_joints3d, extris[view], intris[view], img, viz=True)

    
    for i in range(len(joints)):
        if joints[i] is None:
            flags[i][idx] = -1
        else:
            break
    for i in range(len(joints)):
        if joints[len(joints)-i-1] is None:
            flags[len(joints)-i-1][idx] = -1
        else:
            break
    # interpolation
    start = 0
    end = 0
    if joints[start] is None:
        joints[start] = np.zeros((len(filter_joints_idx),3))
    for n, joint in enumerate(joints):
        if joint is not None:
            if n >= len(joints)-1:
                break
            if joints[start+1] is not None:
                start += 1
            if n != start:
                j1 = joints[start]
                start_t = start
                det = (joint - j1) / (n - start_t)
                for i in range(n - start_t):
                    joints[start] = j1 + det * i
                    start += 1
    t = joints[start]
    while(start<n+1):
        joints[start] = t
        start += 1
        #     else:
        #         end += 1
        #         # start += 1
        # else:
        #     end += 1
        #     pass
    # vis.visualize_joints(np.array(joints))
    return keypoints, flags, joints

def rec_3D_joints(keypoints, extris, intris, idx, frames_seq, img_paths, filter_joints_idx):
    # vis = Visualization()
    joints = []
    for f in range(frames_seq):
        pack = [[k[f][idx][filter_joints_idx], extris[i], intris[i], img_paths[i][f], i] for i, k in enumerate(keypoints) if (k[f][idx] is not None and k[f][idx][:,2].max() > 0.2)]
        if len(pack) < 2: # do not process single view case
            # for v in range(len(extris)):
            #     keypoints[v][f][idx] = None
            joints.append(None)
            continue
        keps = np.array([p[0] for p in pack])
        cam_extris = np.array([p[1] for p in pack])
        cam_intris = np.array([p[2] for p in pack])
        rec_joints3d = recompute3D(cam_extris, cam_intris, keps.copy())
        joints.append(rec_joints3d)
    
    # interpolation
    start = 0
    end = 0
    if joints[start] is None:
        joints[start] = np.zeros((12,3))
    for n, joint in enumerate(joints):
        if joint is not None:
            if n >= len(joints)-1:
                break
            if joints[start+1] is not None:
                start += 1
            if n != start:
                j1 = joints[start]
                start_t = start
                det = (joint - j1) / (n - start_t)
                for i in range(n - start_t):
                    joints[start] = j1 + det * i
                    start += 1
    t = joints[start]
    while(start<n+1):
        joints[start] = t
        start += 1

    return joints

def physical_geometry_filter(keypoints, extris, intris, frames_seq, flags, img_paths, dataset_obj, last_js, filter_joints_idx):
    joints = []
    affinity_model = ComposedAffinity(cameras=[extris, intris])
    n_views = len(extris)
    n_people = dataset_obj.num_people
    Pall = np.array([intri @ extri[:3]  for extri, intri in zip(extris, intris)])
    filterd_keypoints = [[[None for n in range(n_people)] for f in range(frames_seq)] for v in range(n_views)]

    last_2d = [[keypoints[v][0][idx] for v in range(n_views)] for idx in range(n_people)]

    total_joints = []
    for i in range(frames_seq):
        if i == 576:
            print(1)
        keyps = [keypoints[v][i] for v in range(n_views)]
        joint = []
        for idx, last_j in enumerate(last_js):
            affinity, dimGroups = affinity_model(keyps, None, last_2d[idx], last_j, images=img_paths)
            keyps, output = simple_associate(keyps, affinity, dimGroups, Pall, idx)

            pack = [[k, extris[i], intris[i]] for i, k in enumerate(output) if k is not None]
            if len(pack) < 2: # do not process single view case
                # for v in range(len(extris)):
                #     keypoints[v][f][idx] = None
                joint.append(None)
                continue
            keps = np.array([p[0][filter_joints_idx] for p in pack])
            cam_extris = np.array([p[1] for p in pack])
            cam_intris = np.array([p[2] for p in pack])
            rec_joints3d = recompute3D(cam_extris, cam_intris, keps.copy())

            for v in range(n_views):
                filterd_keypoints[v][i][idx] = output[v]
                if output[v] is not None:
                    last_2d[idx][v] = output[v]
            joint.append(rec_joints3d)
            if len(cam_extris) > 2:
                last_js[idx] = rec_joints3d
        total_joints.append(joint)

    interpolated_joints = []
    for idx in range(n_people):
        joints = [j[idx] for j in total_joints]

        # interpolation
        start = 0
        end = 0
        if joints[start] is None:
            joints[start] = np.zeros((26,3))
        for n, joint in enumerate(joints):
            if joint is not None:
                if n >= len(joints)-1:
                    break
                if joints[start+1] is not None:
                    start += 1
                if n != start:
                    j1 = joints[start]
                    start_t = start
                    det = (joint - j1) / (n - start_t)
                    for i in range(n - start_t):
                        joints[start] = j1 + det * i
                        start += 1
        t = joints[start]
        while(start<n+1):
            joints[start] = t
            start += 1
        
        interpolated_joints.append(joints)

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
    filter_joints_idx = [5,6,7,8,9,10,11,12,13,14,15,16]
    init_joints = []
    # # calcul first frame joints
    # for idx in range(dataset_obj.num_people):
    #     pack = [[k[0][idx][filter_joints_idx], extris[i], intris[i]] for i, k in enumerate(keypoints) if k[0][idx] is not None]
    #     if len(pack) < 2: # do not process single view case
    #         init_joints.append(None)
    #         continue
    #     keps = np.array([p[0] for p in pack])
    #     cam_extris = np.array([p[1] for p in pack])
    #     cam_intris = np.array([p[2] for p in pack])
    #     rec_joints3d = recompute3D(cam_extris, cam_intris, keps.copy())
    #     init_joints.append(rec_joints3d)

    # keypoints, flags, init_rec_joints = physical_geometry_filter(keypoints, extris, intris, frames_seq, flags, data['img_path'], dataset_obj, init_joints, filter_joints_idx)


    for idx in range(dataset_obj.num_people):
        # if idx == 0:
        #     continue
        # reset model
        init_t = torch.zeros((frames_seq, 3), dtype=dtype)
        init_r = torch.zeros((frames_seq, 3), dtype=dtype)
        init_s = torch.tensor(fixed_scale, dtype=dtype)
        init_shape = torch.zeros((1, 10), dtype=dtype)
        models[idx].reset_params(transl=init_t, global_orient=init_r, scale=init_s, betas=init_shape)

        init_pose = torch.zeros((frames_seq, 69), dtype=dtype, device=device)
        with torch.no_grad():
            model_output = models[idx](return_verts=False, return_full_pose=False, body_pose=init_pose)
            output_joints = model_output.joints.cpu().numpy()

        model_joint3ds = []
        rec_joints3ds = []
        rotations = []
        translations = []
        rec_joints3d = None

        if not kwargs.get('single_view'):
            if False:
                rec_joints3ds = init_rec_joints[idx]
                model_joint3ds = output_joints

                # keypoints, flags, rec_joints3ds = physical_based_filter(keypoints, extris, intris, idx, frames_seq, flags, data['img_path'], filter_joints_idx)
                # model_joint3ds = output_joints
            else:
                rec_joints3ds = rec_3D_joints(keypoints, extris, intris, idx, frames_seq, data['img_path'], filter_joints_idx)
                model_joint3ds = output_joints
        else:
            for f in range(frames_seq):
                pack = [[k[f][idx][filter_joints_idx], extris[i], intris[i], i] for i, k in enumerate(keypoints) if k[f][idx] is not None]

                if len(pack) < 1: # The first frame and without keypoints
                    rec_joints3ds.append(None)
                elif (len(pack) == 1 and rec_joints3d is None) or (len(data['img_path']) == 1 and len(pack) == 1): # The first frame with single view
                    torso3d = output_joints[f][[5,6,11,12]]
                    torso2d = pack[0][0][[0,1,6,7]]
                    torso3d = np.insert(torso3d, 3, 1, axis=1).T
                    torso3d = (np.dot(pack[0][1], torso3d).T)[:,:3]

                    diff3d = np.array([torso3d[0] - torso3d[2], torso3d[1] - torso3d[3]])
                    mean_height3d = np.mean(np.sqrt(np.sum(diff3d**2, axis=1)))

                    diff2d = np.array([torso2d[0] - torso2d[2], torso2d[1] - torso2d[3]])
                    mean_height2d = np.mean(np.sqrt(np.sum(diff2d**2, axis=1)))

                    est_d = pack[0][2][0][0] * (mean_height3d / mean_height2d)

                    # just set the z value
                    cam_joints = np.dot(pack[0][1], np.insert(output_joints[f].copy(), 3, 1, axis=1).T)
                    cam_joints[2,:] += est_d

                    lines = get_optical_line(pack[0][0][:,:2], pack[0][2])
                    rec_joints3d = get_joints_from_Pluecker(cam_joints[2], lines)
                    rec_joints3d = (np.dot(np.linalg.inv(pack[0][1]), np.insert(rec_joints3d, 3, 1, axis=1).T).T)[:,:3]
                else:
                    rec_joints3d = None

                # # test
                # import os
                # from utils.utils import joint_projection, surface_projection
                # img = cv2.imread(os.path.join(dataset_obj.img_folder, data['img_path'][0][f]))
                # joint_projection(rec_joints3d, pack[0][1], pack[0][2], img, True)

                if False:
                    rec_joints3ds.append(rec_joints3d[[5,6,11,12]])
                    model_joint3ds.append(output_joints[f,[5,6,11,12]])
                else:
                    rec_joints3ds.append(rec_joints3d)
                    model_joint3ds.append(output_joints[f])


            # interpolation
            start = 0
            end = 0
            if rec_joints3ds[start] is None:
                rec_joints3ds[start] = np.zeros((26,3))
            for n, joint in enumerate(rec_joints3ds):
                if joint is not None:
                    if n >= len(rec_joints3ds)-1:
                        break
                    if rec_joints3ds[start+1] is not None:
                        start += 1
                    if n != start:
                        j1 = rec_joints3ds[start]
                        start_t = start
                        det = (joint - j1) / (n - start_t)
                        for i in range(n - start_t):
                            rec_joints3ds[start] = j1 + det * i
                            start += 1
            t = rec_joints3ds[start]
            while(start<n+1):
                rec_joints3ds[start] = t
                start += 1

        model_joint3ds = np.array(model_joint3ds)
        rec_joints3ds = np.array(rec_joints3ds)
        f_length, n_joint, dim = rec_joints3ds.shape
        # rotations[np.where(rotations<1.57)] = rotations[np.where(rotations<1.57)] + np.pi
        from scipy import signal
        b, a = signal.butter(3, 0.05, 'lowpass')
        filtered_joints = signal.filtfilt(b, a, rec_joints3ds.T).T.copy()  # butterworth filter
        # filterd_joints = filterd_joints.reshape(f_length, n_joint, _)

        # viz_tool = Visualization()
        # while True:
        #     # viz_tool.visualize_joints(filtered_joints[:200])
        #     viz_tool.visualize_joints_test(filtered_joints)
            

        for joints, joints3d in zip(model_joint3ds, filtered_joints):
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

        # apply to model
        if est_scale:
            init_s = torch.tensor(scale, dtype=dtype)
        else:
            init_s = torch.tensor(fixed_scale, dtype=dtype)

        translations = np.array(translations)
        rotations = np.array(rotations)
        # # rotations[np.where(rotations<1.57)] = rotations[np.where(rotations<1.57)] + np.pi
        # from scipy import signal
        # b, a = signal.butter(3, 0.05, 'lowpass')
        # filterd_trans = signal.filtfilt(b, a, translations.T).T.copy()  # butterworth filter
        # filterd_rots = signal.filtfilt(b, a, rotations.T).T.copy()  # butterworth filter

        # rotations = (rotations + np.pi * 2 * 1000) % np.pi * 2

        init_t = torch.tensor(translations, dtype=dtype)
        init_r = torch.tensor(rotations, dtype=dtype)
        models[idx].reset_params(transl=init_t, global_orient=init_r, scale=init_s)

        if kwargs.get('use_vposer') or kwargs.get('use_motionprior'):
            with torch.no_grad():   
                setting['pose_embedding'][idx].fill_(0)

        # # load fixed parameters
        # init_s = torch.tensor(7., dtype=dtype)
        # init_shape = torch.tensor([2.39806, 0.678491, -1.38193, -0.966748, -1.29383,-0.795755, -0.303195, -1.1032, -0.197056, -0.102728 ], dtype=dtype)
        # model.reset_params(transl=init_t, global_orient=init_r, scale=init_s)
        # model.betas.requires_grad = False
        # model.scale.requires_grad = False

        del model_output
        torch.cuda.empty_cache()
        # visualize
        if False:
            import os
            from core.utils.module_utils import joint_projection, surface_projection
            from core.utils.render import Renderer

            # img = cv2.imread(os.path.join(dataset_obj.img_folder, data['img_path'][0][0]))
            
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
            # flag = flags[:,idx]
            for i, (joints, verts) in enumerate(zip(model_output.joints.detach().cpu().numpy(), model_output.vertices.detach().cpu().numpy())):
                # if flag[i] == 0:
                #     continue
                for v in range(1):
                    img = cv2.imread(os.path.join(dataset_obj.img_folder, data['img_path'][v][i]))
                    # joint_projection(filtered_joints[i], extris[v], intris[v], img, True)
                    # surface_projection(verts, models[idx].faces, joints, extris[v], intris[v], img, 5)

                    render = Renderer(resolution=(img.shape[1], img.shape[0]))
                    img = render(verts, models[idx].faces, extris[v][:3,:3].copy(), extris[v][:3,3].copy(), intris[v].copy(), img.copy(), color=[1,1,0.9], viz=False)
                    render.vis_img("img", img)
                    render.renderer.delete()
                    del render
    data['keypoints'] = keypoints
    data['flags'] = flags
    return data

# def load_init(setting, data, dataset_obj, results, frames_seq=1, use_torso=False, **kwargs):
#     model = setting['model']
#     dtype = setting['dtype']
#     device = setting['device']
#     # if the loss of last frame is too large, we use init_guess to get initial value
#     if results['loss'] > 99999:
#         init_guess(setting, data, dataset_obj, frames_seq, use_torso=use_torso, **kwargs)
#         setting['seq_start'] = True
#         return

#     init_t = torch.tensor(results['transl'], dtype=dtype)
#     init_r = torch.tensor(results['global_orient'], dtype=dtype)
#     init_s = torch.tensor(results['scale'], dtype=dtype)
#     init_shape = torch.tensor(results['betas'], dtype=dtype)
#     if kwargs.get('use_vposer'):
#         setting['pose_embedding'] = torch.tensor(results['pose_embedding'], dtype=dtype, device=device, requires_grad=True)
#     else:
#         # gmm prior, to do...
#         pass
#         #init_pose = torch.tensor(results['body_pose'], dtype=dtype)

#     # initial value
#     new_params = defaultdict(global_orient=init_r,
#                                 # body_pose=body_mean_pose,
#                                 transl=init_t,
#                                 scale=init_s,
#                                 betas=init_shape,
#                                 )
#     model.reset_params(**new_params)

#     # visualize
#     if False:
#         extris = setting['extrinsics']
#         intris = setting['intrinsics']
#         import os
#         from utils.utils import joint_projection, surface_projection
#         if kwargs.get('use_vposer'):
#             vposer = setting['vposer']
#             init_pose = vposer.decode(
#                 setting['pose_embedding'], output_type='aa').view(
#                     frames_seq, -1)
#         else:
#             init_pose = torch.zeros((frames_seq, 69), dtype=dtype, device=device)
#         model_output = model(return_verts=True, return_full_pose=True, body_pose=init_pose)
#         for i, (joints, verts) in enumerate(zip(model_output.joints.detach().cpu().numpy(), model_output.vertices.detach().cpu().numpy())):
#             for v in range(1):
#                 img = cv2.imread(os.path.join(dataset_obj.img_folder, data['img_path'][v][i]))
#                 surface_projection(verts, model.faces, joints, extris[v], intris[v], img, 5)


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