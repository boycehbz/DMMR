'''
 @FileName    : init.py
 @EditTime    : 2021-11-24 19:51:46
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''
import os
import os.path as osp
import yaml
import torch
import sys
import numpy as np
from core.utils.data_parser import FittingData
from core.utils.writer import FileWriter
from core.utils.prior_terms import create_prior
from core.utils.module_utils import JointMapper
from core.smplx.my_smpl_model import create_scale
from core.utils.prior import load_vposer
from core.utils.motionprior import load_motionpriorHP


def load_model(dataset_obj, setting, **kwarg):

    input_gender = kwarg.pop('gender', 'neutral')
    dtype = setting['dtype']
    device = setting['device']

    # Map SMPL joints to 2D keypoints (halpe-26 format https://github.com/Fang-Haoshu/Halpe-FullBody)
    joint_mapper = JointMapper(dataset_obj.get_model2data())

    model_params = dict(model_path=kwarg.get('model_folder'),
                        joint_mapper=joint_mapper,
                        create_global_orient=True,
                        create_body_pose=not (kwarg.get('use_vposer') or kwarg.get('use_motionprior')),
                        create_betas=True,
                        create_left_hand_pose=False,
                        create_right_hand_pose=False,
                        create_expression=False,
                        create_jaw_pose=False,
                        create_leye_pose=False,
                        create_reye_pose=False,
                        create_transl=True, #set transl in multi-view task  --Buzhen Huang 07/31/2019
                        create_scale=True,
                        batch_size=dataset_obj.frames,
                        dtype=dtype,
                        **kwarg)
    models = []
    pose_embeddings = []
    # load vposer
    vposer = None
    if kwarg.get('use_vposer'):
        vposer = load_vposer()
        vposer = vposer.to(device=device)
        vposer.eval()
    elif kwarg.get('use_motionprior'):
        vposer = load_motionpriorHP()
        vposer = vposer.to(device=device)
        vposer.eval()

    for idx in range(dataset_obj.num_people):
        #model = smplx.create_scale(gender=input_gender, **model_params)
        model = create_scale(gender=input_gender, **model_params)
        model = model.to(device=device)
        models.append(model)

        pose_embedding = None
        # batch_size = frames
        pose_embedding = torch.zeros([dataset_obj.frames, 32],
                                    dtype=dtype, device=device,
                                    requires_grad=True)
        pose_embeddings.append(pose_embedding)


    setting['model'] = models
    setting['vposer'] = vposer
    setting['pose_embedding'] = pose_embeddings
    setting['frames'] = dataset_obj.frames
    return setting

def init(**kwarg):
    '''
    Global settings
    '''
    setting = {}
    # Create folders
    output_folder = kwarg.pop('output_folder')
    output_folder = osp.expandvars(output_folder)
    if not osp.exists(output_folder):
        os.makedirs(output_folder)

    # Store the arguments for the current experiment
    conf_fn = osp.join(output_folder, 'conf.yaml')
    with open(conf_fn, 'w') as conf_file:
        yaml.dump(kwarg, conf_file)

    result_folder = kwarg.pop('result_folder', 'results')
    result_folder = osp.join(output_folder, result_folder)
    if not osp.exists(result_folder):
        os.makedirs(result_folder)

    mesh_folder = kwarg.pop('mesh_folder', 'meshes')
    mesh_folder = osp.join(output_folder, mesh_folder)
    if not osp.exists(mesh_folder):
        os.makedirs(mesh_folder)

    out_cam_folder = osp.join(output_folder, 'camparams')
    if not osp.exists(out_cam_folder) and kwarg.get('opt_cam'):
        os.makedirs(out_cam_folder)

    out_img_folder = osp.join(output_folder, 'images')
    if not osp.exists(out_img_folder):
        os.makedirs(out_img_folder)

    # Assert cuda is available
    use_cuda = kwarg.get('use_cuda', True)
    if use_cuda and not torch.cuda.is_available():
        print('CUDA is not available, exiting!')
        sys.exit(-1)

    # Read gender
    model_type = kwarg.get('model_type')

    if model_type == 'smpllsp':
        pose_format = 'lsp14' 
    elif model_type == 'smplhalpe':
        pose_format = 'halpe'
    else:
        pose_format = 'coco17'

    # Load data
    dataset_obj = FittingData(pose_format=pose_format, **kwarg)

    writer = FileWriter(output_path=output_folder, dataset_dir=dataset_obj.data_folder)

    float_dtype = kwarg.get('float_dtype', 'float32')
    if float_dtype == 'float64':
        dtype = torch.float64
    elif float_dtype == 'float32':
        dtype = torch.float32
    else:
        raise ValueError('Unknown float type {}, exiting!'.format(float_dtype))

    # Create prior
    body_pose_prior = create_prior(
        prior_type=kwarg.get('body_prior_type'),
        dtype=dtype,
        **kwarg)
    shape_prior = create_prior(
        prior_type=kwarg.get('shape_prior_type', 'l2'),
        dtype=dtype, **kwarg)
    angle_prior = create_prior(prior_type='angle', dtype=dtype)

    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        body_pose_prior = body_pose_prior.to(device=device)
        angle_prior = angle_prior.to(device=device)
        shape_prior = shape_prior.to(device=device)
    else:
        device = torch.device('cpu')
    
    # A weight for every joint of the model
    joint_weights = dataset_obj.get_joint_weights().to(device=device,
                                                       dtype=dtype)
    # Add a fake batch dimension for broadcasting
    joint_weights.unsqueeze_(dim=0)

    # process fixed parameters
    setting['fix_scale'] = kwarg.get('fix_scale')
    if kwarg.get('fix_scale'):
        setting['fixed_scale'] = np.array(kwarg.get('scale'))
    else:
        setting['fixed_scale'] = None
    if kwarg.get('fix_shape'):
        setting['fixed_shape'] = np.array(kwarg.get('shape'))
    else:
        setting['fixed_shape'] = None

    # return setting
    setting['use_3d'] = kwarg.pop("use_3d")
    setting['dtype'] = dtype
    setting['device'] = device
    setting['joints_weight'] = joint_weights
    setting['body_pose_prior'] = body_pose_prior
    setting['shape_prior'] = shape_prior
    setting['angle_prior'] = angle_prior
    setting['img_folder'] = out_img_folder
    setting['cam_folder'] = out_cam_folder
    setting['result_folder'] = result_folder
    setting['mesh_folder'] = mesh_folder

    setting['writer'] = writer
    
    return dataset_obj, setting

