'''
 @FileName    : motionprior.py
 @EditTime    : 2021-12-11 18:01:07
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''

import os

def expid2model(expr_dir):
    from configer import Configer

    if not os.path.exists(expr_dir): raise ValueError('Could not find the experiment directory: %s' % expr_dir)

    best_model_fname = os.path.join(expr_dir, 'snapshots', 'motionprior_hp.pkl')
    # try_num = os.path.basename(best_model_fname).split('_')[0]

    print(('Found Trained Model: %s' % best_model_fname))

    # default_ps_fname = glob.glob(os.path.join(expr_dir,'*.ini'))[0]
    # if not os.path.exists(
    #     default_ps_fname): raise ValueError('Could not find the appropriate vposer_settings: %s' % default_ps_fname)
    # ps = Configer(default_ps_fname=default_ps_fname, work_dir = expr_dir, best_model_fname=best_model_fname)

    return best_model_fname

def load_motionprior(expr_dir, vp_model='snapshot'):
    '''

    :param expr_dir:
    :param vp_model: either 'snapshot' to use the experiment folder's code or a VPoser imported module, e.g.
    from human_body_prior.train.vposer_smpl import VPoser, then pass VPoser to this function
    :param if True will load the model definition used for training, and not the one in current repository
    :return:
    '''
    import importlib
    import os
    import torch
    from core.model.MotionVAE import MotionVAE

    # settings of Vposer++
    num_neurons = 512
    latentD = 32
    data_shape = [1,23,3]
    trained_model_fname = expid2model(expr_dir)
    
    vposer_pt = MotionVAE(latentD=32)

    model_dict = vposer_pt.state_dict()
    premodel_dict = torch.load(trained_model_fname).state_dict()
    premodel_dict = {k: v for k ,v in premodel_dict.items() if k in model_dict}
    model_dict.update(premodel_dict)
    vposer_pt.load_state_dict(model_dict)
    print("load pretrain parameters from %s" %trained_model_fname)

    vposer_pt.eval()

    return vposer_pt

def load_motionpriorHP(device):
    import importlib
    import os
    import torch
    from core.model.MotionVAE_HP import MotionVAE_HP

    # settings of Motion Prior
    trained_model_fname = 'data/motionprior_hp.pkl'
    
    vposer_pt = MotionVAE_HP(latentD=32)

    model_dict = vposer_pt.state_dict()
    premodel_dict = torch.load(trained_model_fname, map_location=device)['model'] #.state_dict()
    premodel_dict = {k: v for k ,v in premodel_dict.items() if k in model_dict}
    model_dict.update(premodel_dict)
    vposer_pt.load_state_dict(model_dict)
    print("load pretrain parameters from %s" %trained_model_fname)

    vposer_pt.eval()

    return vposer_pt

