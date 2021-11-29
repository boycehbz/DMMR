import os
import sys

import time
import numpy as np
from tqdm import tqdm
import torch 
from torch import nn
from torch.nn import functional as F
import torchgeometry as tgm

from model.mlp import MLP
from model.rnn import RNN

# class MotionVAE(nn.Module):
#     def __init__(self, latentD):
#         super(MotionVAE, self).__init__()
#         self.nx = nx = 69
#         self.nz = nz = latentD
#         self.t_total = 32
#         self.rnn_type = rnn_type = 'gru'
#         self.e_birnn = e_birnn = False
#         self.use_drnn_mlp = True
#         self.nx_rnn = nx_rnn = 128
#         self.nh_mlp = nh_mlp = [300, 200]
#         self.additive = False
#         # encode
#         self.e_rnn = RNN(nx, nx_rnn, bi_dir=e_birnn, cell_type=rnn_type)
#         self.e_mlp = MLP(nx_rnn, nh_mlp)
#         self.e_mu = nn.Linear(self.e_mlp.out_dim, nz)
#         self.e_logvar = nn.Linear(self.e_mlp.out_dim, nz)
#         # decode
#         if self.use_drnn_mlp:
#             self.drnn_mlp = MLP(nx_rnn, nh_mlp + [nx_rnn], activation='relu')
#         self.d_rnn = RNN(latentD + nx, nx_rnn, cell_type=rnn_type)
#         self.d_mlp = MLP(nx_rnn, nh_mlp)
#         self.d_out = nn.Linear(self.d_mlp.out_dim, nx)
#         self.d_rnn.set_mode('step')

#         self.init_pose_mlp = MLP(latentD, nh_mlp, activation='relu')
#         self.init_pose_out = nn.Linear(self.init_pose_mlp.out_dim, nx)

#     def encode_x(self, x):
#         if self.e_birnn:
#             h_x = self.e_rnn(x).mean(dim=0)
#         else:
#             h_x = self.e_rnn(x)[:,-1]
#         return h_x

#     # def encode_x_all(self, x):
#     #     h_x = self.encode_x(x)
#     #     h = self.e_mlp(h_x)
#     #     return h_x, self.e_mu(h), self.e_logvar(h)


#     def encode(self, x):
#         # self.e_rnn.initialize(batch_size=x.shape[0])
#         h_x = self.encode_x(x)
#         h = self.e_mlp(h_x)
#         return self.e_mu(h), self.e_logvar(h)
        

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5*logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std

#     # def encode_hx(self, h_x):
#     #     h_init_pose = self.init_pose_mlp(h_x)
#     #     h_init_pose = self.init_pose_out(h_init_pose)
#     #     h = self.e_mlp(h_x)
#     #     return self.e_mu(h), self.e_logvar(h), h_init_pose


#     # def decode_hx(self, h_x):
#     #     mu, logvar, h_init_pose = self.encode_hx(h_x)
#     #     z = mu
#     #     return self.decode(h_init_pose[None, ], z), mu, logvar

#     def decode(self, z, x_p = None):
#         if x_p == None:
#             h_init_pose = self.init_pose_mlp(z)
#             x = self.init_pose_out(h_init_pose)
#             x_p = x # Feeding in the first frame of the predicted input
        
#         self.d_rnn.initialize(batch_size=z.shape[0])
#         x_rec = []
        
#         for i in range(self.t_total):
#             rnn_in = torch.cat([x_p, z], dim=1)
#             h = self.d_rnn(rnn_in)
#             h = self.d_mlp(h)
#             x_i = self.d_out(h)
#             # if self.additive:
#                 # x_i[..., :-6] += y_p[..., :-6]
#             x_rec.append(x_i)
#             x_p = x_i
#         x_rec = torch.stack(x_rec)
        
#         return x_rec

#     def forward(self, x):
#         mu, logvar = self.encode(x)
#         z = self.reparameterize(mu, logvar) if self.training else mu
#         return dict(params=self.decode(z), mean=mu, std=logvar)

#     def sample_prior(self, x):
#         z = torch.randn((x.shape[1], self.nz), device=x.device)
#         return self.decode(z)
    
#     def step(self, model):
#         pass
def rotation_matrix_to_angle_axis(rotation_matrix):
    """Convert 3x4 rotation matrix to Rodrigues vector

    Args:
        rotation_matrix (Tensor): rotation matrix.

    Returns:
        Tensor: Rodrigues vector transformation.

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 3)`

    Example:
        >>> input = torch.rand(2, 3, 4)  # Nx4x4
        >>> output = tgm.rotation_matrix_to_angle_axis(input)  # Nx3
    """
    # todo add check that matrix is a valid rotation matrix
    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    return quaternion_to_angle_axis(quaternion)

def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """Convert 3x4 rotation matrix to 4d quaternion vector

    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.

    Return:
        Tensor: the rotation in quaternion

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`

    Example:
        >>> input = torch.rand(4, 3, 4)  # Nx3x4
        >>> output = tgm.rotation_matrix_to_quaternion(input)  # Nx4
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(rotation_matrix)))

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                rotation_matrix.shape))
    if not rotation_matrix.shape[-2:] == (3, 4):
        raise ValueError(
            "Input size must be a N x 3 x 4  tensor. Got {}".format(
                rotation_matrix.shape))

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
                      rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * (~ mask_d0_d1)
    mask_c2 = (~ mask_d2) * mask_d0_nd1
    mask_c3 = (~ mask_d2) * (~ mask_d0_nd1)
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                    t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa
    q *= 0.5
    return q

def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert quaternion vector to angle axis of rotation.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion (torch.Tensor): tensor with quaternions.

    Return:
        torch.Tensor: tensor with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = tgm.quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape Nx4 or 4. Got {}"
                         .format(quaternion.shape))
    # unpack input and compute conversion
    q1: torch.Tensor = quaternion[..., 1]
    q2: torch.Tensor = quaternion[..., 2]
    q3: torch.Tensor = quaternion[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    cos_theta: torch.Tensor = quaternion[..., 0]
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta))

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


class ContinousRotReprDecoder(nn.Module):
    def __init__(self):
        super(ContinousRotReprDecoder, self).__init__()

    def forward(self, module_input):
        reshaped_input = module_input.view(-1, 3, 2)

        b1 = F.normalize(reshaped_input[:, :, 0], dim=1)

        dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
        b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=1)

        return torch.stack([b1, b2, b3], dim=-1)

class MotionVAE(nn.Module):
    def __init__(
            self,
            latentD,
            n_layers=1,
            hidden_size=512,
            bidirectional=False,
    ):
        super(MotionVAE, self).__init__()

        self.dropout = nn.Dropout(p=.1, inplace=False)
        self.latentD = latentD
        self.encoder_gru = nn.GRU(
            input_size=69,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            num_layers=n_layers
        )

        self.decoder_gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            num_layers=n_layers
        )
        self.rot_decoder = ContinousRotReprDecoder()

        self.encoder_linear = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                        nn.LeakyReLU(), \
                                        nn.BatchNorm1d(hidden_size),
                                        nn.Dropout(0.1),\
                                        ) 

        self.mu_linear = nn.Sequential(nn.Linear(hidden_size, latentD),
                                        ) 

        self.var_linear = nn.Sequential(nn.Linear(hidden_size, latentD),
                                        )

        self.decoder_linear1 = nn.Sequential(nn.Linear(latentD, hidden_size),
                                            nn.LeakyReLU(), \
                                            nn.Dropout(0.1),\
                                            nn.Linear(hidden_size, hidden_size),
                                            nn.LeakyReLU(), \
                                            ) 
        self.decoder_linear2 = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                            nn.LeakyReLU(), \
                                            nn.Dropout(0.1),\
                                            nn.Linear(hidden_size, 23*6),
                                            nn.LeakyReLU(), \
                                            ) 

    def forward(self, x):
        n,t,f = x.shape

        q_z = self.encode(x, n, t)
        q_z_sample = q_z.rsample()
        out = self.decode(q_z_sample, n, t)

        return dict(param=out, mean=q_z.mean, std=q_z.scale)

    def encode(self, x, n, t):
        x = x.permute(1,0,2) # NTF -> TNF
        y, _ = self.encoder_gru(x)
        y = y.permute(1,0,2)
        y = y.contiguous().view([-1, y.size(-1)])
        y = self.encoder_linear(y)
        mean = self.mu_linear(y).view([n, t, -1])
        std = self.var_linear(y).view([n, t, -1])

        q_z = torch.distributions.normal.Normal(mean, F.softplus(std))
        return q_z

    def decode(self, q_z_sample, n=1, t=32):
        q_z_sample = q_z_sample.view([-1, q_z_sample.size(-1)])

        decoded_f = self.decoder_linear1(q_z_sample)
        decoded_f = decoded_f.view([n, t, -1]).permute(1,0,2)
        out, _ = self.decoder_gru(decoded_f)
        out = out.permute(1,0,2)
        out = out.contiguous().view([n*t, -1])
        out = self.decoder_linear2(out)
        out = self.rot_decoder(out)
        out = out.view([n*t, 1, 23, 9])
        out = self.matrot2aa(out)
        out = out.view([n, t, 69])
        return out


    def matrot2aa(self, pose_matrot):
        '''
        :param pose_matrot: Nx1xnum_jointsx9
        :return: Nx1xnum_jointsx3
        '''
        batch_size = pose_matrot.size(0)
        homogen_matrot = F.pad(pose_matrot.view(-1, 3, 3), [0,1])
        pose = rotation_matrix_to_angle_axis(homogen_matrot).view(batch_size, 1, -1, 3).contiguous()
        return pose
