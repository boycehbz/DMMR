'''
 @FileName    : MotionVAE_HP.py
 @EditTime    : 2021-12-13 15:42:47
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''

import os
import sys
import time
import numpy as np
from tqdm import tqdm
import torch 
from torch import nn
from torch.nn import functional as F


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

class MotionVAE_HP(nn.Module):
    def __init__(
            self,
            latentD,
            # frame_length,
            n_layers=1,
            hidden_size=512,
            bidirectional=True,
    ):
        super(MotionVAE_HP, self).__init__()

        self.dropout = nn.Dropout(p=.1, inplace=False)
        self.latentD = latentD
        self.encoder_gru = nn.GRU(
            input_size=69,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            num_layers=n_layers
        )

        self.decoder_gru = nn.GRU(
            input_size=256,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            num_layers=n_layers
        )
        self.rot_decoder = ContinousRotReprDecoder()

        if bidirectional:
            self.hidden_dim = hidden_size * 2
        else:
            self.hidden_dim = hidden_size

        self.encoder_linear1 = nn.Sequential(nn.Linear(self.hidden_dim, 256),
                                        nn.BatchNorm1d(256),
                                        nn.LeakyReLU(), \
                                        nn.Dropout(0.1),\
                                        )
        self.encoder_linear2 = nn.Sequential(nn.Linear(512, 256),
                                        nn.BatchNorm1d(256),
                                        nn.LeakyReLU(), \
                                        nn.Dropout(0.1),\
                                        )
        self.encoder_residual = nn.Sequential(nn.Linear(69, 256),
                                        nn.BatchNorm1d(256),
                                        nn.LeakyReLU(), \
                                        nn.Dropout(0.1),\
                                        )

        self.mu_linear = nn.Sequential(nn.Linear(256, latentD),
                                        )

        self.var_linear = nn.Sequential(nn.Linear(256, latentD),
                                        )

        self.decoder_linear1 = nn.Sequential(nn.Linear(latentD, 256),
                                            nn.LeakyReLU(), \
                                            # nn.Dropout(0.1),\
                                            # nn.Linear(256, self.hidden_dim),
                                            # nn.LeakyReLU(), \
                                            ) 
        self.decoder_linear2 = nn.Sequential(nn.Linear(self.hidden_dim, 256),
                                            nn.LeakyReLU(), \
                                            nn.Dropout(0.1),\
                                            ) 
        self.decoder_linear3 = nn.Sequential(nn.Linear(512, 256),
                                            nn.LeakyReLU(), \
                                            nn.Dropout(0.1),\
                                            nn.Linear(256, 23*6),
                                            nn.LeakyReLU(), \
                                            ) 


    def forward(self, x):
        n,t,f = x.shape

        q_z = self.encode(x, n, t)
        q_z_sample = q_z.rsample()
        out = self.decode(q_z_sample, n, t)

        return dict(param=out, mean=q_z.mean, std=q_z.scale)

    def encode(self, x, n, t):
        linear_proj = x.contiguous().view([-1, x.size(-1)])
        linear_proj = self.encoder_residual(linear_proj)
        x = x.permute(1,0,2) # NTF -> TNF
        y, _ = self.encoder_gru(x)
        y = y.permute(1,0,2)
        y = y.contiguous().view([-1, y.size(-1)])
        y = self.encoder_linear1(y)
        y = torch.cat([y, linear_proj], dim=1)
        y = self.encoder_linear2(y)
        mean = self.mu_linear(y).view([n, t, -1])
        std = self.var_linear(y).view([n, t, -1])

        q_z = torch.distributions.normal.Normal(mean, F.softplus(std))
        return q_z

    def decode(self, q_z_sample, n=1, t=32):
        q_z_sample = q_z_sample.view([-1, q_z_sample.size(-1)])

        linear_proj = self.decoder_linear1(q_z_sample)
        decoded_f = linear_proj.view([n, t, -1]).permute(1,0,2)
        out, _ = self.decoder_gru(decoded_f)
        out = out.permute(1,0,2)
        out = out.contiguous().view([n*t, -1])
        out = self.decoder_linear2(out)
        out = torch.cat([out, linear_proj], dim=1)
        out = self.decoder_linear3(out)
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
