import torch
import numpy as np
import core.smplx
from core.smplx import SMPL as _SMPL
from core.smplx.body_models import ModelOutput
from core.smplx.lbs import vertices2joints


class SMPL(_SMPL):
    """ Extension of the official SMPL implementation to support more joints """

    def __init__(self, *args, **kwargs):
        super(SMPL, self).__init__(*args, **kwargs)
        J_regressor_extra = np.load('data/J_regressor_lsp.npy')
        self.register_buffer('J_regressor_LSP', torch.tensor(J_regressor_extra, dtype=torch.float32))

    def forward(self, *args, **kwargs):
        kwargs['get_skin'] = True
        smpl_output = super(SMPL, self).forward(*args, **kwargs)
        lsp_joints = vertices2joints(self.J_regressor_LSP, smpl_output.vertices)
        return lsp_joints, smpl_output.vertices
