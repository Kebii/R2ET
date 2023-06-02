import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import sdf.csrc as _C

class SDFFunction(Function):
    """
    Definition of SDF function
    """

    @staticmethod
    def forward(ctx, phi, faces, vertices):
        return _C.sdf(phi, faces, vertices)

    @staticmethod
    def backward(ctx):
        return None, None, None

class SDF(nn.Module):

    def forward(self, faces, vertices, grid_size=32):
        phi = torch.zeros(vertices.shape[0], grid_size, grid_size, grid_size, device=vertices.device)
        phi = SDFFunction.apply(phi, faces, vertices)
        return phi


class SDFFunction2(Function):
    """
    Definition of SDF function
    """

    @staticmethod
    def forward(ctx, phi, faces, vertices):
        return _C.sdf2(phi, faces, vertices)

    @staticmethod
    def backward(ctx):
        return None, None, None

class SDF2(nn.Module):

    def forward(self, faces, vertices, grid_size=32):
        phi = torch.zeros(vertices.shape[0], grid_size, grid_size, grid_size, device=vertices.device)
        phi = SDFFunction2.apply(phi, faces, vertices)
        return phi