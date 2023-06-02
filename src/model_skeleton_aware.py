import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import math

from src.forward_kinematics import FK
from src.linear_blend_skin import linear_blend_skinning
from src.ops import qlinear, q_mul_q
from torch import atan2
from torch import asin
from collections import OrderedDict


class Attention(nn.Module):
    def __init__(self, dim, out_dim, heads=4, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim**-0.5  # 1/sqrt(dim)

        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)
        torch.nn.init.xavier_uniform_(self.to_qkv.weight)
        torch.nn.init.zeros_(self.to_qkv.bias)

        self.nn1 = nn.Linear(dim, out_dim)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.zeros_(self.nn1.bias)
        self.do1 = nn.Dropout(dropout)

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x)

        qkv = qkv.view(b, n, 3, h, -1).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0, :, :, :, :], qkv[1, :, :, :, :], qkv[2, :, :, :, :]

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(b, n, -1)

        out = self.nn1(out)
        out = self.do1(out)

        return out


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class MLP_Block(nn.Module):
    def __init__(self, dim, hid_dim, dropout=0.1):
        super().__init__()
        self.nn1 = nn.Linear(dim, hid_dim)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std=1e-6)
        self.af1 = nn.ReLU()
        self.do1 = nn.Dropout(dropout)
        self.nn2 = nn.Linear(hid_dim, dim)
        torch.nn.init.xavier_uniform_(self.nn2.weight)
        torch.nn.init.normal_(self.nn2.bias, std=1e-6)
        self.do2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.nn1(x)
        x = self.af1(x)
        x = self.do1(x)
        x = self.nn2(x)
        x = self.do2(x)

        return x


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        if dim == mlp_dim:
            for _ in range(depth):
                self.layers.append(
                    nn.ModuleList(
                        [
                            Residual(
                                Attention(dim, mlp_dim, heads=heads, dropout=dropout)
                            ),
                            Residual(
                                LayerNormalize(
                                    mlp_dim,
                                    MLP_Block(mlp_dim, mlp_dim * 2, dropout=dropout),
                                )
                            ),
                        ]
                    )
                )
        else:
            for _ in range(depth):
                self.layers.append(
                    nn.ModuleList(
                        [
                            Attention(dim, mlp_dim, heads=heads, dropout=dropout),
                            Residual(
                                LayerNormalize(
                                    mlp_dim,
                                    MLP_Block(mlp_dim, mlp_dim * 2, dropout=dropout),
                                )
                            ),
                        ]
                    )
                )

    def forward(self, x):
        for attention, mlp in self.layers:
            x = attention(x)
            x = mlp(x)
        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for non-recurrent neural networks.
    Args:
       dropout (float): dropout parameter
       dim (int): embedding size
    """

    def __init__(self, dropout, dim, max_len=22):
        if dim % 2 != 0:
            raise ValueError(
                "Cannot use sin/cos positional encoding with "
                "odd dim (got dim={:d})".format(dim)
            )
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            (torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim))
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb):
        """Embed inputs.
        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(batch_size, n, self.dim)``
        """
        emb = emb * math.sqrt(self.dim)
        emb = emb + self.pe[:, 0 : emb.size(1), :]
        emb = self.dropout(emb)
        return emb


class QuatEncoder(nn.Module):
    def __init__(self, num_joint, token_channels, hidden_channels, kp):
        super(QuatEncoder, self).__init__()

        self.num_joint = num_joint
        self.token_linear = nn.Linear(4, token_channels)
        self.trans1 = Transformer(token_channels, 1, 4, hidden_channels, 1 - kp)

    def forward(self, pose_t):
        token_q = self.token_linear(pose_t)
        embed_q = self.trans1(token_q)

        return embed_q


class SeklEncoder(nn.Module):
    def __init__(self, num_joint, token_channels, embed_channels, kp):
        super(SeklEncoder, self).__init__()

        self.num_joint = num_joint
        self.token_linear = nn.Linear(3, token_channels)
        self.trans1 = Transformer(token_channels, 1, 2, embed_channels, 1 - kp)

    def forward(self, skel):
        token_s = self.token_linear(skel)
        embed_s = self.trans1(token_s)

        return embed_s


class DeltaDecoder(nn.Module):
    def __init__(self, num_joint, token_channels, embed_channels, hidden_channels, kp):
        super(DeltaDecoder, self).__init__()
        self.num_joint = num_joint
        self.q_encoder = QuatEncoder(num_joint, token_channels, hidden_channels, kp)
        self.skel_encoder = SeklEncoder(num_joint, token_channels, embed_channels, kp)
        self.pos_encoder = PositionalEncoding(
            1 - kp, hidden_channels + (2 * embed_channels)
        )

        self.embed_linear = nn.Linear(
            hidden_channels + (2 * embed_channels), embed_channels
        )
        self.embed_acti = nn.ReLU()
        self.embed_drop = nn.Dropout(1 - kp)
        self.delta_linear = nn.Linear(embed_channels, 4)

    def forward(self, q_t, skelA, skelB):
        q_embed = self.q_encoder(q_t)

        skelA_embed = self.skel_encoder(skelA)
        skelB_embed = self.skel_encoder(skelB)

        cat_embed = torch.cat([q_embed, skelA_embed, skelB_embed], dim=-1)  # bs n c
        pos_embed = self.pos_encoder(cat_embed)

        embed = self.embed_drop(self.embed_acti(self.embed_linear(pos_embed)))
        deltaq_t = self.delta_linear(embed)

        return deltaq_t


class MotionDis(nn.Module):
    def __init__(self, kp):
        super(MotionDis, self).__init__()
        pad = int((4 - 1) / 2)

        self.seq = nn.Sequential(
            OrderedDict(
                [
                    ('dropout', nn.Dropout(p=1 - kp)),
                    ('h0', nn.Conv1d(3, 16, kernel_size=4, padding=pad, stride=2)),
                    ('acti0', nn.LeakyReLU(0.2)),
                    ('h1', nn.Conv1d(16, 32, kernel_size=4, padding=pad, stride=2)),
                    ('bn1', nn.BatchNorm1d(32)),
                    ('acti1', nn.LeakyReLU(0.2)),
                    ('h2', nn.Conv1d(32, 64, kernel_size=4, padding=pad, stride=2)),
                    ('bn2', nn.BatchNorm1d(64)),
                    ('acti2', nn.LeakyReLU(0.2)),
                    ('h3', nn.Conv1d(64, 64, kernel_size=4, padding=pad, stride=2)),
                    ('bn3', nn.BatchNorm1d(64)),
                    ('acti3', nn.LeakyReLU(0.2)),
                    ('h4', nn.Conv1d(64, 1, kernel_size=3, stride=2)),
                    ('sigmoid', nn.Sigmoid()),
                ]
            )
        )

    def forward(self, x):
        # x: bs 3 T
        bs = x.size(0)
        y = self.seq(x)
        return y.view(bs, 1)


class RetNet(nn.Module):
    def __init__(
        self,
        num_joint=22,
        token_channels=64,
        hidden_channels_p=256,
        embed_channels_p=128,
        kp=0.8,
    ):
        super(RetNet, self).__init__()
        self.num_joint = num_joint
        self.delta_dec = DeltaDecoder(
            num_joint, token_channels, embed_channels_p, hidden_channels_p, kp
        )

    def forward(
        self,
        seqA,
        seqB,
        skelA,
        skelB,
        shapeA,
        shapeB,
        quatA,
        inp_height,
        tgt_height,
        local_mean,
        local_std,
        quat_mean,
        quat_std,
        parents,
    ):
        '''
        seqA, seqB: bs T joints*3+4
        skelA, skelB: bs T joints*3
        shapeA, shapeB: bs 6
        height: bs 1
        '''
        self.parents = parents
        bs, T = seqA.size(0), seqA.size(1)
        local_mean = torch.from_numpy(local_mean).float().cuda(seqA.device)
        local_std = torch.from_numpy(local_std).float().cuda(seqA.device)
        quat_mean = torch.from_numpy(quat_mean).float().cuda(seqA.device)
        quat_std = torch.from_numpy(quat_std).float().cuda(seqA.device)
        parents = torch.from_numpy(parents).cuda(seqA.device)
        shapeB = shapeB.view((bs, self.num_joint, 3))
        shapeA = shapeA.view((bs, self.num_joint, 3))

        t_poseB = torch.reshape(skelB[:, 0, :], [bs, self.num_joint, 3])
        t_poseB = t_poseB * local_std + local_mean
        refB = t_poseB
        refB_feed = skelB[:, 0, :]
        refA_feed = skelA[:, 0, :]

        quatA_denorm = quatA * quat_std[None, :] + quat_mean[None, :]

        delta_qs = []
        B_locals_rt = []
        B_quats_rt = []

        """ mapping local rotation from A to B frame by frame"""
        for t in range(T):
            qoutA_t = quatA[:, t, :, :]  # motion copy
            qoutA_t_denorm = quatA_denorm[:, t, :, :]

            # delta qs
            refA_feed = refA_feed.view((bs, self.num_joint, 3))
            refB_feed = refB_feed.view((bs, self.num_joint, 3))
            deltaq_t = self.delta_dec(qoutA_t, refA_feed, refB_feed)  # bs n 4
            deltaq_t = deltaq_t * quat_std + quat_mean
            deltaq_t = normalized(deltaq_t)

            delta_qs.append(deltaq_t)

            qB_t = q_mul_q(qoutA_t_denorm, deltaq_t)
            B_quats_rt.append(qB_t)

            # Forward Kinematics
            localB_out_t = FK.run(parents, refB, qB_t)
            localB_out_t = (localB_out_t - local_mean) / local_std
            B_locals_rt.append(localB_out_t)

        quatB_rt = torch.stack(B_quats_rt, dim=1)  # shape: (batch_size, T, 22, 4)
        delta_qs = torch.stack(delta_qs, dim=1)
        localB_rt = torch.stack(B_locals_rt, dim=1)  # shape: (batch_size, T, 22, 3)

        """ mapping global movements from A to B"""
        gA_vel = seqA[:, :, -4:-1]
        gA_rot = seqA[:, :, -1]

        normalized_vin = torch.cat(
            (torch.divide(gA_vel, inp_height[:, :, None]), gA_rot[:, :, None]), dim=-1
        )
        normalized_vout = normalized_vin.clone()

        gB_vel = normalized_vout[:, :, :-1]
        gB_rot = normalized_vout[:, :, -1]
        globalB_rt = torch.cat(
            (torch.multiply(gB_vel, tgt_height[:, :, None]), gB_rot[:, :, None]), dim=-1
        )  # shape: (batch_size, T, 4)

        if self.training:
            localA_gt = torch.reshape(seqA[:, :, :-4], [bs, T, self.num_joint, 3])
            localB_gt = torch.reshape(seqB[:, :, :-4], [bs, T, self.num_joint, 3])
            globalA_gt = seqA[:, :, -4:]

            return (
                localA_gt,
                localB_rt,
                localB_gt,
                globalA_gt,
                globalB_rt,
                quatB_rt,
            )

        return localB_rt, globalB_rt, quatB_rt, delta_qs

    @staticmethod
    def get_recon_loss(
        atte_lst, num_joint, ae_reg, mask, localB_rt, localB_gt, quatA_gt, quatB_rt
    ):
        attW = torch.ones(num_joint).cuda(localB_rt.device)
        attW[atte_lst] = 2

        """ For training stability, we chose the same character to the input with p = 0.5 """
        ae_joints_err = torch.sum(
            (
                torch.multiply(
                    ae_reg[:, :, None, None] * mask[:, :, None, None],
                    torch.subtract(localB_rt, localB_gt),
                )
            )
            ** 2,
            dim=[0, 1, 3],
        )
        local_ae_loss = torch.sum(attW * ae_joints_err)
        local_ae_loss = torch.divide(
            local_ae_loss,
            torch.maximum(
                torch.sum(ae_reg * mask), torch.tensor(1).cuda(ae_reg.device)
            ),
        )

        quat_ae_loss = torch.sum(
            (
                torch.multiply(
                    ae_reg[:, :, None, None] * mask[:, :, None, None],
                    torch.subtract(quatA_gt, quatB_rt),
                )
            )
            ** 2
        )
        quat_ae_loss = torch.divide(
            quat_ae_loss,
            torch.maximum(
                torch.sum(ae_reg * mask), torch.tensor(1).cuda(ae_reg.device)
            ),
        )
        return local_ae_loss, quat_ae_loss

    @staticmethod
    def get_rot_cons_loss(alpha, euler_ord, quatB_rt):
        rads = alpha / 180.0
        twistB_loss = torch.mean(
            torch.square(
                torch.maximum(
                    torch.tensor(0).cuda(quatB_rt.device),
                    torch.abs(euler_y(quatB_rt, euler_ord)) - rads * np.pi,
                )
            )
        )

        return twistB_loss

    @staticmethod
    def get_gen_loss(fake_score, ae_reg):
        bceloss = nn.BCELoss(reduction='none')
        gen_motion_loss = bceloss(
            fake_score, torch.ones(fake_score.shape).cuda(fake_score.device)
        )
        gen_motion_loss = torch.sum(torch.multiply((1 - ae_reg), gen_motion_loss))
        gen_motion_loss = torch.divide(
            gen_motion_loss,
            torch.maximum(torch.sum(1 - ae_reg), torch.tensor(1).cuda(ae_reg.device)),
        )

        return gen_motion_loss

    @staticmethod
    def get_sem_loss(atte_lst, num_joint, rA, rB, mask):
        attW = torch.ones(num_joint).cuda(rB.device)
        attW[atte_lst] = 1
        # bs T n n
        rela_loss = torch.sum(
            (torch.multiply(mask[:, :, None, None], torch.subtract(rA, rB))) ** 2,
            dim=[0, 1, 3],
        )
        rela_loss = torch.sum(attW * rela_loss)
        rela_loss = torch.divide(
            rela_loss, torch.maximum(torch.sum(mask), torch.tensor(1).cuda(mask.device))
        )

        return rela_loss

    @staticmethod
    def get_rela_matrix(localB_rt, localA_gt, heightB, heightA):
        bs, t, num_joint, d = localB_rt.shape
        localB_rt = localB_rt.view(bs * t, num_joint, d)
        localA_gt = localA_gt.view(bs * t, num_joint, d)

        dis_matrixB = torch.cdist(localB_rt, localB_rt, p=2).view(
            bs, t, num_joint, num_joint
        )
        dis_matrixA = torch.cdist(localA_gt, localA_gt, p=2).view(
            bs, t, num_joint, num_joint
        )

        def normalize_matrix(m, h):
            m = m / (h.unsqueeze(-1).unsqueeze(-1) * 100)
            row_sum = torch.sum(m, dim=3, keepdim=True)
            m = m / row_sum

            return m

        return normalize_matrix(dis_matrixB, heightB), normalize_matrix(
            dis_matrixA, heightA
        )

    @staticmethod
    def get_dis_loss(real_score, fake_score, ae_reg):
        bceloss = nn.BCELoss(reduction='none')
        gen_motion_loss = bceloss(
            fake_score, torch.zeros(fake_score.shape).cuda(fake_score.device)
        )
        gen_motion_loss = torch.sum(torch.multiply((1 - ae_reg), gen_motion_loss))
        gen_motion_loss = torch.divide(
            gen_motion_loss,
            torch.maximum(torch.sum(1 - ae_reg), torch.tensor(1).cuda(ae_reg.device)),
        )

        dis_motion_loss = bceloss(
            real_score, torch.ones(real_score.shape).cuda(real_score.device)
        )
        dis_motion_loss = torch.sum(torch.multiply((1 - ae_reg), dis_motion_loss))
        dis_motion_loss = torch.divide(
            dis_motion_loss,
            torch.maximum(torch.sum(1 - ae_reg), torch.tensor(1).cuda(ae_reg.device)),
        )

        return gen_motion_loss + dis_motion_loss


def get_bounding_boxes(vertices):
    num_people = vertices.shape[0]
    boxes = torch.zeros(num_people, 2, 3, device=vertices.device)
    for i in range(num_people):
        boxes[i, 0, :] = vertices[i].min(dim=0)[0]
        boxes[i, 1, :] = vertices[i].max(dim=0)[0]
    return boxes


def normalized(angles):
    lengths = torch.sqrt(torch.sum(torch.square(angles), dim=-1))
    normalized_angle = angles / lengths[..., None]
    return normalized_angle


def euler_y(angles, order="yzx"):
    q = normalized(angles)
    q0 = q[..., 0]
    q1 = q[..., 1]
    q2 = q[..., 2]
    q3 = q[..., 3]

    if order == "xyz":
        ex = atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        ey = asin(torch.clamp(2 * (q0 * q2 - q3 * q1), -1, 1))
        ez = atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
        return torch.stack(values=[ex, ez], dim=-1)[:, :, 1:]
    elif order == "yzx":
        ex = atan2(2 * (q1 * q0 - q2 * q3), -q1 * q1 + q2 * q2 - q3 * q3 + q0 * q0)
        ey = atan2(2 * (q2 * q0 - q1 * q3), q1 * q1 - q2 * q2 - q3 * q3 + q0 * q0)
        ez = asin(torch.clamp(2 * (q1 * q2 + q3 * q0), -1, 1))
        return ey[:, :, 1:]
    else:
        raise Exception("Unknown Euler order!")


def euler_rot(angles):
    q = normalized(angles)
    q0 = q[..., 0]
    q1 = q[..., 1]
    q2 = q[..., 2]
    q3 = q[..., 3]

    ex = atan2(2 * (q1 * q0 - q2 * q3), -q1 * q1 + q2 * q2 - q3 * q3 + q0 * q0)
    ey = atan2(2 * (q2 * q0 - q1 * q3), q1 * q1 - q2 * q2 - q3 * q3 + q0 * q0)
    ez = asin(torch.clamp(2 * (q1 * q2 + q3 * q0), -1, 1))

    rotx = ex[..., :]
    roty = ey[..., :]
    rotz = ez[..., :]

    rot = torch.stack([rotx, roty, rotz], dim=-1)
    return rot
