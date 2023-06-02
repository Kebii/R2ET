import os

import torch
import torch.nn as nn
import numpy as np
import time
import math

from src.forward_kinematics import FK
from src.linear_blend_skin import linear_blend_skinning
from src.ops import qlinear, q_mul_q
from torch import atan2
from torch import asin
from collections import OrderedDict
import trimesh
from sdf import SDF, SDF2


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
            x = attention(x)  # go to attention
            x = mlp(x)  # go to MLP_Block
        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for non-recurrent neural networks.
    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`
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
        # pose_t: bs joint, 4
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
        # bs = skel.shape[0]
        token_s = self.token_linear(skel)
        embed_s = self.trans1(token_s)

        return embed_s


class ShapeEncoder(nn.Module):
    def __init__(self, num_joint, token_channels, embed_channels, kp):
        super(ShapeEncoder, self).__init__()

        self.num_joint = num_joint
        self.token_linear = nn.Linear(3, token_channels)
        self.trans1 = Transformer(token_channels, 1, 2, embed_channels, 1 - kp)

    def forward(self, shape):
        token_s = self.token_linear(shape)
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


class DeltaShapeDecoder(nn.Module):
    def __init__(self, num_joint, hidden_channels, kp):
        super(DeltaShapeDecoder, self).__init__()

        self.num_joint = num_joint

        self.joint_linear1 = nn.Linear(7 * 22, hidden_channels)
        self.joint_acti1 = nn.ReLU()
        self.joint_drop1 = nn.Dropout(p=1 - kp)
        self.joint_linear2 = nn.Linear(hidden_channels, hidden_channels)
        self.joint_acti2 = nn.ReLU()
        self.joint_drop2 = nn.Dropout(p=1 - kp)

        self.delta_linear = qlinear(hidden_channels, 4 * num_joint)

    def forward(self, shapeB, x):
        bs = shapeB.shape[0]
        x_cat = torch.cat([shapeB, x], dim=-1)
        x_cat = x_cat.view((bs, -1))

        x_embed = self.joint_drop1(self.joint_acti1(self.joint_linear1(x_cat)))
        x_embed = self.joint_drop2(self.joint_acti2(self.joint_linear2(x_embed)))
        deltaq_t = self.delta_linear(x_embed)

        return deltaq_t


class WeightsDecoder(nn.Module):
    def __init__(self, num_joint, hidden_channels, kp):
        super(WeightsDecoder, self).__init__()

        self.num_joint = num_joint

        self.joint_linear1 = nn.Linear(10 * num_joint, 2 * hidden_channels)
        self.joint_acti1 = nn.ReLU()
        self.joint_drop1 = nn.Dropout(p=1 - kp)
        self.joint_linear2 = nn.Linear(2 * hidden_channels, 2 * hidden_channels)
        self.joint_acti2 = nn.ReLU()
        self.joint_drop2 = nn.Dropout(p=1 - kp)
        self.joint_linear3 = nn.Linear(2 * hidden_channels, hidden_channels)
        self.joint_acti3 = nn.ReLU()
        self.joint_drop3 = nn.Dropout(p=1 - kp)

        self.weights_linear = nn.Linear(hidden_channels, num_joint)
        self.weights_acti = nn.Sigmoid()

    def forward(self, refB, shapeB, x):
        bs = refB.shape[0]
        x_cat = torch.cat([refB, shapeB, x], dim=-1)
        x_cat = x_cat.view((bs, -1))

        x_embed = self.joint_drop1(self.joint_acti1(self.joint_linear1(x_cat)))
        x_embed = self.joint_drop2(self.joint_acti2(self.joint_linear2(x_embed)))
        x_embed = self.joint_drop3(self.joint_acti3(self.joint_linear3(x_embed)))

        weights = self.weights_acti(self.weights_linear(x_embed))

        return weights


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


def normalized(angles):
    lengths = torch.sqrt(torch.sum(torch.square(angles), dim=-1))
    normalized_angle = angles / lengths[..., None]
    return normalized_angle


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

        self.delta_leftArm_dec = DeltaShapeDecoder(3, hidden_channels_p, kp)
        self.delta_rightArm_dec = DeltaShapeDecoder(3, hidden_channels_p, kp)
        self.delta_leftLeg_dec = DeltaShapeDecoder(2, hidden_channels_p, kp)
        self.delta_rightLeg_dec = DeltaShapeDecoder(2, hidden_channels_p, kp)

        self.weights_dec = WeightsDecoder(num_joint, hidden_channels_p, kp)

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

        t_poseB = torch.reshape(skelB[:, 0, :], [bs, self.num_joint, 3])
        t_poseB = t_poseB * local_std + local_mean
        refB = t_poseB
        refB_feed = skelB[:, 0, :]
        refA_feed = skelA[:, 0, :]
        shapeB = shapeB.view((bs, self.num_joint, 3))
        shapeA = shapeA.view((bs, self.num_joint, 3))

        quatA_denorm = quatA * quat_std[None, :] + quat_mean[None, :]

        delta_qs = []
        B_locals_rt = []
        B_quats_rt = []
        B_quats_base = []
        B_locals_base = []
        B_gates = []
        delta_qg = []

        # manully adjust k for balacing gate
        k = 1.0
        leftArm_joints = [14, 15, 16]
        rightArm_joints = [18, 19, 20]
        leftLeg_joints = [6, 7]
        rightLeg_joints = [10, 11]

        """ mapping from A to B frame by frame"""
        for t in range(T):
            qoutA_t = quatA[:, t, :, :]  # motion copy
            qoutA_t_denorm = quatA_denorm[:, t, :, :]

            # delta qs
            refA_feed = refA_feed.view((bs, self.num_joint, 3))
            refB_feed = refB_feed.view((bs, self.num_joint, 3))
            delta1 = self.delta_dec(qoutA_t, refA_feed, refB_feed)
            delta1 = delta1 * quat_std + quat_mean
            delta1 = normalized(delta1)
            delta_qs.append(delta1)

            qB_base = q_mul_q(qoutA_t_denorm, delta1)
            qB_base = qB_base.detach()
            qB_base_norm = (qB_base - quat_mean) / quat_std

            # delta qg
            delta2_leftArm = self.delta_leftArm_dec(shapeB, qB_base_norm)
            delta2_leftArm = torch.reshape(delta2_leftArm, [bs, 3, 4])
            delta2_leftArm = (
                delta2_leftArm * quat_std[:, leftArm_joints, :]
                + quat_mean[:, leftArm_joints, :]
            )
            delta2_leftArm = normalized(delta2_leftArm)

            delta2_rightArm = self.delta_rightArm_dec(shapeB, qB_base_norm)
            delta2_rightArm = torch.reshape(delta2_rightArm, [bs, 3, 4])
            delta2_rightArm = (
                delta2_rightArm * quat_std[:, rightArm_joints, :]
                + quat_mean[:, rightArm_joints, :]
            )
            delta2_rightArm = normalized(delta2_rightArm)

            delta2_leftLeg = self.delta_leftLeg_dec(shapeB, qB_base_norm)
            delta2_leftLeg = torch.reshape(delta2_leftLeg, [bs, 2, 4])
            delta2_leftLeg = (
                delta2_leftLeg * quat_std[:, leftLeg_joints, :]
                + quat_mean[:, leftLeg_joints, :]
            )
            delta2_leftLeg = normalized(delta2_leftLeg)

            delta2_rightLeg = self.delta_rightLeg_dec(shapeB, qB_base_norm)
            delta2_rightLeg = torch.reshape(delta2_rightLeg, [bs, 2, 4])
            delta2_rightLeg = (
                delta2_rightLeg * quat_std[:, rightLeg_joints, :]
                + quat_mean[:, rightLeg_joints, :]
            )
            delta2_rightLeg = normalized(delta2_rightLeg)

            # mask
            delta2 = (
                torch.tensor([1, 0, 0, 0], dtype=torch.float32)
                .cuda(seqA.device)
                .repeat(bs, self.num_joint, 1)
            )
            delta2[:, leftArm_joints, :] = delta2_leftArm
            delta2[:, rightArm_joints, :] = delta2_rightArm
            delta2[:, leftLeg_joints, :] = delta2_leftLeg
            delta2[:, rightLeg_joints, :] = delta2_rightLeg
            delta_qg.append(delta2)

            # balacing gate
            bala_gate = self.weights_dec(refB_feed, shapeB, qB_base_norm)

            qB_hat = q_mul_q(qB_base, delta2)

            one_w = np.random.binomial(1, p=0.4)
            if one_w:
                bala_gate = torch.ones(bala_gate.shape, dtype=torch.float32).cuda(
                    seqA.device
                )

            qB_t = torch.lerp(qB_base, qB_hat, bala_gate[:, :, None] * k)

            B_quats_base.append(qB_base)
            B_quats_rt.append(qB_t)
            B_gates.append(bala_gate)

            # Forward Kinematics
            localB_t = FK.run(parents, refB, qB_t)
            localB_t = (localB_t - local_mean) / local_std
            B_locals_rt.append(localB_t)

            localB_base_t = FK.run(parents, refB, qB_base)
            localB_base_t = (localB_base_t - local_mean) / local_std
            B_locals_base.append(localB_base_t)

        # stack all frames
        quatB_rt = torch.stack(B_quats_rt, dim=1)  # shape: (batch_size, T, 22, 4)
        delta_qs = torch.stack(delta_qs, dim=1)
        localB_rt = torch.stack(B_locals_rt, dim=1)  # shape: (batch_size, T, 22, 3)
        quatB_base = torch.stack(B_quats_base, dim=1)
        bala_gates = torch.stack(B_gates, dim=1)  # shape: (batch_size, T, 22)
        delta_qg = torch.stack(delta_qg, dim=1)
        localB_base = torch.stack(B_locals_base, dim=1)  # shape: (batch_size, T, 22, 3)

        """ mapping global movements from A to B"""
        globalA_vel = seqA[:, :, -4:-1]
        globalA_rot = seqA[:, :, -1]
        normalized_vin = torch.cat(
            (
                torch.divide(globalA_vel, inp_height[:, :, None]),
                globalA_rot[:, :, None],
            ),
            dim=-1,
        )
        normalized_vout = normalized_vin.clone()

        globalB_vel = normalized_vout[:, :, :-1]
        globalB_rot = normalized_vout[:, :, -1]
        globalB_rt = torch.cat(
            (
                torch.multiply(globalB_vel, tgt_height[:, :, None]),
                globalB_rot[:, :, None],
            ),
            dim=-1,
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
                quatB_base,
                localB_base,
                bala_gates,
            )

        return localB_rt, globalB_rt, quatB_rt, delta_qs, delta_qg

    @staticmethod
    def get_recon_loss(
        atte_lst, num_joint, ae_reg, mask, localB_rt, localB_gt, quatA_gt, quatB_rt
    ):
        attW = torch.ones(num_joint).cuda(localB_rt.device)
        attW[atte_lst] = 2

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
    def get_cons_loss(
        atte_lst, num_joint, mask, localB_rt, localB_base, quatB_rt, quatB_base
    ):
        attW = torch.ones(num_joint).cuda(localB_rt.device)
        attW[atte_lst] = 2

        ae_joints_err = torch.sum(
            (
                torch.multiply(
                    mask[:, :, None, None], torch.subtract(localB_rt, localB_base)
                )
            )
            ** 2,
            dim=[0, 1, 3],
        )
        local_ae_loss = torch.sum(attW * ae_joints_err)
        local_ae_loss = torch.divide(
            local_ae_loss,
            torch.maximum(torch.sum(mask), torch.tensor(1).cuda(localB_base.device)),
        )

        quat_ae_loss = torch.sum(
            (
                torch.multiply(
                    mask[:, :, None, None], torch.subtract(quatB_rt, quatB_base)
                )
            )
            ** 2,
            dim=[0, 1, 3],
        )
        quat_ae_loss = torch.sum(attW * quat_ae_loss)
        quat_ae_loss = torch.divide(
            quat_ae_loss,
            torch.maximum(torch.sum(mask), torch.tensor(1).cuda(localB_base.device)),
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
    def get_rela_loss(rA, rB, mask):
        # bs T n n
        rela_loss = torch.sum(
            (torch.multiply(mask[:, :, None, None], torch.subtract(rA, rB))) ** 2
        )
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

    @staticmethod
    def get_rep_loss_part(
        parents,
        quatB_rt,
        rest_skelB,
        meshB,
        skinB_weights,
        body_vertices_lst,
        head_vertices_lst,
        leftarm_vertices_lst,
        rightarm_vertices_lst,
        leftleg_vertices_lst,
        rightleg_vertices_lst,
        ifth=False,
    ):
        '''
        quatB_rt: (T, 22, 4)
        meshB: (vertex_num, 3)
        skinB_weights: (vertex_num, bone_num)
        '''
        get_sdf = SDF()
        vertices_lbs = linear_blend_skinning(
            parents, quatB_rt, rest_skelB, meshB, skinB_weights
        )

        scale_factor = 0.2
        boxes = get_bounding_boxes(vertices_lbs)
        boxes_center = boxes.mean(dim=1).unsqueeze(dim=1)
        boxes_scale = (
            (1 + scale_factor)
            * 0.5
            * (boxes[:, 1] - boxes[:, 0]).max(dim=-1)[0][:, None, None]
        )
        vertices_centered = vertices_lbs - boxes_center
        vertices_centered_scaled = vertices_centered / boxes_scale
        assert vertices_centered_scaled.min() >= -1
        assert vertices_centered_scaled.max() <= 1

        T = vertices_lbs.shape[0]

        rep_loss_la, rep_loss_ra, rep_loss_ll, rep_loss_rl = 0, 0, 0, 0

        vertices_centered_scaled_la = vertices_centered_scaled[
            :, leftarm_vertices_lst, :
        ]
        vertices_centered_scaled_ra = vertices_centered_scaled[
            :, rightarm_vertices_lst, :
        ]
        vertices_centered_scaled_ll = vertices_centered_scaled[
            :, leftleg_vertices_lst, :
        ]
        vertices_centered_scaled_rl = vertices_centered_scaled[
            :, rightleg_vertices_lst, :
        ]

        for i in range(T):
            body_vertices = (
                vertices_centered_scaled[i, body_vertices_lst, :].detach().cpu().numpy()
            )
            body_point_cloud = trimesh.points.PointCloud(vertices=body_vertices)
            body_mesh = body_point_cloud.convex_hull

            body_vertices = torch.from_numpy(
                np.array(body_mesh.vertices).astype(np.single)
            ).cuda(quatB_rt.device)
            body_faces = torch.from_numpy(
                np.array(body_mesh.faces).astype(np.int32)
            ).cuda(quatB_rt.device)

            body_sdf = get_sdf(
                body_faces,
                body_vertices[None,],
                grid_size=32,
            )

            head_vertices = (
                vertices_centered_scaled[i, head_vertices_lst, :].detach().cpu().numpy()
            )
            head_point_cloud = trimesh.points.PointCloud(vertices=head_vertices)
            head_mesh = head_point_cloud.convex_hull

            head_vertices = torch.from_numpy(
                np.array(head_mesh.vertices).astype(np.single)
            ).cuda(quatB_rt.device)
            head_faces = torch.from_numpy(
                np.array(head_mesh.faces).astype(np.int32)
            ).cuda(quatB_rt.device)

            head_sdf = get_sdf(
                head_faces,
                head_vertices[None,],
                grid_size=32,
            )

            total_sdf = body_sdf + head_sdf

            vertices_local_la = vertices_centered_scaled_la[i, :, :]
            vert_num = vertices_local_la.shape[0]
            vertices_grid_la = vertices_local_la.view(1, -1, 1, 1, 3)
            phi_val_la = (
                nn.functional.grid_sample(
                    total_sdf[0][None, None], vertices_grid_la, align_corners=False
                )
                .view(-1)
                .sum()
                / vert_num
            ) * 1000
            if ifth:
                phi_val_la = (
                    phi_val_la
                    if phi_val_la > 3.0
                    else torch.tensor(0, dtype=torch.float32, requires_grad=True).cuda(
                        phi_val_la.device
                    )
                )
            rep_loss_la += phi_val_la

            vertices_local_ra = vertices_centered_scaled_ra[i, :, :]
            vert_num = vertices_local_ra.shape[0]
            vertices_grid_ra = vertices_local_ra.view(1, -1, 1, 1, 3)
            phi_val_ra = (
                nn.functional.grid_sample(
                    total_sdf[0][None, None], vertices_grid_ra, align_corners=False
                )
                .view(-1)
                .sum()
                / vert_num
            ) * 1000
            if ifth:
                phi_val_ra = (
                    phi_val_ra
                    if phi_val_ra > 6.0
                    else torch.tensor(0, dtype=torch.float32, requires_grad=True).cuda(
                        phi_val_la.device
                    )
                )
            rep_loss_ra += phi_val_ra

            vertices_local_ll = vertices_centered_scaled_ll[i, :, :]
            vert_num = vertices_local_ll.shape[0]
            vertices_grid_ll = vertices_local_ll.view(1, -1, 1, 1, 3)
            phi_val_ll = (
                nn.functional.grid_sample(
                    total_sdf[0][None, None], vertices_grid_ll, align_corners=False
                )
                .view(-1)
                .sum()
                / vert_num
            ) * 1000
            if ifth:
                phi_val_ll = (
                    phi_val_ll
                    if phi_val_ll > 10.0
                    else torch.tensor(0, dtype=torch.float32, requires_grad=True).cuda(
                        phi_val_la.device
                    )
                )
            rep_loss_ll += phi_val_ll

            vertices_local_rl = vertices_centered_scaled_rl[i, :, :]
            vert_num = vertices_local_rl.shape[0]
            vertices_grid_rl = vertices_local_rl.view(1, -1, 1, 1, 3)
            phi_val_rl = (
                nn.functional.grid_sample(
                    total_sdf[0][None, None], vertices_grid_rl, align_corners=False
                )
                .view(-1)
                .sum()
                / vert_num
            ) * 1000
            if ifth:
                phi_val_rl = (
                    phi_val_rl
                    if phi_val_rl > 15.0
                    else torch.tensor(0, dtype=torch.float32, requires_grad=True).cuda(
                        phi_val_la.device
                    )
                )
            rep_loss_rl += phi_val_rl

        return rep_loss_la / T, rep_loss_ra / T, rep_loss_ll / T, rep_loss_rl / T

    @staticmethod
    def get_att_loss(
        parents,
        quatB_rt,
        rest_skelB,
        meshB,
        skinB_weights,
        body_vertices_lst,
        arm_end_vertices_lst,
    ):
        '''
        quatB_rt: (T, 22, 4)
        meshB: (vertex_num, 3)
        skinB_weights: (vertex_num, bone_num)
        '''
        get_sdf_out = SDF2()
        vertices_lbs = linear_blend_skinning(
            parents, quatB_rt, rest_skelB, meshB, skinB_weights
        )

        scale_factor = 0.2
        boxes = get_bounding_boxes(vertices_lbs)
        boxes_center = boxes.mean(dim=1).unsqueeze(dim=1)
        boxes_scale = (
            (1 + scale_factor)
            * 0.5
            * (boxes[:, 1] - boxes[:, 0]).max(dim=-1)[0][:, None, None]
        )
        vertices_centered = vertices_lbs - boxes_center
        vertices_centered_scaled = vertices_centered / boxes_scale
        assert vertices_centered_scaled.min() >= -1
        assert vertices_centered_scaled.max() <= 1

        T = vertices_lbs.shape[0]

        att_loss_h = 0
        vertices_centered_scaled_h = vertices_centered_scaled[
            :, arm_end_vertices_lst, :
        ]

        for i in range(T):
            body_vertices = (
                vertices_centered_scaled[i, body_vertices_lst, :].detach().cpu().numpy()
            )
            body_point_cloud = trimesh.points.PointCloud(vertices=body_vertices)
            body_mesh = body_point_cloud.convex_hull

            body_vertices = torch.from_numpy(
                np.array(body_mesh.vertices).astype(np.single)
            ).cuda(quatB_rt.device)
            body_faces = torch.from_numpy(
                np.array(body_mesh.faces).astype(np.int32)
            ).cuda(quatB_rt.device)

            body_sdf_out = get_sdf_out(
                body_faces,
                body_vertices[None,],
                grid_size=32,
            )

            vertices_local_h = vertices_centered_scaled_h[i, :, :]
            vert_num = vertices_local_h.shape[0]
            vertices_grid_h = vertices_local_h.view(1, -1, 1, 1, 3)
            phi_val_rh = (
                nn.functional.grid_sample(
                    body_sdf_out[0][None, None], vertices_grid_h, align_corners=False
                )
                .view(-1)
                .sum()
                / vert_num
            ) * 1000
            att_loss_h += phi_val_rh

        return att_loss_h / T

    @staticmethod
    def get_regularization_loss(weights_sp, mask):
        bs = weights_sp.shape[0]
        regular_weights_loss = torch.sum(
            torch.multiply(mask[:, :, None], weights_sp**2)
        ) / (2 * bs)
        return regular_weights_loss


def get_bounding_boxes(vertices):
    num_people = vertices.shape[0]
    boxes = torch.zeros(num_people, 2, 3, device=vertices.device)
    for i in range(num_people):
        boxes[i, 0, :] = vertices[i].min(dim=0)[0]
        boxes[i, 1, :] = vertices[i].max(dim=0)[0]
    return boxes


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
