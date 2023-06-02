import os

import torch
import torch.nn as nn
import numpy as np

from src.forward_kinematics import FK
from src.ops import qlinear, q_mul_q
from src.linear_blend_skin import linear_blend_skinning
from torch import atan2
from torch import asin
from collections import OrderedDict
import trimesh
from sdf import SDF, SDF2


class PoseEncoder(nn.Module):
    def __init__(self, num_joint, hidden_channels, kp):
        super(PoseEncoder, self).__init__()

        self.seq = nn.Sequential(
            OrderedDict(
                [
                    ('hidden1', nn.Linear(3 * num_joint, hidden_channels)),
                    ('relu1', nn.ReLU()),
                    ('dropout1', nn.Dropout(p=1 - kp)),
                    ('hidden2', nn.Linear(hidden_channels, hidden_channels)),
                    ('relu2', nn.ReLU()),
                    ('dropout2', nn.Dropout(p=1 - kp)),
                    ('hidden3', nn.Linear(hidden_channels, hidden_channels)),
                    ('relu3', nn.ReLU()),
                    ('dropout3', nn.Dropout(p=1 - kp)),
                    ('hidden4', nn.Linear(hidden_channels, hidden_channels)),
                    ('relu4', nn.ReLU()),
                ]
            )
        )

    def forward(self, pose_t):
        # pose_t: bs joint*3
        return self.seq(pose_t)


class QEncoder(nn.Module):
    def __init__(self, num_joint, in_channels):
        super(QEncoder, self).__init__()
        self.qlinear = qlinear(in_channels, 4 * num_joint)

    def forward(self, x):
        # x: bs 512
        y = self.qlinear(x)
        return y


class DeltaEncoder(nn.Module):
    def __init__(self, num_joint, embed_channels, hidden_channels, kp):
        super(DeltaEncoder, self).__init__()
        self.num_joint = num_joint

        self.ref_embed = nn.Linear(3 * num_joint, embed_channels)
        self.ref_acti = nn.Sigmoid()
        self.ref_drop = nn.Dropout(p=1 - kp)

        self.delta_linear = qlinear(hidden_channels + embed_channels, 4 * num_joint)

    def forward(self, ref, x):
        # ref: bs joint*3
        # x: bs 512
        ref_embed = self.ref_drop(self.ref_acti(self.ref_embed(ref)))
        x_cat = torch.cat([x, ref_embed], dim=-1)
        deltaq_t = self.delta_linear(x_cat)

        return deltaq_t


class MEncoder(nn.Module):
    def __init__(self, hidden_channels):
        super(MEncoder, self).__init__()
        pad = int((3 - 1) / 2)
        self.conv1 = nn.Conv1d(4, hidden_channels, kernel_size=3, padding=pad)
        self.lrelu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv1d(hidden_channels, 4, kernel_size=3, padding=pad)

    def forward(self, m):
        # m: bs 4 T
        return self.conv2(self.lrelu(self.conv1(m)))


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


class PMNet(nn.Module):
    def __init__(
        self,
        num_joint=22,
        hidden_channels_p=512,
        embed_channels_p=16,
        hidden_channels_m=128,
        kp=0.8,
    ):
        super(PMNet, self).__init__()
        self.num_joint = num_joint
        self.p_enc = PoseEncoder(num_joint, hidden_channels_p, kp)
        self.q_enc = QEncoder(num_joint, hidden_channels_p)
        self.delta_enc = DeltaEncoder(
            num_joint, embed_channels_p, hidden_channels_p, kp
        )
        self.m_enc = MEncoder(hidden_channels_m)

    def forward(
        self,
        seqA,
        seqB,
        skelA,
        skelB,
        inp_height,
        tgt_height,
        local_mean,
        local_std,
        parents,
    ):
        '''
        seqA, seqB: bs T joints*3+4
        skelA, skelB: bs T joints*3
        height: bs 1
        '''
        self.parents = parents
        bs = seqA.size(0)
        T = seqA.size(1)
        local_mean = torch.from_numpy(local_mean).cuda(seqA.device)
        local_std = torch.from_numpy(local_std).cuda(seqA.device)
        parents = torch.from_numpy(parents).cuda(seqA.device)

        t_poseB = torch.reshape(skelB[:, 0, :], [bs, self.num_joint, 3])
        t_poseB = t_poseB * local_std + local_mean
        refB = t_poseB
        refB_feed = skelB[:, 0, :]

        A_pose_features = []
        B_pose_features = []
        A_locals_ik = []
        A_quats_ik = []
        delta_qs = []
        B_locals_rt = []
        B_quats_rt = []

        # ========================= Mapping pose from A to B ======================= #
        for t in range(T):
            """Pose Encoder for A"""
            inputA_t = seqA[:, t, :-4]
            faiA_t = self.p_enc(inputA_t)
            A_pose_features.append(faiA_t)

            """ Inverse Kinematics for A """
            qoutA_t = self.q_enc(faiA_t)
            qoutA_t = torch.reshape(qoutA_t, [bs, self.num_joint, 4])
            qoutA_t = normalized(qoutA_t)
            A_quats_ik.append(qoutA_t)

            """ Mapping A to B """
            deltaq_t = self.delta_enc(refB_feed, faiA_t)
            deltaq_t = torch.reshape(deltaq_t, [bs, self.num_joint, 4])
            deltaq_t = normalized(deltaq_t)
            delta_qs.append(deltaq_t)

            """ Hamilton product """
            qB_t = q_mul_q(qoutA_t, deltaq_t)
            B_quats_rt.append(qB_t)

            """ Forward Kinematics for B """
            localB_out_t = FK.run(parents, refB, qB_t)
            localB_out_t = (localB_out_t - local_mean) / local_std
            B_locals_rt.append(localB_out_t)

        """ Feed-forward Local results """
        quatA_ik = torch.stack(A_quats_ik, dim=1)  # shape: (batch_size, T, 22, 4)
        quatB_rt = torch.stack(B_quats_rt, dim=1)  # shape: (batch_size, T, 22, 4)
        deltaQ = torch.stack(delta_qs, dim=1)
        localB_rt = torch.stack(B_locals_rt, dim=1)  # shape: (batch_size, T, 22, 3)
        A_features = torch.stack(A_pose_features, dim=1)  # shape: (batch_size, T, 512)

        # ================== Mapping overall movements from A to B ================= #
        gA_vel = seqA[:, :, -4:-1]
        gA_rot = seqA[:, :, -1]

        """ Normalize """
        normalized_vin = torch.cat(
            (torch.divide(gA_vel, inp_height[:, :, None]), gA_rot[:, :, None]), dim=-1
        )
        normalized_vin = normalized_vin.permute(
            0, 2, 1
        ).contiguous()  # bs T 4  to  bs 4 T

        """ Movement Regressor """
        normalized_vout = self.m_enc(normalized_vin)
        normalized_vout = normalized_vout.permute(
            0, 2, 1
        ).contiguous()  # bs 4 T  to  bs T 4
        normalized_vin = normalized_vin.permute(
            0, 2, 1
        ).contiguous()  # bs 4 T  to  bs T 4

        """ De-normalize """
        gB_vel = normalized_vout[:, :, :-1]
        gB_rot = normalized_vout[:, :, -1]
        globalB_rt = torch.cat(
            (torch.multiply(gB_vel, tgt_height[:, :, None]), gB_rot[:, :, None]), dim=-1
        )  # shape: (batch_size, T, 4)

        if self.training:
            t_poseA = torch.reshape(skelA[:, 0, :], [bs, self.num_joint, 3])
            t_poseA = t_poseA * local_std + local_mean
            refA = t_poseA
            for t in range(T):
                """Reconstruct A"""
                qoutA_t = quatA_ik[:, t, :, :]
                localA_out_t = FK.run(parents, refA, qoutA_t)
                localA_out_t = (
                    localA_out_t - local_mean
                ) / local_std  # shape: (batch_size, 22, 3)
                A_locals_ik.append(localA_out_t)

                """ Pose Encoder again for B """
                B_locals_rt = localB_rt[:, t, :, :]
                inputB_t = torch.reshape(B_locals_rt, [bs, -1]).type(torch.float32)
                faiB_t = self.p_enc(inputB_t)
                B_pose_features.append(faiB_t)

            localA_ik = torch.stack(A_locals_ik, dim=1)

            localA_gt = torch.reshape(seqA[:, :, :-4], [bs, T, self.num_joint, 3])
            localB_gt = torch.reshape(seqB[:, :, :-4], [bs, T, self.num_joint, 3])
            globalA_gt = seqA[:, :, -4:]

            localA_gt = torch.reshape(seqA[:, :, :-4], [bs, T, self.num_joint, 3])
            localB_gt = torch.reshape(seqB[:, :, :-4], [bs, T, self.num_joint, 3])

            B_features = torch.stack(
                B_pose_features, dim=1
            )  # shape: (batch_size, max_len, 512)
            return (
                localA_ik,
                localA_gt,
                localB_rt,
                localB_gt,
                globalA_gt,
                globalB_rt,
                normalized_vin,
                normalized_vout,
                A_features,
                B_features,
                quatA_ik,
                quatB_rt,
            )

        return localB_rt, globalB_rt, quatB_rt, deltaQ

    @staticmethod
    def get_prec_loss(posefeatureA, posefeatureB, ae_reg, mask):
        prec_loss = torch.sum(
            torch.square(
                torch.multiply(
                    (1 - ae_reg[:, :, None]) * mask[:, :, None],
                    torch.subtract(posefeatureA, posefeatureB),
                )
            )
        )
        prec_loss = torch.divide(prec_loss, torch.sum(mask))
        return prec_loss

    @staticmethod
    def get_recon_loss(
        atte_lst,
        num_joint,
        ae_reg,
        mask,
        localA_ik,
        localA_gt,
        localB_rt,
        localB_gt,
        normalized_vin,
        normalized_vout,
    ):
        attW = torch.ones(num_joint).cuda(localA_ik.device)
        attW[atte_lst] = 2
        joints_err = torch.sum(
            torch.square(
                torch.multiply(
                    mask[:, :, None, None], torch.subtract(localA_ik, localA_gt)
                )
            ),
            dim=[0, 1, 3],
        )

        IK_loss = torch.sum(attW * joints_err)
        IK_loss = torch.divide(IK_loss, torch.sum(mask))

        """ For training stability, we chose the same character to the input with p = 0.5 """
        ae_joints_err = torch.sum(
            torch.square(
                torch.multiply(
                    ae_reg[:, :, None, None] * mask[:, :, None, None],
                    torch.subtract(localB_rt, localB_gt),
                )
            ),
            dim=[0, 1, 3],
        )
        local_ae_loss = torch.sum(attW * ae_joints_err)
        local_ae_loss = torch.divide(
            local_ae_loss,
            torch.maximum(
                torch.sum(ae_reg * mask), torch.tensor(1).cuda(ae_reg.device)
            ),
        )

        global_ae_loss = torch.sum(
            torch.square(
                torch.multiply(
                    ae_reg[:, :, None] * mask[:, :, None],
                    torch.subtract(normalized_vin, normalized_vout),
                )
            )
        )

        global_ae_loss = torch.divide(
            global_ae_loss,
            torch.maximum(
                torch.sum(ae_reg * mask), torch.tensor(1).cuda(ae_reg.device)
            ),
        )

        return IK_loss, local_ae_loss, global_ae_loss

    @staticmethod
    def get_rot_cons_loss(alpha, euler_ord, quatA_ik, quatB_rt):
        rads = alpha / 180.0
        twistA_loss = torch.mean(
            torch.square(
                torch.maximum(
                    torch.tensor(0).cuda(quatA_ik.device),
                    torch.abs(euler_y(quatA_ik, euler_ord)) - rads * np.pi,
                )
            )
        )
        twistB_loss = torch.mean(
            torch.square(
                torch.maximum(
                    torch.tensor(0).cuda(quatB_rt.device),
                    torch.abs(euler_y(quatB_rt, euler_ord)) - rads * np.pi,
                )
            )
        )
        twist_loss = twistA_loss + twistB_loss

        return twist_loss

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
    def get_smooth_loss(normalized_vout, global_mean, global_std, mask):
        dnorm_offA_ = normalized_vout * global_std + global_mean
        cycle_smooth = torch.sum(
            torch.square(
                torch.multiply(
                    mask[:, 1:, None],
                    dnorm_offA_[:, 1:, :-1] - dnorm_offA_[:, :-1, :-1],
                )
            )
        )
        cycle_smooth = torch.divide(cycle_smooth, torch.sum(mask))

        return cycle_smooth


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


def get_bounding_boxes(vertices):
    num_people = vertices.shape[0]
    boxes = torch.zeros(num_people, 2, 3, device=vertices.device)
    for i in range(num_people):
        boxes[i, 0, :] = vertices[i].min(dim=0)[0]
        boxes[i, 1, :] = vertices[i].max(dim=0)[0]
    return boxes


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


if __name__ == '__main__':
    model = PMNet()
    model.delta_enc.parameters()
