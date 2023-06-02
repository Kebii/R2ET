import numpy as np
import torch
import torch.nn as nn


class qlinear(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True):
        super(qlinear, self).__init__()
        W_ = nn.init.normal_(
            torch.zeros((out_channel, in_channel), dtype=torch.float32)
        )
        self.W = nn.Parameter(W_)
        if bias:
            #    bias_ = np.tile(np.array([1, 0, 0, 0]), out_channel // 4 - 1)
            #    bias_ = np.concatenate([bias_, np.array([0, 0, 0, 0])])
            bias_ = np.tile(np.array([1, 0, 0, 0]), out_channel // 4)
            self.bias = nn.Parameter(torch.from_numpy(bias_.astype(np.float32)))
        else:
            self.bias = 0

    def forward(self, x):
        y = nn.functional.linear(x, self.W, self.bias)
        return y


def lrelu(x, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * torch.abs(x)


def get_vels(input_, n_joints, dmean, dstd, omean, ostd):
    joints = (
        torch.reshape(input_[:, :, :-4], input_.shape[:-1].as_list() + [n_joints, 3])
        * dstd[None]
        + dmean[None]
    )
    root_x = input_[:, :, -4] * ostd[0, 0] + omean[0, 0]
    root_y = input_[:, :, -3] * ostd[0, 1] + omean[0, 1]
    root_z = input_[:, :, -2] * ostd[0, 2] + omean[0, 2]
    root_r = input_[:, :, -1] * ostd[0, 3] + omean[0, 3]

    rotation = np.repeat(
        np.array([[[1.0, 0.0, 0.0, 0.0]]]), int(input_.shape[0]), axis=0
    ).astype("float32")
    rotation = torch.from_numpy(rotation).cuda(joints.device)
    translation = np.repeat(
        np.array([[[0.0, 0.0, 0.0]]]), int(input_.shape[0]), axis=0
    ).astype("float32")
    translation = torch.from_numpy(translation).cuda(joints.device)
    axis = np.repeat(np.array([[0.0, 1.0, 0.0]]), int(input_.shape[0]), axis=0).astype(
        "float32"
    )
    axis = torch.from_numpy(axis).cuda(joints.device)
    joints_list = []

    for t in range(int(joints.shape[1])):
        joints_list.append(q_mul_v(rotation, joints[:, t, :, :]))
        joints_x = joints_list[-1][:, :, 0:1] + translation[:, 0:1, 0:1]
        joints_y = joints_list[-1][:, :, 1:2] + translation[:, 0:1, 1:2]
        joints_z = joints_list[-1][:, :, 2:3] + translation[:, 0:1, 2:3]
        joints_list[-1] = torch.cat([joints_x, joints_y, joints_z], dim=-1)

        rotation = q_mul_q(from_angle_axis(-root_r[:, t], axis), rotation)

        translation += q_mul_v(
            rotation,
            torch.cat(
                [root_x[:, t : t + 1], root_y[:, t : t + 1], root_z[:, t : t + 1]],
                dim=-1,
            )[:, None, :],
        )

    joints = torch.reshape(
        torch.stack(joints_list, dim=1), input_.shape[:-1].as_list() + [-1]
    )
    return joints[:, 1:, :] - joints[:, :-1, :], joints  # [:,:-1,:]


def q_mul_v(a, b):
    vs = torch.cat([torch.zeros(list(b.shape[:-1]) + [1]).cuda(b.device), b], dim=-1)
    return q_mul_q(a, q_mul_q(vs, q_neg(a)))[..., 1:4]


def q_mul_q(a, b):
    sqs, oqs = q_broadcast(a, b)

    q0 = sqs[..., 0:1]
    q1 = sqs[..., 1:2]
    q2 = sqs[..., 2:3]
    q3 = sqs[..., 3:4]
    r0 = oqs[..., 0:1]
    r1 = oqs[..., 1:2]
    r2 = oqs[..., 2:3]
    r3 = oqs[..., 3:4]

    qs0 = r0 * q0 - r1 * q1 - r2 * q2 - r3 * q3
    qs1 = r0 * q1 + r1 * q0 - r2 * q3 + r3 * q2
    qs2 = r0 * q2 + r1 * q3 + r2 * q0 - r3 * q1
    qs3 = r0 * q3 - r1 * q2 + r2 * q1 + r3 * q0

    return torch.cat([qs0, qs1, qs2, qs3], dim=-1)


def q_neg(a):
    return a * torch.tensor([[[1, -1, -1, -1]]]).cuda(a.device)


def q_broadcast(sqs, oqs):
    if int(sqs.shape[-2]) == 1:
        sqsn = []
        for l in range(oqs.shape[-2]):
            sqsn.append(sqs)
        sqs = torch.cat(sqsn, dim=-2)

    if int(oqs.shape[-2]) == 1:
        oqsn = []
        for l in range(sqs.shape[-2]):
            oqsn.append(oqs)
        oqs = torch.cat(oqsn, dim=-2)

    return sqs, oqs


def from_angle_axis(angles, axis):
    axis = axis / (torch.sqrt(torch.sum(axis**2, dim=-1)) + 1e-10)[..., None]
    sines = torch.sin(angles / 2.0)[..., None]
    cosines = torch.cos(angles / 2.0)[..., None]
    return torch.cat([cosines, axis * sines], dim=-1)[:, None, :]


def gaussian_noise(input_, input_mean, input_std, stddev):
    noise = torch.normal(size=input_.shape, mean=0.0, std=stddev)
    noisy_input = noise + input_ * input_std + input_mean
    return (noisy_input - input_mean) / input_std


def get_wjs(local_motion, global_motion):
    """
    input:
        local_motion shape: (batch_size, max_len, 22, 3)
        global_motion shape: (batch_size, max_len, 4)
    output:
        wjs shape: (batch_size, max_len, 66)
    """
    joints = local_motion

    root_x = global_motion[:, :, -4]
    root_y = global_motion[:, :, -3]
    root_z = global_motion[:, :, -2]
    root_r = global_motion[:, :, -1]

    rotation = np.repeat(
        np.array([[[1.0, 0.0, 0.0, 0.0]]]), int(local_motion.shape[0]), axis=0
    ).astype("float32")
    rotation = torch.from_numpy(rotation).cuda(local_motion.device)
    translation = np.repeat(
        np.array([[[0.0, 0.0, 0.0]]]), int(local_motion.shape[0]), axis=0
    ).astype("float32")
    translation = torch.from_numpy(translation).cuda(local_motion.device)
    axis = np.repeat(
        np.array([[0.0, 1.0, 0.0]]), int(local_motion.shape[0]), axis=0
    ).astype("float32")
    axis = torch.from_numpy(axis).cuda(local_motion.device)
    joints_list = []

    for t in range(int(joints.shape[1])):
        joints_list.append(q_mul_v(rotation, joints[:, t, :, :]))
        joints_x = joints_list[-1][:, :, 0:1] + translation[:, 0:1, 0:1]
        joints_y = joints_list[-1][:, :, 1:2] + translation[:, 0:1, 1:2]
        joints_z = joints_list[-1][:, :, 2:3] + translation[:, 0:1, 2:3]
        joints_list[-1] = torch.cat([joints_x, joints_y, joints_z], dim=-1)

        rotation = q_mul_q(from_angle_axis(-root_r[:, t], axis), rotation)

        translation += q_mul_v(
            rotation,
            torch.cat(
                [root_x[:, t : t + 1], root_y[:, t : t + 1], root_z[:, t : t + 1]],
                dim=-1,
            )[:, None, :],
        )

    wjs = torch.reshape(
        torch.stack(joints_list, dim=1), list(local_motion.shape[:-2]) + [-1]
    )
    return wjs


def slerp(q0, q1, weight, DOT_THRESHOLD=0.9995):
    '''
    q0, q1: (4) normalized
    weight: (1)
    '''
    dot = torch.sum(q0 * q1)
    if torch.abs(dot) > DOT_THRESHOLD:
        return torch.lerp(q0, q1, weight)
    # Calculate initial angle between v0 and v1
    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)
    # Angle at timestep t
    theta_t = theta_0 * weight
    sin_theta_t = torch.sin(theta_t)
    # Finish the slerp algorithm
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    res = s0 * q0 + s1 * q1
    return res


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return normalize_digraph(A)


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD
