import torch
import numpy as np


def transforms_rotations(rotations):
    q_length = torch.sqrt(torch.sum(torch.square(rotations), dim=-1))
    qw = rotations[..., 0] / q_length
    qx = rotations[..., 1] / q_length
    qy = rotations[..., 2] / q_length
    qz = rotations[..., 3] / q_length
    """Unit quaternion based rotation matrix computation"""
    x2 = qx + qx
    y2 = qy + qy
    z2 = qz + qz
    xx = qx * x2
    yy = qy * y2
    wx = qw * x2
    xy = qx * y2
    yz = qy * z2
    wy = qw * y2
    xz = qx * z2
    zz = qz * z2
    wz = qw * z2

    dim0 = torch.stack([1.0 - (yy + zz), xy - wz, xz + wy], dim=-1)
    dim1 = torch.stack([xy + wz, 1.0 - (xx + zz), yz - wx], dim=-1)
    dim2 = torch.stack([xz - wy, yz + wx, 1.0 - (xx + yy)], dim=-1)
    m = torch.stack([dim0, dim1, dim2], dim=-2)

    return m


def repr6d2mat(repr):
    x = repr[..., :3]
    y = repr[..., 3:]
    x = x / x.norm(dim=-1, keepdim=True)
    z = torch.cross(x, y)
    z = z / z.norm(dim=-1, keepdim=True)
    y = torch.cross(z, x)
    res = [x, y, z]
    res = [v.unsqueeze(-2) for v in res]
    mat = torch.cat(res, dim=-2)
    return mat


def linear_blend_skinning(parents, quat, rest_skel, mesh_vertices, skin_weights):
    '''
    quat: (T, 22, 4)
    bone_position: (T, 22, 3)
    rest_skel: (22, 3)
    mesh_vertices: (vertex_num, 3)    rest pose
    skin_weights: (vertex_num, 22)
    '''
    rot_mat = transforms_rotations(quat)  # (T, 22, 3, 3)
    # rot_mat = torch.cat([rot_mat,
    #                         torch.zeros((rot_mat.shape[0], rot_mat.shape[1], rot_mat.shape[2], 1), dtype=torch.float).cuda(quat.device)], dim=-1)
    rot_mat = torch.cat(
        [rot_mat, rest_skel[None, :, :, None].repeat(quat.shape[0], 1, 1, 1)], dim=-1
    )
    rot_mat = torch.cat(
        [
            rot_mat,
            torch.tensor([0, 0, 0, 1], dtype=torch.float)
            .repeat(rot_mat.shape[0], rot_mat.shape[1], 1, 1)
            .cuda(quat.device),
        ],
        dim=-2,
    )  # rot_mat: (T, 22, 4, 4)

    rest_pose_mat = torch.cat(
        [
            torch.eye(3, dtype=torch.float)
            .repeat(rest_skel.shape[0], 1, 1)
            .cuda(quat.device),
            rest_skel.unsqueeze(-1),
        ],
        dim=-1,
    )
    rest_pose_mat = torch.cat(
        [
            rest_pose_mat,
            torch.tensor([0, 0, 0, 1], dtype=torch.float)
            .repeat(rest_pose_mat.shape[0], 1, 1)
            .cuda(quat.device),
        ],
        dim=-2,
    )  # rest_pose_mat: (22, 4, 4)

    num_bone = rest_skel.shape[0]

    bone_matrix_rest = rest_pose_mat.clone()
    bone_matrix_rot = rot_mat.clone()

    # ******************************************** fail to compute gradient **************************************************
    # for i in range(1, num_bone):
    #     bone_matrix_rest[i, :, :] = torch.matmul(bone_matrix_rest[parents[i],:,:], bone_matrix_rest[i,:,:])
    #     bone_matrix_rot[:, i, :, :] = torch.matmul(bone_matrix_rot[:,parents[i],:,:], bone_matrix_rot[:,i,:,:])

    bone_matrix_rest_lst = list(torch.split(bone_matrix_rest, 1, dim=0))
    bone_matrix_rot_lst = list(torch.split(bone_matrix_rot, 1, dim=1))

    for i in range(1, num_bone):
        bone_matrix_rest_lst[i] = torch.matmul(
            bone_matrix_rest_lst[parents[i]][:, :, :], bone_matrix_rest_lst[i][:, :, :]
        )
        bone_matrix_rot_lst[i] = torch.matmul(
            bone_matrix_rot_lst[parents[i]][:, :, :, :],
            bone_matrix_rot_lst[i][:, :, :, :],
        )

    bone_matrix_rest = torch.cat(bone_matrix_rest_lst, dim=0)
    bone_matrix_rot = torch.cat(bone_matrix_rot_lst, dim=1)

    bone_matrix_rest_inverse = bone_matrix_rest.inverse().repeat(
        bone_matrix_rot.shape[0], 1, 1, 1
    )
    bone_matrix_word = torch.einsum(
        'tjmn,tjnk->tjmk', bone_matrix_rot, bone_matrix_rest_inverse
    )

    rest_root_x, rest_root_y, rest_root_z = (
        rest_skel[0, 0],
        rest_skel[0, 1],
        rest_skel[0, 2],
    )
    root_recover = (
        torch.tensor([rest_root_x, rest_root_y, rest_root_z], dtype=torch.float)
        .unsqueeze(-1)
        .cuda(quat.device)
    )
    root_recover_matrix = torch.cat(
        [torch.eye(3, dtype=torch.float).cuda(quat.device), root_recover], dim=-1
    )
    root_recover_matrix = torch.cat(
        [
            root_recover_matrix,
            torch.tensor([0, 0, 0, 1], dtype=torch.float)
            .unsqueeze(0)
            .cuda(quat.device),
        ],
        dim=0,
    )  # (4,4)
    root_recover_matrix = root_recover_matrix.repeat(
        bone_matrix_word.shape[0], bone_matrix_word.shape[1], 1, 1
    )  # (T, 22, 4, 4)

    bone_matrix_word_tran_recover = torch.einsum(
        'tjmn,tjnk->tjmk', bone_matrix_word, root_recover_matrix
    )

    T = quat.shape[0]
    verts_rest = torch.cat(
        [
            mesh_vertices,
            torch.ones((mesh_vertices.shape[0], 1), dtype=torch.float).cuda(
                quat.device
            ),
        ],
        dim=-1,
    )  # (vertex_num, 4)
    verts_lbs = torch.zeros((T, mesh_vertices.shape[0], 4)).cuda(quat.device)
    for i in range(T):
        for j in range(num_bone):
            tfs = bone_matrix_word_tran_recover[i, j, :, :]  # (4，4)
            weight = skin_weights[:, j].unsqueeze(1)  # (vertex_num, 1)
            verts_lbs[i, :, :] += weight * tfs.matmul(
                verts_rest.transpose(0, 1)
            ).transpose(0, 1)

    verts_lbs = verts_lbs[:, :, :3]

    return verts_lbs
