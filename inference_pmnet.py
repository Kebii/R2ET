import os
import sys

sys.path.append("./outside-code")
import time
import datetime
import random
import yaml
import argparse
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from os.path import exists, join
from os import listdir, makedirs

import statistics

from datasets.test_feeder_r2et import Feeder
from src.model_pmnet import PMNet as RetNet
import BVH as BVH
import Animation as Animation
from Quaternions import Quaternions
from src.utils import put_in_world_bvh
from src.utils import get_orient_start
from src.utils import put_in_world2, get_height, get_height_from_skel


def get_parser():
    # parameter priority: command line > config file > default
    parser = argparse.ArgumentParser(description='PMNet_pytorch')
    parser.add_argument(
        '--config',
        default='./config/inference_cfg.yaml',
        help='path to the configuration file',
    )
    parser.add_argument(
        '--save_path',
        default='./work_dir/inference/Robot_Hip_Hop_Dance.bvh',
        help='path to the configuration file',
    )
    parser.add_argument('--phase', default='test', help='must be train or test')
    parser.add_argument('--load_inp_data', type=dict, default=dict(), help='')
    parser.add_argument('--weights', default='', help='xxx.pt weights for generator')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing',
    )
    parser.add_argument(
        '--num_joint', type=int, default=22, help='number of the joints'
    )
    parser.add_argument(
        '--ret_model_args',
        type=dict,
        default=dict(),
        help='the arguments of retargetor',
    )

    return parser


def load_inp_data(
    device,
    inp_seq_path,
    inp_quat_path,
    inp_skel_path,
    tgt_skel_path,
    inp_shape_path,
    tgt_shape_path,
    stats_path,
    inp_bvh_path,
    tgt_bvh_path,
):
    inpskel = np.load(inp_skel_path)
    tgtskel = np.load(tgt_skel_path)
    inpquat = np.load(inp_quat_path)
    inseq = np.load(inp_seq_path)
    offset = inseq[:, -8:-4]
    inseq = np.reshape(inseq[:, :-8], [inseq.shape[0], -1, 3])
    # inseq = inseq[:, :-8]
    T = inpskel.shape[0]
    print("Sequence length:", T)

    inp_fbx_file = np.load(inp_shape_path)
    inp_body_width = inp_fbx_file['body_width'].astype(np.single)
    inp_arm_width = inp_fbx_file['arm_width'].astype(np.single)
    inp_full_width = inp_fbx_file['full_width'].astype(np.single)
    inp_joint_shape = inp_fbx_file['joint_shape'].astype(np.single)
    inp_shape = np.divide(inp_joint_shape, inp_full_width[None, :]).reshape(-1)

    tgt_fbx_file = np.load(tgt_shape_path)
    tgt_body_width = tgt_fbx_file['body_width'].astype(np.single)
    tgt_arm_width = tgt_fbx_file['arm_width'].astype(np.single)
    tgt_full_width = tgt_fbx_file['full_width'].astype(np.single)
    tgt_joint_shape = tgt_fbx_file['joint_shape'].astype(np.single)
    tgt_shape = np.divide(tgt_joint_shape, tgt_full_width[None, :]).reshape(-1)

    local_mean = np.load(join(stats_path, "mixamo_local_motion_mean.npy"))
    local_std = np.load(join(stats_path, "mixamo_local_motion_std.npy"))
    global_mean = np.load(join(stats_path, "mixamo_global_motion_mean.npy"))
    global_std = np.load(join(stats_path, "mixamo_global_motion_std.npy"))
    quat_mean = np.load(join(stats_path, "mixamo_quat_mean.npy"))
    quat_std = np.load(join(stats_path, "mixamo_quat_std.npy"))
    local_std[local_std == 0] = 1

    """ Height ratio """
    # Input sequence (un-normalize)
    inp_skel = inpskel[0, :].reshape([22, 3])

    # Ground truth
    out_skel = tgtskel[0, :].reshape([22, 3])

    inp_height = get_height(inp_skel) / 100
    tgt_height = get_height(out_skel) / 100

    inseq = (inseq - local_mean) / local_std
    # offset = (offset - global_mean) / global_std
    offset = offset
    tgtskel = (tgtskel - local_mean) / local_std
    inpskel = (inpskel - local_mean) / local_std

    inpquat = (inpquat - quat_mean) / quat_std

    inseq = inseq.reshape([inseq.shape[0], -1])
    inpskel = inpskel.reshape([inpskel.shape[0], -1])
    tgtskel = tgtskel.reshape([tgtskel.shape[0], -1])

    inp_seq = np.concatenate((inseq, offset), axis=-1)

    tgtanim, tgtnames, tgtftime = BVH.load(tgt_bvh_path)
    inpanim, inpnames, inpftime = BVH.load(inp_bvh_path)

    ibvh_file = open(tgt_bvh_path).read().split("JOINT")
    ibvh_joints = [
        f.split("\n")[0].split(":")[-1].split(" ")[-1] for f in ibvh_file[1:]
    ]
    ito_keep = [0]
    joints_list = [
        "Spine",
        "Spine1",
        "Spine2",
        "Neck",
        "Head",
        "LeftUpLeg",
        "LeftLeg",
        "LeftFoot",
        "LeftToeBase",
        "RightUpLeg",
        "RightLeg",
        "RightFoot",
        "RightToeBase",
        "LeftShoulder",
        "LeftArm",
        "LeftForeArm",
        "LeftHand",
        "RightShoulder",
        "RightArm",
        "RightForeArm",
        "RightHand",
    ]

    for jname in joints_list:
        for k in range(len(ibvh_joints)):
            if jname == ibvh_joints[k][-len(jname) :]:
                ito_keep.append(k + 1)
                break

    inp_seq = torch.from_numpy(inp_seq.astype(np.single))[None, :].cuda(device)
    inpskel = torch.from_numpy(inpskel.astype(np.single))[None, :].cuda(device)
    tgtskel = torch.from_numpy(tgtskel.astype(np.single))[None, :].cuda(device)
    inp_shape = torch.from_numpy(inp_shape.astype(np.single))[None, :].cuda(device)
    tgt_shape = torch.from_numpy(tgt_shape.astype(np.single))[None, :].cuda(device)
    inpquat = torch.from_numpy(inpquat.astype(np.single))[None, :].cuda(device)
    inp_height_ = torch.zeros((1, 1)).cuda(device)
    tgt_height_ = torch.zeros((1, 1)).cuda(device)
    inp_height_[0, 0] = inp_height
    tgt_height_[0, 0] = tgt_height

    return (
        inp_seq,
        inpskel,
        tgtskel,
        inp_shape,
        tgt_shape,
        inpquat,
        inp_height_,
        tgt_height_,
        local_mean,
        local_std,
        quat_mean,
        quat_std,
        global_mean,
        global_std,
        tgtanim,
        tgtnames,
        tgtftime,
        inpanim,
        inpnames,
        inpftime,
        ito_keep,
    )


def get_height(joints):
    return (
        np.sqrt(((joints[5, :] - joints[4, :]) ** 2).sum(axis=-1))
        + np.sqrt(((joints[4, :] - joints[3, :]) ** 2).sum(axis=-1))
        + np.sqrt(((joints[3, :] - joints[2, :]) ** 2).sum(axis=-1))
        + np.sqrt(((joints[2, :] - joints[1, :]) ** 2).sum(axis=-1))
        + np.sqrt(((joints[1, :] - joints[0, :]) ** 2).sum(axis=-1))
        + np.sqrt(((joints[6, :] - joints[7, :]) ** 2).sum(axis=-1))
        + np.sqrt(((joints[7, :] - joints[8, :]) ** 2).sum(axis=-1))
        + np.sqrt(((joints[8, :] - joints[9, :]) ** 2).sum(axis=-1))
    )


def getmodel(weight_path, arg):
    model = RetNet(**arg.ret_model_args).cuda(arg.device[0])
    model = nn.DataParallel(model, device_ids=arg.device)

    print("load weight from: " + weight_path)
    weights = torch.load(weight_path)
    model.load_state_dict(weights)
    model.eval()
    return model


def inference(ret_model, parents, arg):
    ret_model.eval()
    ret_model.requires_grad_(False)
    (
        inp_seq,
        inpskel,
        tgtskel,
        inp_shape,
        tgt_shape,
        inpquat,
        inp_height,
        tgt_height,
        local_mean,
        local_std,
        quat_mean,
        quat_std,
        global_mean,
        global_std,
        tgtanim,
        tgtname,
        tgtftime,
        inpanim,
        inpname,
        inpftime,
        tgtjoints,
    ) = load_inp_data(arg.device[0], **arg.load_inp_data)

    oursL, oursG, quatsB, delta_q = ret_model(
        inp_seq,
        None,
        inpskel,
        tgtskel,
        inp_height,
        tgt_height,
        local_mean,
        local_std,
        parents,
    )

    localB = oursL.clone()
    oursL = oursL.cpu().numpy()
    oursG = oursG.cpu().numpy()
    quatsB = quatsB.cpu().numpy()
    delta_q = delta_q.cpu().numpy()
    inp_seq_gpu = inp_seq.clone()
    inp_seq = inp_seq.cpu().numpy()
    tgt_height_gpu = tgt_height.clone()
    inp_height_gpu = inp_height.clone()
    tgt_height = tgt_height.cpu().numpy()
    inp_height = inp_height.cpu().numpy()
    tgtskel = tgtskel.cpu().numpy()

    oursL = oursL.reshape([oursL.shape[0], oursL.shape[1], -1])

    """ Un-normalize the output and input sequence """
    # Un-normalize the output
    local_mean_rshp = local_mean.reshape((1, 1, -1))
    local_std_rshp = local_std.reshape((1, 1, -1))
    oursL[:, :, :] = oursL[:, :, :] * local_std_rshp + local_mean_rshp
    # oursG[:, :, :] = oursG[:, :, :] * global_std + global_mean
    oursG[:, :, :] = oursG[:, :, :]

    localA = (
        inp_seq_gpu[:, :, :-4].view(localB.shape)
        * torch.from_numpy(local_std)[None, :].cuda()
        + torch.from_numpy(local_mean)[None, :].cuda()
    )
    localB = (
        localB * torch.from_numpy(local_std)[None, :].cuda()
        + torch.from_numpy(local_mean)[None, :].cuda()
    )

    if not exists(arg.save_path):
        makedirs(arg.save_path)

    ours_total = np.concatenate((oursL, oursG), axis=-1)

    oursG_scale = np.concatenate(
        [inp_seq[:, :, -4:-1] * (tgt_height / inp_height), oursG[:, :, -1:]], axis=-1
    )  # scale root velocity

    """ VIDEO BVH """
    # offset (skel)
    i = 0
    max_steps = tgtskel.shape[1]
    tjoints = np.reshape(
        tgtskel[i] * local_std_rshp + local_mean_rshp, [max_steps, -1, 3]
    )
    bl_tjoints = tjoints.copy()

    # tmp_gt: (120, 67, 3) i.e. total joint positions
    tmp_gt = Animation.positions_global(
        tgtanim
    )  # Given an animation compute the global joint positions at at every frame
    start_rots = get_orient_start(
        tmp_gt,
        tgtjoints[14],  # left shoulder
        tgtjoints[18],  # right shoulder
        tgtjoints[6],  # left upleg
        tgtjoints[10],
    )  # right upleg

    """Exclude angles in exclude_list as they will rotate non-existent children During training."""
    exclude_list = [
        5,
        17,
        21,
        9,
        13,
    ]  # head, left hand, right hand, left toe base, right toe base
    canim_joints = []
    cquat_joints = []
    # print(tgtjoints[i].shape)
    for l in range(len(tgtjoints)):
        if l not in exclude_list:
            canim_joints.append(tgtjoints[l])
            cquat_joints.append(l)

    outputB_bvh = ours_total[i].copy()  # ours local and global gt

    """Follow the same motion direction as the input and zero speeds
                    that are zero in the input."""
    outputB_bvh[:, -4:] = outputB_bvh[:, -4:] * (
        np.sign(inp_seq[i, :, -4:]) * np.sign(ours_total[i, :, -4:])
    )
    outputB_bvh[:, -3][np.abs(inp_seq[i, :, -3]) <= 1e-2] = 0.0

    outputB_bvh[:, :3] = tgtanim.positions[:1, 0, :].copy()
    wjs, rots = put_in_world_bvh(outputB_bvh.copy(), start_rots)
    tjoints[:, 0, :] = wjs[0, :, 0].copy()

    """ Quaternion results """
    cquat = quatsB[i][:, cquat_joints].copy()

    from_name = arg.load_inp_data["inp_skel_path"].split('/')[-2]
    to_name = arg.load_inp_data["tgt_skel_path"].split('/')[-2]
    bvh_name = arg.load_inp_data["inp_bvh_path"].split('/')[-1]

    BVH.save(join(arg.save_path, from_name + '_inp.bvh'), inpanim, inpname, inpftime)

    BVH.save(
        join(arg.save_path, to_name + '_gt_' + bvh_name), tgtanim, tgtname, tgtftime
    )

    bvh_path = join(arg.save_path, from_name + '_to_' + to_name + '_' + bvh_name)

    """ Ours bvh file """
    tgtanim.positions[:, tgtjoints] = tjoints
    tgtanim.offsets[tgtjoints[1:]] = tjoints[0, 1:]
    cquat[:, 0:1, :] = (rots * Quaternions(cquat[:, 0:1, :])).qs
    tgtanim.rotations.qs[:, canim_joints] = cquat

    BVH.save(bvh_path, tgtanim, tgtname, tgtftime)


def main(arg):
    retarget_net = getmodel(arg.weights, arg)
    parents = np.array(
        [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 0, 10, 11, 12, 3, 14, 15, 16, 3, 18, 19, 20]
    )
    inference(retarget_net, parents, arg)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = get_parser()
    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG:', k)
                assert k in key
        parser.set_defaults(**default_arg)
    arg = parser.parse_args()

    main(arg)
