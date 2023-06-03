import os
import time
import datetime
import random
import yaml
import argparse
import shutil
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import torch.optim as optim
from os import listdir, makedirs
from os.path import exists, join

from src.ops import get_wjs
from datasets.train_feeder_r2et import Feeder
from src.model_shape_aware import RetNet, MotionDis


def get_parser():
    # parameter priority: command line > config file > default
    parser = argparse.ArgumentParser(description='R2ET for motion retargeting')
    parser.add_argument(
        '--config',
        default='./config/train_cfg.yaml',
        help='path to the configuration file',
    )
    parser.add_argument('--phase', default='train', help='train or test')
    parser.add_argument(
        '--work_dir',
        default='./work_dir/r2et_shape_aware',
        help='the work folder for storing results',
    )
    parser.add_argument(
        '--mesh_path', default='./datasets/mixamo_train_mesh', help='the mesh file path'
    )
    parser.add_argument(
        '--model_save_name', default='r2et_shape_aware', help='model saved name'
    )
    parser.add_argument(
        '--train_feeder_args',
        default=dict(),
        help='the arguments of data loader for training',
    )
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing',
    )
    parser.add_argument(
        '--base_lr', type=float, default=0.0001, help='initial learning rate'
    )
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument(
        '--alpha', type=float, default=100.0, help='threshold for euler angle'
    )
    parser.add_argument(
        '--mu', type=float, default=100.0, help='weight factor for sem loss'
    )
    parser.add_argument(
        '--nu', type=float, default=10.0, help='weight factor for twist loss'
    )
    parser.add_argument(
        '--kappa',
        type=float,
        default=1.0,
        help='weight factor for Repulsive and Attractive loss',
    )
    parser.add_argument(
        '--tao', type=float, default=1.0, help='weight factor for regularzation'
    )
    parser.add_argument('--euler_ord', default='yzx', help='order of the euler angle')
    parser.add_argument(
        '--max_length', type=int, default=60, help='max sequence length: T'
    )
    parser.add_argument(
        '--num_joint', type=int, default=22, help='number of the joints'
    )
    parser.add_argument(
        '--kp', type=float, default=0.8, help='keep prob in dropout layers'
    )
    parser.add_argument('--margin', type=float, default=0.3, help='fake score margin')
    parser.add_argument('--lam', type=int, default=2, help='balance the GAN loss')
    parser.add_argument(
        '--ret_model_args',
        type=dict,
        default=dict(),
        help='the arguments of retargetor',
    )
    parser.add_argument(
        '--dis_model_args',
        type=dict,
        default=dict(),
        help='the arguments of discriminator',
    )
    parser.add_argument(
        '--weight_decay', type=float, default=0.0005, help='weight decay for optimizer'
    )
    parser.add_argument(
        '--step',
        type=int,
        default=[],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate',
    )
    parser.add_argument(
        '--epoch', type=int, default=100, help='stop training in which epoch'
    )
    parser.add_argument(
        '--ret_weights',
        default='./work_dir/pmnet.pt',
        help='the path of the weights file for ret net',
    )
    parser.add_argument(
        '--dis_weights',
        default='./work_dir/pmnet.pt',
        help='the path of the weights file for dis net',
    )
    parser.add_argument(
        '--ignore_weights', default=[], help='the ret weights that should be ignored'
    )

    return parser


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def print_log_txt(s, work_dir, print_time=True):
    if print_time:
        localtime = time.asctime(time.localtime(time.time()))
        s = f'[ {localtime} ] {s}'
    print(s)
    with open(os.path.join(work_dir, 'log.txt'), 'a') as f:
        print(s, file=f)


def load_model(ret_model, dis_model, arg):
    output_device = arg.device[0]

    if arg.ret_weights:
        print_log_txt(f'Loading ret weights from {arg.ret_weights}', arg.work_dir)
        ret_weights = torch.load(arg.ret_weights)

        ret_weights = OrderedDict(
            [
                [k.split('module.')[-1], v.cuda(output_device)]
                for k, v in ret_weights.items()
            ]
        )

        pop_lst = []
        for k in ret_weights.keys():
            for w in arg.ignore_weights:
                if w in k:
                    pop_lst.append(k)
        for k in pop_lst:
            ret_weights.pop(k)
            print_log_txt(f'Sucessfully Remove Weights: {k}', arg.work_dir)

        try:
            ret_model.load_state_dict(ret_weights)
        except:
            state = ret_model.state_dict()
            diff = list(set(state.keys()).difference(set(ret_weights.keys())))
            print_log_txt('Can not find these weights:', arg.work_dir, False)
            for d in diff:
                print_log_txt('  ' + d, arg.work_dir, False)
            state.update(ret_weights)
            ret_model.load_state_dict(state)

    if arg.dis_weights:
        print_log_txt(f'Loading dis weights from {arg.dis_weights}', arg.work_dir)
        dis_weights = torch.load(arg.dis_weights)

        dis_weights = OrderedDict(
            [
                [k.split('module.')[-1], v.cuda(output_device)]
                for k, v in dis_weights.items()
            ]
        )

        dis_model.load_state_dict(dis_weights)


def train(
    retarget_net,
    discriminator,
    data_loader,
    optimizer_ret,
    scheduler,
    global_mean,
    global_std,
    local_mean,
    local_std,
    quat_mean,
    quat_std,
    parents,
    mesh_file_dic,
    all_names,
    body_vertices_dic,
    head_vertices_dic,
    leftarm_vertices_dic,
    rightarm_vertices_dic,
    leftleg_vertices_dic,
    rightleg_vertices_dic,
    hands_vertices_dic,
    epoch,
    logger,
    arg,
):
    pbar = tqdm(total=len(data_loader), ncols=140)
    epoch_loss_ret = AverageMeter()
    epoch_loss_rep = AverageMeter()
    epoch_time = AverageMeter()

    global_mean = torch.from_numpy(global_mean).cuda(arg.device[0])
    global_std = torch.from_numpy(global_std).cuda(arg.device[0])

    for batch_idx, (
        indexesA,
        indexesB,
        seqA,
        skelA,
        seqB,
        skelB,
        aeReg,
        mask,
        heightA,
        heightB,
        shapeA,
        shapeB,
        quatA_cp,
    ) in enumerate(data_loader):
        seqA = seqA.float().cuda(arg.device[0])
        skelA = skelA.float().cuda(arg.device[0])
        seqB = seqB.float().cuda(arg.device[0])
        skelB = skelB.float().cuda(arg.device[0])
        aeReg = aeReg.float().cuda(arg.device[0])
        mask = mask.float().cuda(arg.device[0])
        heightA = heightA.float().cuda(arg.device[0])
        heightB = heightB.float().cuda(arg.device[0])
        quatA_cp = quatA_cp.float().cuda(arg.device[0])
        shapeA = shapeA.float().cuda(arg.device[0])
        shapeB = shapeB.float().cuda(arg.device[0])

        pbar.set_description("Train Epoch %i  Step %i" % (epoch + 1, batch_idx))
        start_time = time.time()

        # ------------------------------------ train generator ------------------------------------#
        retarget_net.train()
        discriminator.eval()
        optimizer_ret.zero_grad()

        (
            localA_gt,
            localB_rt,
            localB_gt,
            globalA_gt,
            globalB_rt,
            quatB_rt,
            quatB_base,
            localB_base,
            weights_sp,
        ) = retarget_net(
            seqA,
            seqB,
            skelA,
            skelB,
            shapeA,
            shapeB,
            quatA_cp,
            heightA,
            heightB,
            local_mean,
            local_std,
            quat_mean,
            quat_std,
            parents,
        )

        # ----------------------------- motion disc -------------------------------- #
        num_joint = arg.num_joint
        batch_size = localA_gt.shape[0]
        max_len = arg.max_length

        wjsB = get_wjs(localB_rt, globalB_rt)
        wjsB = torch.reshape(wjsB, [batch_size, max_len, num_joint, 3])
        tgtxyz = torch.mean(wjsB, dim=2)
        motion_fake = torch.divide(tgtxyz, heightB[:, :, None]).float()
        motion_fake = motion_fake.permute(0, 2, 1).contiguous()
        score_fake = discriminator(motion_fake)

        attention_list = [7, 8, 11, 12, 15, 16, 19, 20]
        local_ae_loss, quat_ae_loss = RetNet.get_cons_loss(
            attention_list,
            num_joint,
            mask,
            localB_rt,
            localB_base,
            quatB_rt,
            quatB_base,
        )
        twist_loss = RetNet.get_rot_cons_loss(arg.alpha, arg.euler_ord, quatB_rt)

        gen_loss = RetNet.get_gen_loss(score_fake, aeReg)

        regular_loss = RetNet.get_regularization_loss(weights_sp, mask)

        base_loss = (
            local_ae_loss + quat_ae_loss + arg.mu * twist_loss + arg.tao * regular_loss
        )

        # -------------------------- geometry ---------------------------------------
        bs = quatB_rt.shape[0]
        t_poseB = torch.reshape(skelB[:, 0, :], [bs, num_joint, 3])
        t_poseB = t_poseB * torch.from_numpy(local_std).cuda(
            t_poseB.device
        ) + torch.from_numpy(local_mean).cuda(t_poseB.device)
        t_poseB = t_poseB.float()
        rep_loss_la, rep_loss_ra, rep_loss_ll, rep_loss_rl = 0, 0, 0, 0
        att_loss = 0

        for i in range(bs):
            fbx_data = mesh_file_dic[all_names[indexesB[i]]]
            vertices_np = fbx_data['rest_vertices']
            sk_weights_np = fbx_data['skinning_weights']

            body_vertices_lst = body_vertices_dic[all_names[indexesB[i]]]
            head_vertices_lst = head_vertices_dic[all_names[indexesB[i]]]
            leftarm_vertices_lst = leftarm_vertices_dic[all_names[indexesB[i]]]
            rightarm_vertices_lst = rightarm_vertices_dic[all_names[indexesB[i]]]
            leftleg_vertices_lst = leftleg_vertices_dic[all_names[indexesB[i]]]
            rightleg_vertices_lst = rightleg_vertices_dic[all_names[indexesB[i]]]
            hands_vertices_lst = hands_vertices_dic[all_names[indexesB[i]]]

            vertices = (
                torch.from_numpy(vertices_np).cuda(quatB_rt.device).type(torch.float)
            )
            sk_weights = (
                torch.from_numpy(sk_weights_np).cuda(quatB_rt.device).type(torch.float)
            )

            la, ra, ll, rl = RetNet.get_rep_loss_part(
                parents,
                quatB_rt[i],
                t_poseB[i],
                vertices,
                sk_weights,
                body_vertices_lst,
                head_vertices_lst,
                leftarm_vertices_lst,
                rightarm_vertices_lst,
                leftleg_vertices_lst,
                rightleg_vertices_lst,
                True,
            )
            att_loss += RetNet.get_att_loss(
                parents,
                quatB_rt[i],
                t_poseB[i],
                vertices,
                sk_weights,
                body_vertices_lst,
                hands_vertices_lst,
            )
            rep_loss_la += la
            rep_loss_ra += ra
            rep_loss_ll += ll
            rep_loss_rl += rl

        rep_loss_la /= bs
        rep_loss_ra /= bs
        rep_loss_ll /= bs
        rep_loss_rl /= bs
        att_loss /= bs

        # ----------------------------------------- backward ----------------------------------------------
        for para in retarget_net.parameters():
            para.requires_grad = False
        for para in retarget_net.module.delta_leftArm_dec.parameters():
            para.requires_grad = True
        rep_loss_la.backward(retain_graph=True)

        for para in retarget_net.parameters():
            para.requires_grad = False
        for para in retarget_net.module.delta_rightArm_dec.parameters():
            para.requires_grad = True
        rep_loss_ra.backward(retain_graph=True)

        for para in retarget_net.parameters():
            para.requires_grad = False
        for para in retarget_net.module.delta_leftLeg_dec.parameters():
            para.requires_grad = True
        rep_loss_ll.backward(retain_graph=True)

        for para in retarget_net.parameters():
            para.requires_grad = False
        for para in retarget_net.module.delta_rightLeg_dec.parameters():
            para.requires_grad = True
        rep_loss_rl.backward(retain_graph=True)

        rep_loss = arg.kappa * (
            rep_loss_la + rep_loss_ra + rep_loss_ll + rep_loss_rl + att_loss
        )

        ret_loss = arg.lam * gen_loss + base_loss

        for para in retarget_net.parameters():
            para.requires_grad = False
        for para in retarget_net.module.weights_dec.parameters():
            para.requires_grad = True
        rep_loss.backward(retain_graph=True)

        for para in retarget_net.parameters():
            para.requires_grad = True
        ret_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(retarget_net.parameters(), max_norm=25)

        optimizer_ret.step()

        end_time = time.time()
        epoch_time.update(end_time - start_time)
        epoch_loss_ret.update(float(ret_loss.item()))
        epoch_loss_rep.update(float(rep_loss.item()))

        pbar.set_postfix(
            loss_r=float(ret_loss.item()),
            loss_sp=float(rep_loss.item()),
            time=end_time - start_time,
        )
        pbar.update(1)
    scheduler.step()
    pbar.close()

    logger.add_scalar('train_loss_ret', epoch_loss_ret.avg, epoch)
    logger.add_scalar('train_loss_rep', epoch_loss_rep.avg, epoch)

    return epoch_loss_ret, epoch_loss_rep, epoch_time


def main(arg):
    if not exists(arg.work_dir):
        makedirs(arg.work_dir)

    data_feeder = Feeder(**arg.train_feeder_args)
    retarget_net = RetNet(**arg.ret_model_args).cuda(arg.device[0])
    discriminator_net = MotionDis(**arg.dis_model_args).cuda(arg.device[0])

    load_model(retarget_net, discriminator_net, arg)

    retarget_net = nn.DataParallel(retarget_net, device_ids=arg.device)
    discriminator_net = nn.DataParallel(discriminator_net, device_ids=arg.device)

    data_loader = torch.utils.data.DataLoader(
        dataset=data_feeder, batch_size=arg.batch_size, num_workers=8, shuffle=True
    )

    opt_para_lst = [
        retarget_net.module.delta_leftArm_dec.parameters(),
        retarget_net.module.delta_rightArm_dec.parameters(),
        retarget_net.module.delta_leftLeg_dec.parameters(),
        retarget_net.module.delta_rightLeg_dec.parameters(),
        retarget_net.module.weights_dec.parameters(),
    ]
    params = []
    for pl in opt_para_lst:
        for p in pl:
            params.append(p)
    print_log_txt(f'Optimize params: ' + str(len(params)), arg.work_dir)
    optimizer_ret = optim.Adam(
        params, lr=arg.base_lr, weight_decay=arg.weight_decay, betas=(0.5, 0.999)
    )
    scheduler_ret = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_ret, milestones=arg.step, gamma=0.1, last_epoch=-1
    )

    train_writer = SummaryWriter(
        os.path.join(arg.work_dir, arg.model_save_name, 'train_log'), 'train'
    )

    mesh_file_dic = {}
    file_names = listdir(arg.mesh_path)
    for mesh_name in file_names:
        mesh_file_dic[mesh_name.split('.')[0]] = np.load(join(arg.mesh_path, mesh_name))

    # ----------------------------------------- load mesh data ----------------------------------------------
    body_vertices_dic = {}
    head_vertices_dic = {}
    leftarm_vertices_dic = {}
    rightarm_vertices_dic = {}
    leftleg_vertices_dic = {}
    rightleg_vertices_dic = {}
    hands_vertices_dic = {}

    body_bone_lst = data_feeder.body_bone_lst.tolist()[0:4]
    head_bone_lst = data_feeder.body_bone_lst.tolist()[4:]
    leftarm_bone_lst = data_feeder.leftarm_bone_lst.tolist()
    rightarm_bone_lst = data_feeder.rightarm_bone_lst.tolist()
    leftleg_bone_lst = data_feeder.leftleg_bone_lst.tolist()
    rightleg_bone_lst = data_feeder.rightleg_bone_lst.tolist()

    for mesh_name in mesh_file_dic.keys():
        fbx_data = mesh_file_dic[mesh_name]
        vertex_part_np = fbx_data['vertex_part']
        vertex_num = vertex_part_np.shape[0]
        body_lst = []
        head_lst = []
        leftarm_lst = []
        rightarm_lst = []
        leftleg_lst = []
        rightleg_lst = []
        hands_lst = []
        for i in range(vertex_num):
            if vertex_part_np[i] in body_bone_lst:
                body_lst.append(i)
            if vertex_part_np[i] in head_bone_lst:
                head_lst.append(i)
            if vertex_part_np[i] in leftarm_bone_lst:
                leftarm_lst.append(i)
            if vertex_part_np[i] in rightarm_bone_lst:
                rightarm_lst.append(i)
            if vertex_part_np[i] in leftleg_bone_lst:
                leftleg_lst.append(i)
            if vertex_part_np[i] in rightleg_bone_lst:
                rightleg_lst.append(i)
            if (
                vertex_part_np[i] in leftarm_bone_lst[-1:]
                or vertex_part_np[i] in rightarm_bone_lst[-1:]
            ):
                hands_lst.append(i)

        body_vertices_dic[mesh_name] = body_lst
        head_vertices_dic[mesh_name] = head_lst
        leftarm_vertices_dic[mesh_name] = leftarm_lst
        rightarm_vertices_dic[mesh_name] = rightarm_lst
        leftleg_vertices_dic[mesh_name] = leftleg_lst
        rightleg_vertices_dic[mesh_name] = rightleg_lst
        hands_vertices_dic[mesh_name] = hands_lst
    # -----------------------------------------------------------------------------------------------

    # save cfg file
    arg_dict = vars(arg)
    with open(join(arg.work_dir, 'config.yaml'), 'w') as f:
        yaml.dump(arg_dict, f)

    for i in range(arg.epoch):
        epoch_loss_r, epoch_loss_sfpen, epoch_time = train(
            retarget_net,
            discriminator_net,
            data_loader,
            optimizer_ret,
            scheduler_ret,
            data_feeder.global_mean,
            data_feeder.global_std,
            data_feeder.local_mean,
            data_feeder.local_std,
            data_feeder.quat_mean,
            data_feeder.quat_std,
            data_feeder.parents,
            mesh_file_dic,
            data_feeder.all_names,
            body_vertices_dic,
            head_vertices_dic,
            leftarm_vertices_dic,
            rightarm_vertices_dic,
            leftleg_vertices_dic,
            rightleg_vertices_dic,
            hands_vertices_dic,
            i,
            train_writer,
            arg,
        )
        lr = optimizer_ret.param_groups[0]['lr']
        log_txt = (
            'epoch:'
            + str(i + 1)
            + "  ret loss:"
            + str(epoch_loss_r.avg)
            + "  sfpen loss:"
            + str(epoch_loss_sfpen.avg)
            + "  epoch time:"
            + str(epoch_time.avg)
            + "  lr:"
            + str(lr)
        )
        print_log_txt(log_txt, arg.work_dir)

        if (i + 1) % 5 == 0:
            state_dict_ret = retarget_net.state_dict()

            weights_gen = OrderedDict([[k, v.cpu()] for k, v in state_dict_ret.items()])
            torch.save(
                weights_gen,
                os.path.join(
                    arg.work_dir, arg.model_save_name + '_ret-' + str(i + 1) + '.pt'
                ),
            )
            log_txt = arg.model_save_name + '_ret-' + str(i + 1) + '.pt has been saved!'
            print_log_txt(log_txt, arg.work_dir)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    init_seed(3047)
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
