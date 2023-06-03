import os
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
import torch.optim as optim
from os import listdir, makedirs
from os.path import exists, join

from src.ops import get_wjs
from datasets.train_feeder_r2et import Feeder
from src.model_skeleton_aware import RetNet, MotionDis


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
        '--work-dir',
        default='./work_dir/r2et_skeleton_aware',
        help='the work folder for storing results',
    )
    parser.add_argument(
        '--mesh-path', default='./datasets/mixamo_train_mesh', help='the mesh file path'
    )
    parser.add_argument(
        '--model-save-name', default='r2et_skeleton_aware', help='model saved name'
    )
    parser.add_argument(
        '--train-feeder-args',
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
        '--base-lr', type=float, default=0.0001, help='initial learning rate'
    )
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument(
        '--alpha', type=float, default=100.0, help='threshold for euler angle'
    )
    parser.add_argument(
        '--nu', type=float, default=100.0, help='threshold for euler angle'
    )
    parser.add_argument(
        '--mu', type=float, default=10.0, help='weight factor for twist loss'
    )
    parser.add_argument('--euler-ord', default='yzx', help='order of the euler angle')
    parser.add_argument(
        '--max-length', type=int, default=60, help='max sequence length: T'
    )
    parser.add_argument(
        '--num-joint', type=int, default=22, help='number of the joints'
    )
    parser.add_argument(
        '--kp', type=float, default=0.8, help='keep prob in dropout layers'
    )
    parser.add_argument('--margin', type=float, default=0.3, help='fake score margin')
    parser.add_argument(
        '--lam', type=int, default=2, help='balancing factor for GAN loss'
    )
    parser.add_argument(
        '--ret-model-args',
        type=dict,
        default=dict(),
        help='the arguments of retargetor',
    )
    parser.add_argument(
        '--dis-model-args',
        type=dict,
        default=dict(),
        help='the arguments of discriminator',
    )
    parser.add_argument(
        '--weight-decay', type=float, default=0.0005, help='weight decay for optimizer'
    )
    parser.add_argument(
        '--step',
        type=int,
        default=[],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate',
    )
    parser.add_argument('--epoch', type=int, default=30, help='training epoch')

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


def train(
    retarget_net,
    discriminator,
    data_loader,
    optimizer_r,
    optimizer_d,
    scheduler_r,
    scheduler_d,
    global_mean,
    global_std,
    local_mean,
    local_std,
    quat_mean,
    quat_std,
    parents,
    epoch,
    logger,
    arg,
):
    pbar = tqdm(total=len(data_loader), ncols=140)
    epoch_loss_ret = AverageMeter()
    epoch_loss_base = AverageMeter()
    epoch_loss_sem = AverageMeter()
    epoch_loss_disc = AverageMeter()
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
        inp_height,
        tgt_height,
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
        inp_height = inp_height.float().cuda(arg.device[0])
        tgt_height = tgt_height.float().cuda(arg.device[0])
        quatA_cp = quatA_cp.float().cuda(arg.device[0])
        shapeA = shapeA.float().cuda(arg.device[0])
        shapeB = shapeB.float().cuda(arg.device[0])

        pbar.set_description("Train Epoch %i  Step %i" % (epoch + 1, batch_idx))
        start_time = time.time()

        # ----------------------------- train retarget_net --------------------------------#
        retarget_net.train()
        discriminator.eval()
        optimizer_r.zero_grad()

        (
            localA_gt,
            localB_rt,
            localB_gt,
            globalA_gt,
            globalB_rt,
            quatB_rt,
        ) = retarget_net(
            seqA,
            seqB,
            skelA,
            skelB,
            shapeA,
            shapeB,
            quatA_cp,
            inp_height,
            tgt_height,
            local_mean,
            local_std,
            quat_mean,
            quat_std,
            parents,
        )

        # motion disc #
        num_joint = arg.num_joint
        batch_size = localA_gt.shape[0]
        max_len = arg.max_length

        wjsA = get_wjs(localA_gt, globalA_gt)
        wjsA = torch.reshape(wjsA, [batch_size, max_len, num_joint, 3])

        wjsB = get_wjs(localB_rt, globalB_rt)
        wjsB = torch.reshape(wjsB, [batch_size, max_len, num_joint, 3])

        inpxyz = torch.mean(wjsA, dim=2)
        motion_real = torch.divide(inpxyz, inp_height[:, :, None]).float()
        motion_real = motion_real.permute(0, 2, 1).contiguous()
        score_real = discriminator(motion_real)

        tgtxyz = torch.mean(wjsB, dim=2)
        motion_fake = torch.divide(tgtxyz, tgt_height[:, :, None]).float()
        motion_fake = motion_fake.permute(0, 2, 1).contiguous()
        score_fake = discriminator(motion_fake)

        attention_list = [
            7,
            8,
            11,
            12,
            15,
            16,
            19,
            20,
        ]  # L/R knee, foot, arm, forearm  (1.95)
        quatA_cp = (
            quatA_cp * torch.from_numpy(quat_std).cuda(quatA_cp.device)[None, :]
            + torch.from_numpy(quat_mean).cuda(quatA_cp.device)[None, :]
        )
        quatA_cp = quatA_cp.float()

        local_ae_loss, quat_ae_loss = RetNet.get_recon_loss(
            attention_list,
            num_joint,
            aeReg,
            mask,
            localB_rt,
            localB_gt,
            quatA_cp,
            quatB_rt,
        )

        twist_loss = RetNet.get_rot_cons_loss(arg.alpha, arg.euler_ord, quatB_rt)

        gen_loss = RetNet.get_gen_loss(score_fake, aeReg)

        base_loss = local_ae_loss + quat_ae_loss + arg.mu * twist_loss

        local_std_ts = torch.from_numpy(local_std).cuda(skelB.device)
        local_mean_ts = torch.from_numpy(local_mean).cuda(skelB.device)

        bs = quatB_rt.shape[0]
        t_poseB = torch.reshape(skelB[:, 0, :], [bs, num_joint, 3])
        t_poseB = t_poseB * local_std_ts + local_mean_ts
        t_poseB = t_poseB.float()

        t_poseA = torch.reshape(skelA[:, 0, :], [bs, num_joint, 3])
        t_poseA = t_poseA * local_std_ts + local_mean_ts
        t_poseA = t_poseA.float()

        localB_rt = (
            localB_rt * local_std_ts[:, None, :, :] + local_mean_ts[:, None, :, :]
        ).float()

        localA_gt = (
            localA_gt * local_std_ts[:, None, :, :] + local_mean_ts[:, None, :, :]
        ).float()

        attention_list = [15, 16, 19, 20]
        normed_matrixB, normed_matrixA = RetNet.get_rela_matrix(
            localB_rt, localA_gt, tgt_height, inp_height
        )
        sem_loss = RetNet.get_sem_loss(
            attention_list, num_joint, normed_matrixA, normed_matrixB, mask
        )

        ret_loss = arg.lam * gen_loss + base_loss + arg.nu * sem_loss

        ret_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(retarget_net.parameters(), max_norm=25)
        optimizer_r.step()

        # ------------------------------------------- train discriminator -----------------------------------------------
        retarget_net.eval()
        discriminator.train()
        optimizer_d.zero_grad()

        score_real = discriminator(motion_real.detach())
        score_fake = discriminator(motion_fake.detach())

        disc_loss = RetNet.get_dis_loss(score_real, score_fake, aeReg)

        for i in range(score_fake.shape[0]):
            if score_fake[i] > arg.margin:
                disc_loss.backward()
                nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=25)
                optimizer_d.step()
                break

        end_time = time.time()
        epoch_time.update(end_time - start_time)
        epoch_loss_ret.update(float(ret_loss.item()))
        epoch_loss_disc.update(float(disc_loss.item()))
        epoch_loss_base.update(float(base_loss.item()))
        epoch_loss_sem.update(float(arg.mu * sem_loss.item()))

        pbar.set_postfix(
            loss_r=float(ret_loss.item()),
            loss_d=float(disc_loss.item()),
            loss_base=float(base_loss.item()),
            loss_rela=float(arg.mu * sem_loss.item()),
            time=end_time - start_time,
        )
        pbar.update(1)
    scheduler_r.step()
    scheduler_d.step()
    pbar.close()

    logger.add_scalar('train_loss_ret', epoch_loss_ret.avg, epoch)
    logger.add_scalar('train_loss_disc', epoch_loss_disc.avg, epoch)
    logger.add_scalar('train_loss_base', epoch_loss_base.avg, epoch)
    logger.add_scalar('train_loss_sem', epoch_loss_sem.avg, epoch)

    return epoch_loss_ret, epoch_loss_disc, epoch_loss_base, epoch_loss_sem, epoch_time


def print_log_txt(s, work_dir, print_time=True):
    if print_time:
        localtime = time.asctime(time.localtime(time.time()))
        s = f'[ {localtime} ] {s}'
    print(s)
    with open(os.path.join(work_dir, 'log.txt'), 'a') as f:
        print(s, file=f)


def main(arg):
    data_feeder = Feeder(**arg.train_feeder_args)
    retarget_net = RetNet(**arg.ret_model_args).cuda(arg.device[0])
    discriminator_net = MotionDis(**arg.dis_model_args).cuda(arg.device[0])
    retarget_net = nn.DataParallel(retarget_net, device_ids=arg.device)
    discriminator_net = nn.DataParallel(discriminator_net, device_ids=arg.device)

    data_loader = torch.utils.data.DataLoader(
        dataset=data_feeder, batch_size=arg.batch_size, num_workers=8, shuffle=True
    )
    optimizer_ret = optim.Adam(
        retarget_net.parameters(),
        lr=arg.base_lr,
        weight_decay=arg.weight_decay,
        betas=(0.5, 0.999),
    )
    optimizer_dis = optim.Adam(
        discriminator_net.parameters(),
        lr=arg.base_lr,
        weight_decay=arg.weight_decay,
        betas=(0.5, 0.999),
    )

    scheduler_ret = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_ret, milestones=arg.step, gamma=0.1, last_epoch=-1
    )
    scheduler_dis = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_dis, milestones=arg.step, gamma=0.1, last_epoch=-1
    )

    train_writer = SummaryWriter(
        os.path.join(arg.work_dir, arg.model_save_name, 'train_log'), 'train'
    )

    # -----------------------------------------------------------------------------------------------

    # save cfg file
    arg_dict = vars(arg)
    if not exists(arg.work_dir):
        makedirs(arg.work_dir)
    with open(join(arg.work_dir, 'config.yaml'), 'w') as f:
        yaml.dump(arg_dict, f)

    for i in range(arg.epoch):
        (
            epoch_loss_r,
            epoch_loss_d,
            epoch_loss_base,
            epoch_loss_rela,
            epoch_time,
        ) = train(
            retarget_net,
            discriminator_net,
            data_loader,
            optimizer_ret,
            optimizer_dis,
            scheduler_ret,
            scheduler_dis,
            data_feeder.global_mean,
            data_feeder.global_std,
            data_feeder.local_mean,
            data_feeder.local_std,
            data_feeder.quat_mean,
            data_feeder.quat_std,
            data_feeder.parents,
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
            + "  dis loss:"
            + str(epoch_loss_d.avg)
            + "  base loss:"
            + str(epoch_loss_base.avg)
            + "  rela loss:"
            + str(epoch_loss_rela.avg)
            + "  epoch time:"
            + str(epoch_time.avg)
            + "  lr:"
            + str(lr)
        )
        print_log_txt(log_txt, arg.work_dir)

        if (i + 1) % 2 == 0:
            state_dict_ret = retarget_net.state_dict()
            state_dict_dis = discriminator_net.state_dict()

            weights_gen = OrderedDict([[k, v.cpu()] for k, v in state_dict_ret.items()])
            torch.save(
                weights_gen,
                os.path.join(
                    arg.work_dir, arg.model_save_name + '_ret-' + str(i + 1) + '.pt'
                ),
            )
            log_txt = arg.model_save_name + '_ret-' + str(i + 1) + '.pt has been saved!'
            print_log_txt(log_txt, arg.work_dir)

            weights_dis = OrderedDict([[k, v.cpu()] for k, v in state_dict_dis.items()])
            torch.save(
                weights_dis,
                os.path.join(
                    arg.work_dir, arg.model_save_name + '_dis-' + str(i + 1) + '.pt'
                ),
            )
            log_txt = arg.model_save_name + '_dis-' + str(i + 1) + '.pt has been saved!'
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
