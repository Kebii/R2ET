import sys

sys.path.append("./outside-code")
import torch
import numpy as np
from torch.utils.data import Dataset
from os import listdir, makedirs
from os.path import exists, join
import BVH as BVH


class Feeder(Dataset):
    def __init__(
        self, data_path, stats_path, shape_path, min_steps, max_steps, is_h36m=False
    ):
        self.data_path = data_path
        self.stats_path = stats_path
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.is_h36m = is_h36m
        self.parents = np.array(
            [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 0, 10, 11, 12, 3, 14, 15, 16, 3, 18, 19, 20]
        )

        self.leftarm_bone_lst = np.array([15, 16, 17])
        self.rightarm_bone_lst = np.array([19, 20, 21])
        self.leftleg_bone_lst = np.array([6, 7, 8, 9])
        self.rightleg_bone_lst = np.array([10, 11, 12, 13])
        self.body_bone_lst = np.array([0, 1, 2, 3, 5])

        self.load_data()

        # ----------------- load shape files -------------------------------------------------------------
        self.shape_dic = {}
        file_names = listdir(shape_path)
        self.shape_mean = np.load(join(stats_path, "mixamo_shape_mean_xyz.npy"))
        self.shape_std = np.load(join(stats_path, "mixamo_shape_std_xyz.npy"))
        for shape_name in file_names:
            fbx_file = np.load(join(shape_path, shape_name))
            full_width = fbx_file['full_width'].astype(np.single)
            joint_shape = fbx_file['joint_shape'].astype(np.single)

            shape_vecotr = np.divide(joint_shape, full_width[None, :])
            self.shape_dic[shape_name.split('.')[0]] = shape_vecotr.reshape(-1)
        # -----------------------------------------------------------------------------------------------

    def load_data(self):
        data_path = self.data_path
        min_steps = self.min_steps
        max_steps = self.max_steps
        is_h36m = self.is_h36m
        stats_path = self.stats_path

        inlocal = []
        inglobal = []
        tgtdata = []
        inpjoints = []
        inpanims = []
        tgtjoints = []
        tgtanims = []
        tgtskels = []
        inpskels = []
        gtanims = []
        from_names = []
        to_names = []
        from_shape_names = []
        to_shape_names = []

        tgtquats = []
        inpquats = []

        km_kc = {
            "known_motion/Kaya/": "known_character/Warrok_W_Kurniawan1/",
            "known_motion/Big_Vegas/": "known_character/Mousey1/",
        }
        km_nc = {
            "known_motion/AJ/": "new_character/Mutant1/",
            "known_motion/Peasant_Man/": "new_character/Ortiz1/",
        }
        nm_kc = {
            "new_motion/Sporty_Granny/": "known_character/Mousey2/",
            "new_motion/Big_Vegas/": "known_character/Warrok_W_Kurniawan2/",
        }
        nm_nc = {
            "new_motion/Big_Vegas/": "new_character/Ortiz2/",
            "new_motion/Castle_Guard/": "new_character/Mutant2/",
        }

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

        test_list = [km_kc, km_nc, nm_kc, nm_nc]

        count = 0
        for test_item in test_list:
            for inp, tgt in test_item.items():
                files = sorted(
                    [
                        f
                        for f in listdir(join(data_path, inp))
                        if not f.startswith(".") and f.endswith("_seq.npy")
                    ]
                )
                for cfile in files:
                    # Put the skels at the same height as the sequence
                    tgtskel = np.load(
                        join(data_path, tgt + "/" + cfile[:-8] + "_skel.npy")
                    )
                    inpskel = np.load(
                        join(data_path, inp + "/" + cfile[:-8] + "_skel.npy")
                    )
                    if tgtskel.shape[0] >= min_steps + 1:
                        if not ("Claire" in inp and "Warrok" in tgt):
                            count += 1
                    inpanim, inpnames, inpftime = BVH.load(
                        join(data_path, inp + "/" + cfile[:-8] + ".bvh")
                    )

                    tgtanim, tgtnames, tgtftime = BVH.load(
                        join(data_path, tgt + "/" + cfile[:-8] + ".bvh")
                    )

                    gtanim = tgtanim.copy()

                    ibvh_file = (
                        open(join(data_path, inp + "/" + cfile[:-8] + ".bvh"))
                        .read()
                        .split("JOINT")
                    )
                    ibvh_joints = [
                        f.split("\n")[0].split(":")[-1].split(" ")[-1]
                        for f in ibvh_file[1:]
                    ]
                    ito_keep = [0]
                    for jname in joints_list:
                        for k in range(len(ibvh_joints)):
                            if jname == ibvh_joints[k][-len(jname) :]:
                                ito_keep.append(k + 1)
                                break

                    tbvh_file = (
                        open(join(data_path, tgt + "/" + cfile[:-8] + ".bvh"))
                        .read()
                        .split("JOINT")
                    )
                    tbvh_joints = [
                        f.split("\n")[0].split(":")[-1].split(" ")[-1]
                        for f in tbvh_file[1:]
                    ]
                    tto_keep = [0]
                    for jname in joints_list:
                        for k in range(len(tbvh_joints)):
                            if jname == tbvh_joints[k][-len(jname) :]:
                                tto_keep.append(k + 1)
                                break

                    tgtanim.rotations.qs[...] = tgtanim.orients.qs[None]
                    if not is_h36m:
                        """Copy joints we don't predict"""
                        cinames = []
                        for jname in inpnames:
                            cinames.append(jname.split(":")[-1])

                        ctnames = []
                        for jname in tgtnames:
                            ctnames.append(jname.split(":")[-1])

                        for jname in cinames:
                            if jname in ctnames:
                                idxt = ctnames.index(jname)
                                idxi = cinames.index(jname)
                                tgtanim.rotations[:, idxt] = inpanim.rotations[
                                    :, idxi
                                ].copy()

                        tgtanim.positions[:, 0] = inpanim.positions[:, 0].copy()

                    inseq = np.load(
                        join(data_path, inp + "/" + cfile[:-8] + "_seq.npy")
                    )

                    if inseq.shape[0] < min_steps:
                        continue

                    outseq = np.load(
                        join(data_path, tgt + "/" + cfile[:-8] + "_seq.npy")
                    )
                    """Subtract lowers point in first timestep for floor contact"""
                    floor_diff = inseq[0, 1:-8:3].min() - outseq[0, 1:-8:3].min()
                    outseq[:, 1:-8:3] += floor_diff
                    tgtskel[:, 0, 1] = outseq[:, 1].copy()

                    offset = inseq[:, -8:-4]
                    inseq = np.reshape(inseq[:, :-8], [inseq.shape[0], -1, 3])
                    num_samples = inseq.shape[0] // max_steps

                    # load quat: (T, 22, 4)
                    tgtquat = np.load(
                        join(data_path, tgt + "/" + cfile[:-8] + "_quat.npy")
                    )
                    inpquat = np.load(
                        join(data_path, inp + "/" + cfile[:-8] + "_quat.npy")
                    )

                    for s in range(num_samples):
                        inpjoints.append(ito_keep)
                        tgtjoints.append(tto_keep)
                        inpanims.append(
                            [
                                inpanim.copy()[s * max_steps : (s + 1) * max_steps],
                                inpnames,
                                inpftime,
                            ]
                        )
                        tgtanims.append(
                            [
                                tgtanim.copy()[s * max_steps : (s + 1) * max_steps],
                                tgtnames,
                                tgtftime,
                            ]
                        )
                        gtanims.append(
                            [
                                gtanim.copy()[s * max_steps : (s + 1) * max_steps],
                                tgtnames,
                                tgtftime,
                            ]
                        )
                        inlocal.append(inseq[s * max_steps : (s + 1) * max_steps])
                        inglobal.append(offset[s * max_steps : (s + 1) * max_steps])
                        tgtdata.append(outseq[s * max_steps : (s + 1) * max_steps, :-4])
                        tgtskels.append(tgtskel[s * max_steps : (s + 1) * max_steps])
                        inpskels.append(inpskel[s * max_steps : (s + 1) * max_steps])
                        from_names.append(inp.split("/")[0] + "_" + inp.split("/")[1])
                        to_names.append(tgt.split("/")[0] + "_" + tgt.split("/")[1])
                        if tgt.split("/")[1][-1] == '1' or tgt.split("/")[1][-1] == '2':
                            to_shape_names.append(tgt.split("/")[1][:-1])
                        else:
                            to_shape_names.append(tgt.split("/")[1])
                        if inp.split("/")[1][-1] == '1' or inp.split("/")[1][-1] == '2':
                            from_shape_names.append(inp.split("/")[1][:-1])
                        else:
                            from_shape_names.append(inp.split("/")[1])

                        tgtquats.append(tgtquat[s * max_steps : (s + 1) * max_steps])
                        inpquats.append(inpquat[s * max_steps : (s + 1) * max_steps])

                    if not inseq.shape[0] % max_steps == 0:
                        inpjoints.append(ito_keep)
                        tgtjoints.append(tto_keep)
                        inpanims.append(
                            [inpanim.copy()[-max_steps:], inpnames, inpftime]
                        )
                        tgtanims.append(
                            [tgtanim.copy()[-max_steps:], tgtnames, tgtftime]
                        )
                        gtanims.append([gtanim.copy()[-max_steps:], tgtnames, tgtftime])
                        inlocal.append(inseq[-max_steps:])
                        inglobal.append(offset[-max_steps:])
                        tgtdata.append(outseq[-max_steps:, :-4])
                        tgtskels.append(tgtskel[-max_steps:])
                        inpskels.append(inpskel[-max_steps:])
                        tgtquats.append(tgtquat[-max_steps:])
                        inpquats.append(inpquat[-max_steps:])
                        from_names.append(inp.split("/")[0] + "_" + inp.split("/")[1])
                        to_names.append(tgt.split("/")[0] + "_" + tgt.split("/")[1])
                        if tgt.split("/")[1][-1] == '1' or tgt.split("/")[1][-1] == '2':
                            to_shape_names.append(tgt.split("/")[1][:-1])
                        else:
                            to_shape_names.append(tgt.split("/")[1])
                        if inp.split("/")[1][-1] == '1' or inp.split("/")[1][-1] == '2':
                            from_shape_names.append(inp.split("/")[1][:-1])
                        else:
                            from_shape_names.append(inp.split("/")[1])

        self.testlocal = inlocal
        self.testglobal = inglobal
        self.testoutseq = tgtdata
        self.inpjoints = inpjoints
        self.inpanims = inpanims
        self.tgtjoints = tgtjoints
        self.tgtanims = tgtanims
        self.testskel = tgtskels
        self.inpskel = inpskels
        self.gtanims = gtanims
        self.from_names = from_names
        self.to_names = to_names
        self.from_shape_names = from_shape_names
        self.to_shape_names = to_shape_names

        self.tgtquats = tgtquats
        self.inpquats = inpquats

        self.local_mean = np.load(join(stats_path, "mixamo_local_motion_mean.npy"))
        self.local_std = np.load(join(stats_path, "mixamo_local_motion_std.npy"))
        self.quat_mean = np.load(join(stats_path, "mixamo_quat_mean.npy"))
        self.quat_std = np.load(join(stats_path, "mixamo_quat_std.npy"))
        self.global_mean = np.load(join(stats_path, "mixamo_global_motion_mean.npy"))
        self.global_std = np.load(join(stats_path, "mixamo_global_motion_std.npy"))
        self.local_std[self.local_std == 0] = 1

        for i in range(len(self.testlocal)):
            self.testlocal[i] = (self.testlocal[i] - self.local_mean) / self.local_std
            # self.testglobal[i] = (self.testglobal[i] - self.global_mean) / self.global_std                              # for pmnet
            self.testskel[i] = (self.testskel[i] - self.local_mean) / self.local_std
            self.inpskel[i] = (self.inpskel[i] - self.local_mean) / self.local_std
            self.inpquats[i] = (self.inpquats[i] - self.quat_mean) / self.quat_std
            self.tgtquats[i] = (self.tgtquats[i] - self.quat_mean) / self.quat_std

    def __len__(self):
        return len(self.testlocal)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        mask = np.zeros((self.max_steps,), dtype="float32")
        heightA = np.zeros((1,), dtype="float32")
        heightB = np.zeros((1,), dtype="float32")

        localA_batch = self.testlocal[index][: self.max_steps].reshape(
            [self.max_steps, -1]
        )
        globalA_batch = self.testglobal[index][: self.max_steps].reshape(
            [self.max_steps, -1]
        )
        seqA = np.concatenate((localA_batch, globalA_batch), axis=-1)
        skelB = self.testskel[index][: self.max_steps].reshape([self.max_steps, -1])
        skelA = self.inpskel[index][: self.max_steps].reshape([self.max_steps, -1])

        step = self.max_steps
        mask[:step] = 1.0

        local_mean = self.local_mean.reshape((1, 1, -1))
        local_std = self.local_std.reshape((1, 1, -1))

        """ Height ratio """
        # Input sequence (un-normalize)
        inp_skel = seqA[0, :-4].copy() * local_std + local_mean
        inp_skel = inp_skel.reshape([22, 3])

        # Ground truth
        gt = self.testoutseq[index][: self.max_steps, :].copy()
        out_skel = gt[0, :-4].reshape([22, 3])

        inp_height = Feeder.get_height(inp_skel) / 100
        out_height = Feeder.get_height(out_skel) / 100

        heightA[0] = inp_height
        heightB[0] = out_height

        inp_shape = self.shape_dic[self.from_shape_names[index]]  # normalized shape
        tgt_shape = self.shape_dic[self.to_shape_names[index]]

        tgtquats = self.tgtquats[index][: self.max_steps]
        inpquats = self.inpquats[index][: self.max_steps]

        return (
            index,
            seqA,
            skelB,
            skelA,
            mask,
            heightA,
            heightB,
            gt,
            self.from_names[index],
            self.to_names[index],
            self.to_shape_names[index],
            np.array(self.tgtjoints[index]),
            np.array(self.inpjoints[index]),
            inp_shape,
            tgt_shape,
            tgtquats,
            inpquats,
        )

    @staticmethod
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

    @staticmethod
    def get_height_from_skel(skel):
        diffs = np.sqrt((skel**2).sum(axis=-1))
        height = diffs[1:6].sum() + diffs[7:10].sum()
        return height


if __name__ == '__main__':
    pass
