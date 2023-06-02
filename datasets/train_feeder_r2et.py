import torch
import numpy as np
from torch.utils.data import Dataset
from os import listdir, makedirs
from os.path import exists, join


class Feeder(Dataset):
    def __init__(self, data_path, stats_path, shape_path, max_length):
        self.data_path = data_path
        self.stats_path = stats_path
        self.max_length = max_length
        self.parents = np.array(
            [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 0, 10, 11, 12, 3, 14, 15, 16, 3, 18, 19, 20]
        )

        self.leftarm_bone_lst = np.array([15, 16, 17])
        self.rightarm_bone_lst = np.array([19, 20, 21])
        self.leftleg_bone_lst = np.array([6, 7, 8, 9])
        self.rightleg_bone_lst = np.array([10, 11, 12, 13])
        self.body_bone_lst = np.array([0, 1, 2, 3, 5])

        # ----------------- load shape files -------------------------------------------------------------
        self.shape_dic = {}
        shape_lst = []
        file_names = listdir(shape_path)
        for shape_name in file_names:
            fbx_file = np.load(join(shape_path, shape_name))
            full_width = fbx_file['full_width'].astype(np.single)
            joint_shape = fbx_file['joint_shape'].astype(np.single)

            shape_vecotr = np.divide(joint_shape, full_width[None, :])
            self.shape_dic[shape_name.split('.')[0]] = shape_vecotr

            shape_lst.append(shape_vecotr[:, :])

        shape_array = np.concatenate(shape_lst, axis=0)
        self.shape_mean = shape_array.mean(axis=0)
        self.shape_std = shape_array.std(axis=0)

        # -----------------------------------------------------------------------------------------------
        self.load_data()

    def load_data(self):
        all_local = []
        all_global = []
        all_skel = []
        all_names = []
        t_skel = []
        all_quats = []
        seq_names = []

        folders = [
            f
            for f in listdir(self.data_path)
            if not f.startswith(".") and not f.endswith("py") and not f.endswith(".npz")
        ]
        for folder_name in folders:
            files = [
                f
                for f in listdir(join(self.data_path, folder_name))
                if not f.startswith(".") and f.endswith("_seq.npy")
            ]
            for cfile in files:
                file_name = cfile[:-8]
                # Real joint positions
                positions = np.load(
                    join(self.data_path, folder_name, file_name + "_skel.npy")
                )

                # After processed (Maybe, last 4 elements are dummy values)
                sequence = np.load(
                    join(self.data_path, folder_name, file_name + "_seq.npy")
                )

                # Processed global positions (#frames, 4)
                offset = sequence[:, -8:-4]

                # Processed local positions (#frames, #joints, 3)
                sequence = np.reshape(sequence[:, :-8], [sequence.shape[0], -1, 3])
                positions[:, 0, :] = sequence[:, 0, :]  # root joint

                all_local.append(sequence)
                all_global.append(offset)
                all_skel.append(positions)
                all_names.append(folder_name)
                seq_names.append(file_name)

                # ground truth quat (#frames, #joints, 4)
                quat = np.load(
                    join(self.data_path, folder_name, file_name + "_quat.npy")
                )
                all_quats.append(quat)

        # Joint positions before processed
        train_skel = all_skel  # N T J 3

        # After processed, relative position
        train_local = all_local  # N T J 3
        train_global = all_global  # N T 4

        # T-pose (real position)
        for tt in train_skel:
            t_skel.append(tt[0:1])

        # Total training samples
        all_frames = np.concatenate(train_local)
        ntotal_samples = all_frames.shape[0]
        ntotal_sequences = len(train_local)
        print("Number of sequences: " + str(ntotal_sequences))

        # ============================= Data Normalize ============================= #
        '''Calculate total mean and std'''
        allframes_n_skel = np.concatenate(train_local + t_skel)
        local_mean = allframes_n_skel.mean(axis=0)[None, :]
        global_mean = np.concatenate(train_global).mean(axis=0)[None, :]
        local_std = allframes_n_skel.std(axis=0)[None, :]
        global_std = np.concatenate(train_global).std(axis=0)[None, :]

        allframes_quat = np.concatenate(all_quats)
        quat_mean = allframes_quat.mean(axis=0)[None, :]  # 1 J 4
        quat_std = allframes_quat.std(axis=0)[None, :]

        '''Save the data stats'''
        if not exists(self.stats_path):
            makedirs(self.stats_path)
        np.save(join(self.stats_path, "mixamo_local_motion_mean.npy"), local_mean)
        np.save(join(self.stats_path, "mixamo_local_motion_std.npy"), local_std)
        np.save(join(self.stats_path, "mixamo_global_motion_mean.npy"), global_mean)
        np.save(join(self.stats_path, "mixamo_global_motion_std.npy"), global_std)

        np.save(join(self.stats_path, "mixamo_shape_mean_xyz.npy"), self.shape_mean)
        np.save(join(self.stats_path, "mixamo_shape_std_xyz.npy"), self.shape_std)

        np.save(join(self.stats_path, "mixamo_quat_mean.npy"), quat_mean)
        np.save(join(self.stats_path, "mixamo_quat_std.npy"), quat_std)

        '''Normalize the data'''
        self.num_joint = all_local[0].shape[-2]
        self.T = all_local[0].shape[0]
        local_std[local_std == 0] = 1

        for i in range(len(train_local)):
            train_local[i] = (train_local[i] - local_mean) / local_std
            train_global[i] = train_global[i]
            train_skel[i] = (train_skel[i] - local_mean) / local_std
            all_quats[i] = (all_quats[i] - quat_mean) / quat_std

        self.train_local = train_local
        self.train_global = train_global
        self.train_skel = train_skel
        self.local_mean = local_mean
        self.local_std = local_std
        self.quat_mean = quat_mean
        self.quat_std = quat_std
        self.global_mean = global_mean
        self.global_std = global_std
        self.all_names = all_names
        self.seq_names = seq_names
        self.all_quats = all_quats

    def __len__(self):
        return len(self.train_skel)

    def __iter__(self):
        return self

    def __getitem__(self, indexA):
        local_i = self.train_local[indexA]
        global_i = self.train_global[indexA]
        skel_i = self.train_skel[indexA]
        quat_i = self.all_quats[indexA]

        n_joints = local_i.shape[1]
        max_len = self.max_length

        mask = torch.zeros((max_len,), dtype=torch.float32)
        aeReg = torch.zeros((1,), dtype=torch.float32)
        heightA = torch.zeros((1,), dtype=torch.float32)
        heightB = torch.zeros((1,), dtype=torch.float32)

        inp_armspan_batch = torch.zeros((1,), dtype=torch.float32)
        tgt_armspan_batch = torch.zeros((1,), dtype=torch.float32)

        low = 0
        high = local_i.shape[0] - max_len
        if low >= high:
            stidx = 0
        else:
            stidx = np.random.randint(low=low, high=high)

        # ---------------------------------- Character A ----------------------------------------------------

        clocalA = local_i[stidx : (stidx + max_len)]
        mask[: np.min([max_len, clocalA.shape[0]])] = 1.0

        if clocalA.shape[0] < max_len:
            clocalA = np.concatenate(
                (clocalA, np.zeros((max_len - clocalA.shape[0], n_joints, 3)))
            )

        cglobalA = global_i[stidx : (stidx + max_len)]
        if cglobalA.shape[0] < max_len:
            cglobalA = np.concatenate(
                (cglobalA, np.zeros((max_len - cglobalA.shape[0], n_joints, 3)))
            )

        cskelA = skel_i[stidx : (stidx + max_len)]
        if cskelA.shape[0] < max_len:
            cskelA = np.concatenate(
                (cskelA, np.zeros((max_len - cskelA.shape[0], n_joints, 3)))
            )

        cquatA = quat_i[stidx : (stidx + max_len)]
        if cquatA.shape[0] < max_len:
            zeros = np.zeros((max_len - cquatA.shape[0], n_joints, 4))
            zeros[:, :, 0] = 1.0
            cquatA = np.concatenate((cquatA, zeros))

        # ---------------------------------- Character B ----------------------------------------------------
        indexB = np.random.randint(len(self.train_skel))

        cskelB = self.train_skel[indexB][0:max_len]
        if cskelB.shape[0] < max_len:
            cskelB = np.concatenate(
                (cskelB, np.zeros((max_len - cskelB.shape[0], n_joints, 3)))
            )

        joints_a = cskelA[0].copy()
        joints_a = joints_a[None]
        joints_a = (joints_a * self.local_std) + self.local_mean
        height_a = Feeder.get_height_from_skel(joints_a[0])
        height_a = height_a / 100

        joints_b = cskelB[0].copy()
        joints_b = joints_b[None]
        joints_b = joints_b * self.local_std + self.local_mean
        height_b = Feeder.get_height_from_skel(joints_b[0])
        height_b = height_b / 100

        aeReg_on = np.random.binomial(1, p=0.5)
        if aeReg_on:
            cskelB = cskelA.copy()
            aeReg[0] = 1
            heightA[0] = height_a
            heightB[0] = height_a
            indexB = indexA
        else:
            aeReg[0] = 0
            heightA[0] = height_a
            heightB[0] = height_b

        localA = clocalA.reshape((max_len, -1))
        globalA = cglobalA.reshape((max_len, -1))
        seqA = np.concatenate((localA, globalA), axis=-1).astype(np.float32)
        skelA = cskelA.reshape((max_len, -1)).astype(np.float32)
        quatA = cquatA.astype(np.float32)

        localB = clocalA.reshape((max_len, -1))
        globalB = cglobalA.reshape((max_len, -1))
        seqB = np.concatenate((localB, globalB), axis=-1).astype(np.float32)
        skelB = cskelB.reshape((max_len, -1)).astype(np.float32)

        shapeA = self.shape_dic[self.all_names[indexA]].reshape(-1)
        shapeB = self.shape_dic[self.all_names[indexB]].reshape(-1)

        return (
            indexA,
            indexB,
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
            quatA,
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
