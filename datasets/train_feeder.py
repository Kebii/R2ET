import torch
import numpy as np
from torch.utils.data import Dataset
from os import listdir, makedirs
from os.path import exists, join


class Feeder(Dataset):
    def __init__(self, data_path, stats_path, max_length):
        """
        param data_path:
        stats_path: path for saving stats
        """

        self.data_path = data_path
        self.stats_path = stats_path
        self.max_length = max_length
        self.parents = np.array(
            [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 0, 10, 11, 12, 3, 14, 15, 16, 3, 18, 19, 20]
        )
        self.load_data()

    def load_data(self):
        all_local = []
        all_global = []
        all_skel = []
        all_names = []
        t_skel = []

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

        '''Save the data stats'''
        if not exists(self.stats_path):
            makedirs(self.stats_path)
        np.save(join(self.stats_path, "mixamo_local_motion_mean.npy"), local_mean)
        np.save(join(self.stats_path, "mixamo_local_motion_std.npy"), local_std)
        np.save(join(self.stats_path, "mixamo_global_motion_mean.npy"), global_mean)
        np.save(join(self.stats_path, "mixamo_global_motion_std.npy"), global_std)

        '''Normalize the data'''
        self.num_joint = all_local[0].shape[-2]
        self.T = all_local[0].shape[0]
        local_std[local_std == 0] = 1

        for i in range(len(train_local)):
            train_local[i] = (train_local[i] - local_mean) / local_std
            train_global[i] = (train_global[i] - global_mean) / global_std
            train_skel[i] = (train_skel[i] - local_mean) / local_std

        self.train_local = train_local
        self.train_global = train_global
        self.train_skel = train_skel
        self.local_mean = local_mean
        self.local_std = local_std
        self.global_mean = global_mean
        self.global_std = global_std

    def __len__(self):
        return len(self.train_skel)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        local_i = self.train_local[index]
        global_i = self.train_global[index]
        skel_i = self.train_skel[index]

        n_joints = local_i.shape[1]
        max_len = self.max_length

        mask_batch = torch.zeros((max_len,), dtype=torch.float32)
        aeReg_batch = torch.zeros((1,), dtype=torch.float32)
        inp_height_batch = torch.zeros((1,), dtype=torch.float32)
        tgt_height_batch = torch.zeros((1,), dtype=torch.float32)

        low = 0
        high = local_i.shape[0] - max_len
        if low >= high:
            stidx = 0
        else:
            stidx = np.random.randint(low=low, high=high)

        clocalA = local_i[stidx : (stidx + max_len)]
        mask_batch[: np.min([max_len, clocalA.shape[0]])] = 1.0

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

        rnd_idx = np.random.randint(len(self.train_skel))

        cskelB = self.train_skel[rnd_idx][0:max_len]
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
            aeReg_batch[0] = 1

            inp_height_batch[0] = height_a
            tgt_height_batch[0] = height_a
        else:
            aeReg_batch[0] = 0

            inp_height_batch[0] = height_a
            tgt_height_batch[0] = height_b

        localA = clocalA.reshape((max_len, -1))
        globalA = cglobalA.reshape((max_len, -1))
        seqA = np.concatenate((localA, globalA), axis=-1).astype(np.float32)
        skelA = cskelA.reshape((max_len, -1)).astype(np.float32)

        localB = clocalA.reshape((max_len, -1))
        globalB = cglobalA.reshape((max_len, -1))
        seqB = np.concatenate((localB, globalB), axis=-1).astype(np.float32)
        skelB = cskelB.reshape((max_len, -1)).astype(np.float32)

        return (
            seqA,
            skelA,
            seqB,
            skelB,
            aeReg_batch,
            mask_batch,
            inp_height_batch,
            tgt_height_batch,
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
    feeder = Feeder(
        "/mnt/data1/zjx/code/cvpr2018nkn/datasets/train",
        "/mnt/data1/zjx/code/cvpr2018nkn/datasets/stats",
        60,
    )

    print(len(feeder))
