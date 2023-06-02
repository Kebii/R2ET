import re
import sys
import numpy as np
import utils.BVH as BVH
from utils.Animation import Animation
from utils.Quaternions import Quaternions

import pdb

channelmap = {'Xrotation': 'x', 'Yrotation': 'y', 'Zrotation': 'z'}

channelmap_inv = {
    'x': 'Xrotation',
    'y': 'Yrotation',
    'z': 'Zrotation',
}

ordermap = {
    'x': 0,
    'y': 1,
    'z': 2,
}

JOINT_NAME_MIXAMO_24 = [
    'Hips',
    'LeftUpLeg',
    'LeftLeg',
    'LeftFoot',
    'LeftToeBase',
    'LeftToe_End',
    'RightUpLeg',
    'RightLeg',
    'RightFoot',
    'RightToeBase',
    'RightToe_End',
    'Spine',
    'Spine1',
    'Spine2',
    'Neck',
    'Head',
    'LeftShoulder',
    'LeftArm',
    'LeftForeArm',
    'LeftHand',
    'RightShoulder',
    'RightArm',
    'RightForeArm',
    'RightHand',
]  # simplified joint subset of mixamo, 25 connected joints are selected, no new joint added
EE_NAME_MIXAMO_24 = ['LeftToe_End', 'RightToe_End', 'Head', 'LeftHand', 'RightHand']

JOINT_NAME_MIXAMO_22 = [
    "Hips",
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

EE_NAME_MIXAMO_22 = ['LeftToeBase', 'RightToeBase', 'Head', 'LeftHand', 'RightHand']


class BVHData:
    def __init__(self, file_path):

        # load in the static and motion data from indicated bvh file
        self.file_path = file_path
        self.load(file_path)
        self.simplify_pose_structure()
        self.get_simplified_joint_position()

    @staticmethod
    def transform_from_quaternion(quater):
        qw = quater[..., 0]
        qx = quater[..., 1]
        qy = quater[..., 2]
        qz = quater[..., 3]

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

        m = np.empty(quater.shape[:-1] + (3, 3))
        m[..., 0, 0] = 1.0 - (yy + zz)
        m[..., 0, 1] = xy - wz
        m[..., 0, 2] = xz + wy
        m[..., 1, 0] = xy + wz
        m[..., 1, 1] = 1.0 - (xx + zz)
        m[..., 1, 2] = yz - wx
        m[..., 2, 0] = xz - wy
        m[..., 2, 1] = yz + wx
        m[..., 2, 2] = 1.0 - (xx + yy)

        return m

    @property
    def simplified_topology(self):
        # if self._topology is None:
        #     self._topology = self.anim.parents[self.simplified_corps].copy()
        #     for i in range(self._topology.shape[0]):
        #         if i >= 1: self._topology[i] = self.simplify_map[self._topology[i]]      # have bug for mixamo24 joints
        #         # if i >= 1:
        #         #     if self.simplify_map[self._topology[i]]!=-1:
        #         #         self._topology[i] = self.simplify_map[self._topology[i]]
        #         #     else:
        #         #         self._topology[i] = 0
        #     self._topology = tuple(self._topology)
        self._topology = tuple(
            [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 0, 10, 11, 12, 3, 14, 15, 16, 3, 18, 19, 20]
        )
        return self._topology

    def build_simplified_edge_topology(self):
        # get all edges (pa, child, offset)
        edges = []
        joint_num = len(self.simplified_topology)
        for i in range(1, joint_num):
            edges.append((self.simplified_topology[i], i, self.joint_offsets[i]))
        return edges

    def load(self, filename, world=False):
        """
        load the bvh motion data (euler angle) and return the quaternion / axis angle
        """

        f = open(filename, "r")

        i = 0
        active = -1
        end_site = False

        names = []
        orients = Quaternions.id(0)
        offsets = np.array([]).reshape((0, 3))
        parents = np.array([], dtype=int)

        for line in f:

            if "HIERARCHY" in line:
                continue
            if "MOTION" in line:
                continue

            """ Modified line read to handle mixamo data """
            rmatch = re.match(r"ROOT (\w+:?\w+)", line)
            if rmatch:
                names.append(rmatch.group(1))
                offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
                orients.qs = np.append(orients.qs, np.array([[1, 0, 0, 0]]), axis=0)
                parents = np.append(parents, active)
                active = len(parents) - 1
                continue

            if "{" in line:
                continue

            if "}" in line:
                if end_site:
                    end_site = False
                else:
                    active = parents[active]
                continue

            offmatch = re.match(
                r"\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)", line
            )
            if offmatch:
                if not end_site:
                    offsets[active] = np.array([list(map(float, offmatch.groups()))])
                continue

            chanmatch = re.match(r"\s*CHANNELS\s+(\d+)", line)
            if chanmatch:
                channels = int(chanmatch.group(1))
                channelis = 0 if channels == 3 else 3
                channelie = 3 if channels == 3 else 6
                parts = line.split()[2 + channelis : 2 + channelie]
                if any([p not in channelmap for p in parts]):
                    continue
                order = "".join([channelmap[p] for p in parts])
                continue

            """ Modified line read to handle mixamo data """
            #        jmatch = re.match("\s*JOINT\s+(\w+)", line)
            jmatch = re.match("\s*JOINT\s+(\w+:?\w+)", line)
            if jmatch:
                names.append(jmatch.group(1))
                offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
                orients.qs = np.append(orients.qs, np.array([[1, 0, 0, 0]]), axis=0)
                parents = np.append(parents, active)
                active = len(parents) - 1
                continue

            if "End Site" in line:
                end_site = True
                continue

            fmatch = re.match("\s*Frames:\s+(\d+)", line)
            if fmatch:
                fnum = int(fmatch.group(1))
                jnum = len(parents)
                positions = offsets[np.newaxis].repeat(fnum, axis=0)
                rot_euler = np.zeros((fnum, len(orients), 3))
                continue

            fmatch = re.match("\s*Frame Time:\s+([\d\.]+)", line)
            if fmatch:
                frametime = float(fmatch.group(1))
                continue

            # dmatch = line.strip().split(' ')
            dmatch = line.strip().split()
            if dmatch:
                data_block = np.array(list(map(float, dmatch)))
                N = len(parents)
                fi = i
                if channels == 3:
                    positions[fi, 0:1] = data_block[0:3]
                    rot_euler[fi, :] = data_block[3:].reshape(N, 3)
                elif channels == 6:
                    data_block = data_block.reshape(N, 6)
                    positions[fi, :] = data_block[:, 0:3]
                    rot_euler[fi, :] = data_block[:, 3:6]
                elif channels == 9:
                    positions[fi, 0] = data_block[0:3]
                    data_block = data_block[3:].reshape(N - 1, 9)
                    rot_euler[fi, 1:] = data_block[:, 3:6]
                    positions[fi, 1:] += data_block[:, 0:3] * data_block[:, 6:9]
                else:
                    raise Exception("Too many channels! %i" % channels)

                i += 1

        f.close()

        # delete ":" from joint name
        for i, name in enumerate(names):
            if ':' in name:
                name = name[name.find(':') + 1 :]
                names[i] = name

        # joint information
        self.frame_num = fnum
        self.joint_names = names
        self.joint_parents = parents
        self.joint_offsets = offsets
        self.root_position = positions[:, 0, :]
        self.rotation_order = order
        self.quaternions = Quaternions.from_euler(
            np.radians(rot_euler), order=order, world=world
        )

        self.rot_euler = rot_euler
        self.rot_quat = self.quaternions.qs
        self.rot_angle, self.rot_axis = self.quaternions.angle_axis()

        self.rot_axis_angle = (
            np.tile(np.expand_dims(self.rot_angle, axis=2), (1, 1, 3)) * self.rot_axis
        )
        self.rot_axis_angle.reshape((self.rot_axis_angle.shape[0], -1))

        # sequence information
        self.frametime = frametime
        self.anim = Animation(rot_euler, positions, orients, offsets, parents)

    def simplify_pose_structure(self):

        self.simplified_edges = []
        self._topology = None
        self.ee_length = []  # ee for end-effector

        self.details = [
            i
            for i, name in enumerate(self.joint_names)
            if name not in JOINT_NAME_MIXAMO_22
        ]
        self.joint_num = len(self.joint_names)
        self.simplified_corps = []
        self.simplified_joint_names = []
        self.simplify_map = {}
        self.inverse_simplify_map = {}

        for name in JOINT_NAME_MIXAMO_22:
            for j in range(self.joint_num):
                if name == self.joint_names[j]:
                    self.simplified_corps.append(j)
                    break

        if len(self.simplified_corps) != len(JOINT_NAME_MIXAMO_22):
            for i in self.simplified_corps:
                print(self.joint_names[i], end=' ')
            print(self.simplified_corps, len(self.simplified_corps), sep='\n')
            raise Exception('Problem in file', self.file_path)

        self.ee_id = []
        for i in EE_NAME_MIXAMO_22:
            self.ee_id.append(JOINT_NAME_MIXAMO_22.index(i))

        self.simplified_joint_num = len(self.simplified_corps)
        for i, j in enumerate(self.simplified_corps):
            self.simplify_map[j] = i
            self.inverse_simplify_map[i] = j
            self.simplified_joint_names.append(self.joint_names[j])

        self.inverse_simplify_map[0] = -1
        for i in range(self.joint_num):
            if i in self.details:
                self.simplify_map[i] = -1

        self.simplified_edges = self.build_simplified_edge_topology()
        self.simplified_axis_angle = self.rot_axis_angle[:, self.simplified_corps, :]
        self.simplified_quaternion = self.rot_quat[:, self.simplified_corps, :]
        self.simplified_joint_offsets = self.joint_offsets[self.simplified_corps, :]
        # self.simplified_joint_positions = self.root_position[:, self.simplified_corps, :]

    def get_simplified_joint_position(self):
        """
        obtain joint world positions by applying forward kinematics
        """
        # input arguments
        rotation = self.simplified_quaternion
        offsets = self.simplified_joint_offsets
        root_position = self.root_position

        # pre-processing
        norm = np.repeat(
            np.linalg.norm(rotation, axis=2)[:, :, np.newaxis],
            rotation.shape[-1],
            axis=2,
        )
        rotation = self.simplified_quaternion / norm
        offsets = offsets.reshape((-1, 1, offsets.shape[-2], offsets.shape[-1], 1))
        transform = self.transform_from_quaternion(rotation)

        joint_positions = np.empty(self.simplified_quaternion.shape[:-1] + (3,))
        joint_positions[:, 0, :] = root_position

        for i, pi in enumerate(self._topology):
            if pi == -1:
                assert i == 0
                continue

            joint_positions[..., i, :] = np.matmul(
                transform[..., pi, :, :], offsets[..., i, :, :]
            ).squeeze()
            transform[..., i, :, :] = np.matmul(
                transform[..., pi, :, :].copy(), transform[..., i, :, :].copy()
            )
            joint_positions[..., i, :] += joint_positions[..., pi, :]

        self.simplified_joint_position = joint_positions
