import sys
import os
sys.path.append('./')
sys.path.append("../utils")
from utils import BVH
import numpy as np
import bpy
import mathutils
import pdb
import math
# scale factor for bone length
global_scale = 1.5

class BVH_file:
    def __init__(self, file_path, x_bias=0, y_bias=0, z_bias=0, custom_scale=None, fix_hip=False, format_type=None):
        self.anim, self.names, self.frametime = BVH.load(file_path)
        self.x_bias = x_bias
        self.y_bias = y_bias
        self.z_bias = z_bias

        self.scale = custom_scale if custom_scale else global_scale
        self.format_type = format_type


        self.joint_num = self.anim.rotations.shape[1]
        self.frame_num = self.anim.rotations.shape[0]
        self.normalize(dir=1)

        tmp = self.anim.offsets.copy()
        self.anim.offsets[..., 0] = tmp[..., 2]
        self.anim.offsets[..., 1] = tmp[..., 0]
        self.anim.offsets[..., 2] = tmp[..., 1]

        tmp = self.anim.positions.copy()
        self.anim.positions[..., 0] = tmp[..., 2] # after assignment (x)
        self.anim.positions[..., 1] = tmp[..., 0] # after assignment (y)
        self.anim.positions[..., 2] = tmp[..., 1] # after assignment (z)

        tmp = self.anim.rotations.qs.copy()
        self.anim.rotations.qs[..., 1] = tmp[..., 3]
        self.anim.rotations.qs[..., 2] = tmp[..., 1]
        self.anim.rotations.qs[..., 3] = tmp[..., 2]

        if fix_hip:
            self.anim.positions[..., 0] = 0 # x
            self.anim.positions[..., 1] = 0 # y

        self.shift()


        # pdb.set_trace()

    @property
    def topology(self):
        return self.anim.parents

    @property
    def offsets(self):
        return self.anim.offsets

    # Normalize bone length by height and translate the (x, y) mean to (0, 0)
    def normalize(self,dir=-1):

        height = self.get_height(dir=dir) / self.scale
        self.anim.offsets /= height
        self.anim.positions /= height
        mean_position = np.mean(self.anim.positions[:, 0, :], axis=0)

        self.anim.positions[..., 0] -= self.anim.positions[0, 0, 0] # y
        self.anim.positions[..., 2] -= self.anim.positions[0, 0, 2] # x



    def shift(self, dir=-1):
        self.anim.positions[..., 0] += self.x_bias
        self.anim.positions[..., 1] += self.y_bias

        if str(self.z_bias).isdigit():
            self.anim.positions[..., 2] += self.z_bias
        else:
            self.anim.positions[..., 2] -= self.anim.positions[0, 0, 2] # z, locate hips to the horizon

            traverse_idx = self.names.index('Head') # head
            head_offset = self.anim.offsets[traverse_idx, 2]
            while self.topology[traverse_idx] != -1:
                head_offset += self.anim.offsets[self.topology[traverse_idx], 2]
                traverse_idx = self.topology[traverse_idx]
            self.anim.positions[..., 2] += global_scale-head_offset


    def get_height(self, dir=-1):
        low = high = 0

        def dfs(i, pos):
            nonlocal low
            nonlocal high
            low = min(low, pos[dir])
            high = max(high, pos[dir])

            for j in range(self.joint_num):
                if self.topology[j] == i:
                    dfs(j, pos + self.offsets[j])

        dfs(0, np.array([0, 0, 0]))
        return high - low


def add_bone(offset, parent_obj, name):
    center = parent_obj.location + offset / 2
    length = offset.dot(offset) ** 0.5

    base = mathutils.Vector((0., 0., 1.))
    target = offset.normalized()
    axis = base.cross(target)
    theta = np.math.acos(base.dot(target))
    rot = mathutils.Quaternion(axis, theta)

    bpy.ops.mesh.primitive_cone_add(vertices=5, radius1=0.022 * global_scale, radius2=0.0132 * global_scale, depth=length, enter_editmode=False, location=center)
    new_bone = bpy.context.object
    new_bone.name = name
    new_bone.rotation_mode = 'QUATERNION'
    new_bone.rotation_quaternion = rot

    set_parent(parent_obj, new_bone)

    return new_bone

def add_joint(location, parent_obj, name):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=0.001, enter_editmode=False, location=location)
    new_joint = bpy.context.object
    if parent_obj is not None:
        set_parent(parent_obj, new_joint)
        pass
    new_joint.name = name

    return new_joint

def build_t_pose(file: BVH_file, joint, parent_obj, all_obj):
    if joint != 0:
        offset = mathutils.Vector(file.offsets[joint])
        new_bone = add_bone(offset, parent_obj, file.names[joint] + '_bone')
        new_joint = add_joint(parent_obj.location + offset, new_bone, file.names[joint] + '_end')
        all_obj.append(new_bone)
        all_obj.append(new_joint)
    else:
        new_joint = add_joint(mathutils.Vector((0., 0., 0.)), None, file.names[joint] + '_end')
        all_obj.append(new_joint)

    for i in range(len(file.topology)):
        if file.topology[i] == joint:
            build_t_pose(file, i, new_joint, all_obj)


def set_parent(parent, child):
    child.parent = parent
    child.matrix_parent_inverse = parent.matrix_world.inverted()
    '''
        See https://blender.stackexchange.com/questions/9200/how-to-make-object-a-a-parent-of-object-b-via-blenders-python-api
    '''


def set_animation(file, joints):
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = file.anim.rotations.shape[0] - 1

    bpy.context.scene.render.fps = 1 / file.frametime

    bpy.ops.object.select_all(action='DESELECT')

    print('Set fps to', bpy.context.scene.render.fps)
    print(file.frame_num, 'frames in total')

    for frame in range(0, file.frame_num):
        joints[0].location = file.anim.positions[frame, 0, :]
        joints[0].keyframe_insert(data_path='location', frame=frame)
        if frame % 100 == 99:
            print('[{}/{}] done.'.format(frame+1, file.frame_num))
        for j in range(file.joint_num):
            joints[j].rotation_mode = 'QUATERNION'
            joints[j].rotation_quaternion = mathutils.Quaternion(file.anim.rotations.qs[frame, j, :])
            joints[j].keyframe_insert(data_path='rotation_quaternion', frame=frame)

    bpy.context.scene.frame_current = 0


def load_bvh(file_name, x_bias=0, y_bias=0, z_bias=0, custom_scale=None, fix_hip=False, collecttion_name='Character', format_type='mixamo'):
    print('Loading BVH file......')
    file = BVH_file(file_name, x_bias, y_bias, z_bias, custom_scale, fix_hip, format_type=format_type)
    print('Loading BVH file done.')

    print('Building T-Pose......')
    all_obj = []
    build_t_pose(file, 0, None, all_obj)
    print('Building T-Pose done.')

    print('Loading keyframes......')

    #pairing object order and file.animation's order
    all_joints = []
    for j in range(file.joint_num):
        name = file.names[j]
        for obj in all_obj:
            # if obj.name == name + '_end':
            if name + '_end' in obj.name:
                all_joints.append(obj)
                break

    set_animation(file, all_joints)
    print('Loading keyframes done.')

    bpy.ops.object.select_all(action='DESELECT')
    for obj in all_obj:
        obj.select_set(True)
    bpy.ops.object.move_to_collection(collection_index=0, is_new=True, new_collection_name=collecttion_name)
    bpy.ops.object.select_all(action='DESELECT')
    print('Load bvh all done!')

    return all_obj


if __name__ == '__main__':
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    file_name = './mocap/result.bvh'
    load_bvh(file_name, 0)

    file_name = './mocap/gt.bvh'
    load_bvh(file_name, 5)
