import os
import bpy
import bmesh
import numpy as np
import sys

sys.path.append(".")

from bvh import BVHData
from os import listdir, makedirs
from os.path import exists, join
from tqdm import tqdm

import pdb


def rm_prefix(str):
    if ':' in str:
        return str[str.index(':') + 1 :]
    else:
        return str


def get_width(vertices):
    # vertices: (num_joint, 3)
    box = np.zeros((2, 3))
    box[0, :] = vertices.min(axis=0)
    box[1, :] = vertices.max(axis=0)
    width = box[1, :] - box[0, :]
    return width


'''22 joints mixamo'''
body_joint_lst = [0, 1, 2, 3]
right_arm_joint_lst = [14, 15, 16, 17]
left_arm_joint_lst = [18, 19, 20, 21]
arms_joint_lst = right_arm_joint_lst + left_arm_joint_lst


def extract_data(fbx_path, subject_name, save_path):
    bpy.ops.import_scene.fbx(filepath=fbx_path, use_anim=True)
    context = bpy.context
    scene = context.scene
    dg = context.evaluated_depsgraph_get()

    # obtain mesh under rest pose
    bpy.context.object.data.pose_position = 'REST'

    source_arm = bpy.data.objects['Armature']
    rest_x, rest_y, rest_z = source_arm.data.bones[
        0
    ].head_local  # the location of hips under rest pose
    # rest_x, rest_y, rest_z = 0, 0, 0
    for obj in scene.objects:
        if obj.type == 'MESH' and not obj.name == 'Cube':
            bme_rest = bmesh.new()
            bme_rest.from_mesh(obj.data)

            bm_rest_verts = bme_rest.verts
            bm_rest_faces_ori = bme_rest.faces
            bm_rest_faces_tri = bmesh.ops.triangulate(
                bme_rest,
                faces=bm_rest_faces_ori,
                quad_method='BEAUTY',
                ngon_method='BEAUTY',
            )['faces']

            rest_verts_lst = []
            for v in bm_rest_verts:
                rest_verts_lst.append(
                    (v.co.x - rest_x, v.co.y - rest_y, v.co.z - rest_z)
                )

            rest_faces_lst = (
                []
            )  # estimated in each frame, therefore the faces may change
            for face in bm_rest_faces_tri:
                f_verts = face.verts
                rest_faces_lst.append(
                    (f_verts[0].index, f_verts[1].index, f_verts[2].index)
                )

            np_rest_verts = np.array(rest_verts_lst)
            np_rest_faces = np.array(rest_faces_lst)

    # obtain the frame indices of begining and ending
    bpy.context.object.data.pose_position = 'POSE'
    a = bpy.context.object.animation_data.action
    frame_start, frame_end = map(int, a.frame_range)
    seq_length = frame_end - frame_start + 1

    # export bvh file and load back in with simplified version
    output_bvh_path = fbx_path.replace('fbx', 'bvh')
    bpy.ops.export_anim.bvh(
        filepath=output_bvh_path,
        frame_start=frame_start,
        frame_end=frame_end,
        root_transform_only=True,
    )

    bvh_data = BVHData(output_bvh_path)

    # ====== extract data block ======
    # extract skinning weight and simplify it
    for obj in scene.objects:
        if obj.type == 'MESH' and not obj.name == 'Cube':
            verts = obj.data.vertices
            vgrps = obj.vertex_groups  # vertex groups correspond to the joints.

            np_skinning_weights = np.zeros((len(verts), len(vgrps)))
            mask = np.zeros(np_skinning_weights.shape, dtype=np.int)
            vgrp_label = vgrps.keys()

            for i, vert in enumerate(verts):
                for g in vert.groups:
                    j = g.group
                    np_skinning_weights[i, j] = g.weight
                    mask[i, j] = 1

        if obj.type == 'ARMATURE':
            source_arm = bpy.data.objects[obj.name]

    np_simplified_skinning_weights = np.zeros(
        (len(verts), len(bvh_data.simplified_joint_names))
    )
    for j, name in enumerate(vgrp_label):
        bone = source_arm.data.bones[name]
        while (
            bone.parent is not None
            and rm_prefix(bone.name) not in bvh_data.simplified_joint_names
        ):
            bone = bone.parent

        idx = bvh_data.simplified_joint_names.index(rm_prefix(bone.name))
        np_simplified_skinning_weights[:, idx] += np_skinning_weights[:, j]

    vertex_part = np.argmax(np_simplified_skinning_weights, axis=1)
    num_face = np_rest_faces.shape[0]
    face_part = []
    for i in range(num_face):
        face_part.append(vertex_part[np_rest_faces[i][0]])

    face_part = np.array(face_part)
    body_vid_lst = []
    arm_vid_lst = []
    for i in range(vertex_part.shape[0]):
        if vertex_part[i] in body_joint_lst:
            body_vid_lst.append(i)
        if vertex_part[i] in arms_joint_lst:
            arm_vid_lst.append(i)

    rest_body_vertices = np_rest_verts[body_vid_lst, :]
    rest_arm_vertices = np_rest_verts[arm_vid_lst, :]

    body_width = get_width(rest_body_vertices)
    full_width = get_width(np_rest_verts)

    # detail shape
    joint_shape_lst = []
    for i in range(22):
        joint_i = []
        for j in range(vertex_part.shape[0]):
            if vertex_part[j] == i:
                joint_i.append(j)
        joint_shape_lst.append(joint_i)

    shape_lst = []
    for joint_i in joint_shape_lst:
        if len(joint_i) == 0:
            shape_lst.append(np.array([0, 0, 0]))
        else:
            joint_i_vertices = np_rest_verts[joint_i, :]
            joint_i_width = get_width(joint_i_vertices)
            shape_lst.append(joint_i_width)

    shape_lst_array = np.stack(shape_lst, axis=0)

    skinning_weights_data = np_simplified_skinning_weights.astype(np.single)
    joint_names_data = bvh_data.simplified_joint_names
    root_orient_data = bvh_data.simplified_axis_angle[:, :3].astype(np.single)
    rest_vertices_data = np_rest_verts.astype(np.single)
    rest_faces_data = np_rest_faces
    subject_data = subject_name
    skeleton_data = bvh_data.simplified_joint_offsets.astype(np.single)
    rest_body_vertices_data = rest_body_vertices
    rest_arm_vertices_data = rest_arm_vertices

    output_root = save_path
    output_path = os.path.join(output_root, '%s.npz' % (subject_name))

    np.savez(
        output_path,
        skinning_weights=skinning_weights_data,
        joint_names=joint_names_data,
        root_orient=root_orient_data,
        rest_vertices=rest_vertices_data,
        rest_faces=rest_faces_data,
        skeleton=skeleton_data,
        subject=subject_data,
        vertex_part=vertex_part,
        rest_body_vertices=rest_body_vertices_data,
        rest_arm_vertices=rest_arm_vertices_data,
        body_width=body_width,
        full_width=full_width,
        joint_shape=shape_lst_array,
    )

    # for obj in bpy.context.scene.objects:
    #     obj.select = True
    bpy.ops.object.delete()


if __name__ == '__main__':
    root_path = "./fbx_datapath"
    fbx_name_lst = listdir(root_path)

    for fbx_name in tqdm(fbx_name_lst):
        subject_name = fbx_name.split(".")[0]
        fbx_path = join(root_path, fbx_name)
        save_path = "./datasets/train_shape"
        extract_data(fbx_path, subject_name, save_path)
