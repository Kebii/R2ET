import bpy
import sys

sys.path.append('./')
import numpy as np
import argparse
import os
from scene import make_scene, add_material_for_character, add_rendering_parameters
from options import Options
from load_bvh import load_bvh

import pdb


colors = [
    (0.278894, 0.278894, 0.278894, 1, 1),  # input color
    (57.0 / 255, 221.0 / 255, 109.0 / 255, 0, 0.7),  # dcopy ground truht color
    (240.0 / 255, 93.0 / 255, 1.0 / 255, 1, 1),  # dcopy result color
    (57.0 / 255, 221.0 / 255, 109.0 / 255, 0, 0.7),  # intra ground truht color 1
    (19.0 / 255, 73.0 / 255, 152.0 / 255, 1, 1),  # intra result color
    (57.0 / 255, 221.0 / 255, 109.0 / 255, 0, 0.7),  # cross ground truht color 2
    (93.0 / 255, 1.0 / 255, 240.0 / 255, 1, 1),
]  # cross result color 2

cam_pos = {
    'front': (17.229, 0.0, 2.126),  # front
    'r_front': (12.263, 11.743, 5.9369),  # right_front
    'l_front': (15.483, -7.9034, 4.6168),  # left_front
    'back': (-17.229, 0.0, 2.126),  # front
    'top': (0.0, 0.0, 13.229),
}

cam_rot = {
    'front': (1.5359, 0, 1.57),  # front
    'r_front': (1.2333, 0.0139, 2.320),  # right_front
    'l_front': (1.3101, 0.0139, 1.0934),  # left_front
    'back': (1.5359, 0, -1.57),  # front
    'top': (0.0, 0.0, 1.57),
}


def clean_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()


def load_fbx(source):
    bpy.ops.import_scene.fbx(filepath=source, use_anim=False)


def load_armature(source, source_format=None):
    bpy.ops.import_anim.bvh(filepath=source)
    return source.split('/')[-1][:-4]


def set_rest_pose_bvh(filename, source_arm, source_format=None):
    """
    This helps recover the rest pose position from the rest pose of fbx reference file
    """
    dest_filename = filename[:-4] + '_tmp.bvh'
    dest_file = open(dest_filename, 'w')
    rest_loc = source_arm.data.bones[0].head_local  # 0:hips 1:pelvis 12:spine

    source_file = open(filename, 'r')
    content = source_file.readlines()

    flag = 0
    for i in range(len(content)):
        if 'ROOT' in content[i]:
            content[i + 2] = '\tOFFSET %.6f %.6f %.6f\n' % (
                rest_loc[0],
                rest_loc[1],
                rest_loc[2],
            )  # rest_loc[0], rest_loc[1]
            flag = 1
            break

    if flag == 0:
        raise Exception('Illegal bvh file')

    dest_file.write(''.join(content))
    dest_file.close()
    return dest_filename


def extract_weight(me):
    """
    Extract skinning weight from a given mesh
    """
    verts = me.data.vertices
    vgrps = me.vertex_groups

    weight = np.zeros((len(verts), len(vgrps)))
    mask = np.zeros(weight.shape, dtype=np.int)
    vgrp_label = vgrps.keys()

    for i, vert in enumerate(verts):
        for g in vert.groups:
            j = g.group
            weight[i, j] = g.weight
            mask[i, j] = 1

    return weight, vgrp_label, mask


def clean_vgrps(me):
    vgrps = me.vertex_groups
    for _ in range(len(vgrps)):
        vgrps.remove(vgrps[0])


def load_weight(me, label, weight):
    clean_vgrps(me)
    verts = me.data.vertices
    vgrps = me.vertex_groups

    for name in label:
        vgrps.new(name=name)

    for j in range(weight.shape[1]):
        idx = vgrps.find(label[j])
        if idx == -1:
            continue

        for i in range(weight.shape[0]):
            vgrps[idx].add([i], weight[i, j], 'REPLACE')


def set_modifier(me, arm):
    modifiers = me.modifiers
    for modifier in modifiers:
        if modifier.type == 'ARMATURE':
            modifier.object = arm
            modifier.use_vertex_groups = True
            modifier.use_deform_preserve_volume = True
            return

    modifiers.new(name='Armature', type='ARMATURE')
    modifier = modifiers[0]
    modifier.object = arm
    modifier.use_vertex_groups = True
    modifier.use_deform_preserve_volume = True


def adapt_weight(source_weight, source_label, source_arm, dest_arm):
    """
    The targeted armature could be a reduced one, e.g. no fingers. So move the skinning weight of each reduced armature to its nearest ancestor.
    """
    weight = np.zeros((source_weight.shape[0], len(dest_arm.data.bones)))

    # Skinning weight is bond to armature names. For simplicity, a common prefix
    # is removed in our retargeting output. Here we solve this problem.
    prefix = ''
    ref_name = source_arm.data.bones[0].name

    if ':' in ref_name and ':' not in dest_arm.data.bones[0].name:
        idx = ref_name.index(':')
        prefix = ref_name[: idx + 1]
    dest_name = [prefix + bone.name for bone in dest_arm.data.bones]
    # for i, d_name in enumerate(dest_name):
    #     dest_name[i] = d_name.replace('Warrok', 'mixamorig')

    for j, name in enumerate(source_label):
        bone = source_arm.data.bones.find(name)
        bone = source_arm.data.bones[bone]
        while bone.parent is not None and bone.name not in dest_name:
            bone = bone.parent
        idx = dest_name.index(bone.name)
        weight[:, idx] += source_weight[:, j]

    return weight


def in_other_colls():
    exist_obj = []
    for coll in bpy.data.collections:
        for obj in coll.all_objects:
            exist_obj.append(obj.name)

    return exist_obj


def check_repeat_name(name, exist_lst, current_lst):
    for obj in bpy.data.objects:
        if not (obj.name in exist_lst or obj.name in current_lst) and name in obj.name:
            return obj.name


def mesh_visualize(
    input_fbx,
    input_bvh,
    x_bias=0,
    y_bias=0,
    z_bias=0,
    collection_name='Character Mesh',
    source_format=None,
):
    exist_obj_names = in_other_colls()

    load_fbx(input_fbx)
    bpy.context.object.rotation_euler[2] = 1.5708
    bpy.context.object.location[0] = x_bias
    bpy.context.object.location[1] = y_bias
    bpy.context.object.location[2] = z_bias

    for obj in bpy.data.objects:
        if ('Armature' in obj.name) and (not obj.name in exist_obj_names):
            armature_name = obj.name
    source_arm = bpy.data.objects[armature_name]

    meshes = []
    current_obj_names = []
    for obj in bpy.data.objects:
        if not (obj.name in exist_obj_names):
            current_obj_names.append(obj.name)
            if obj.type == 'MESH':
                meshes.append(obj)

    bvh_file = set_rest_pose_bvh(input_bvh, source_arm, source_format)
    bvh_name = load_armature(bvh_file, source_format)
    bpy.context.object.rotation_euler[2] = 1.5708
    bpy.context.object.location[0] = x_bias
    bpy.context.object.location[1] = y_bias
    bpy.context.object.location[2] = z_bias

    bvh_name = check_repeat_name(bvh_name, exist_obj_names, current_obj_names)
    dest_arm = bpy.data.objects[bvh_name]
    dest_arm.scale = source_arm.scale  # scale the bvh to match the fbx

    for me in meshes:
        weight, label, _ = extract_weight(me)
        weight = adapt_weight(weight, label, source_arm, dest_arm)
        load_weight(me, dest_arm.data.bones.keys(), weight)
        set_modifier(me, dest_arm)

    for obj in current_obj_names:
        bpy.data.objects[obj].select_set(True)

    bpy.ops.object.move_to_collection(
        collection_index=0, is_new=True, new_collection_name=collection_name
    )
    source_arm.hide_viewport = True
    bpy.ops.object.select_all(action='DESELECT')

    os.system('rm %s' % bvh_file)  # remove temporary file


def main():
    print(sys.argv)
    args = Options(sys.argv).parse()

    clean_scene()

    # Character Mesh Visualization
    mesh_visualize(
        args.dcopy_fbx_file0,
        args.dcopy_result0_bvh,
        y_bias=-2.0,
        collection_name='Copy Mesh1',
        source_format='h36m',
    )
    mesh_visualize(
        args.dcopy_fbx_file1,
        args.dcopy_result1_bvh,
        y_bias=0.0,
        collection_name='Copy Mesh1',
        source_format='h36m',
    )
    mesh_visualize(
        args.dcopy_fbx_file2,
        args.dcopy_result2_bvh,
        y_bias=2.0,
        collection_name='Copy Mesh2',
        source_format='h36m',
    )

    # Skeleton Visualization
    characters = []
    # characters.append(load_bvh(args.input_bvh, x_bias=0, y_bias=-1.5, z_bias=0,collecttion_name='input', format_type='h36m'))
    # characters.append(load_bvh(args.dcopy_result1_bvh, x_bias=0, y_bias=0, z_bias=0,collecttion_name='dcopy_result1', format_type='h36m_copy'))
    # characters.append(load_bvh(args.dcopy_result2_bvh, x_bias=0, y_bias=1.5, z_bias=0,collecttion_name='dcopy_result2', format_type='h36m_copy'))

    for i, character in enumerate(characters):
        add_material_for_character(character, colors[i * 2][:4], colors[i][4])

    scene = make_scene(
        camera_position=cam_pos[args.view], camera_rotation=cam_rot[args.view]
    )
    add_rendering_parameters(bpy.context.scene, args, scene[1])
    bpy.ops.object.select_all(action='DESELECT')

    if args.render:
        bpy.ops.render.render(animation=True, use_viewport=True)


if __name__ == "__main__":
    main()
