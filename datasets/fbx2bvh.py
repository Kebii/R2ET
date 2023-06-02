import pdb
import bpy
import numpy as np

from os import listdir, makedirs, system
from os.path import exists

data_path = "./datasets/mixamo/train_char/"
directories = sorted([f for f in listdir(data_path) if not f.startswith(".")])
for d in directories:
    files = sorted([f for f in listdir(data_path + d) if f.endswith(".fbx")])

    for f in files:
        sourcepath = data_path + d + "/" + f
        dumppath = data_path + d + "/" + f.split(".fbx")[0] + ".bvh"
        if exists(dumppath):
            continue

        bpy.ops.import_scene.fbx(filepath=sourcepath)

        frame_start = int(9999)
        frame_end = int(-9999)
        action = bpy.data.actions[-1]
        if action.frame_range[1] > frame_end:
            frame_end = int(action.frame_range[1])
        if action.frame_range[0] < frame_start:
            frame_start = int(action.frame_range[0])

        frame_end = np.max([60, frame_end])
        bpy.ops.export_anim.bvh(
            filepath=dumppath,
            frame_start=frame_start,
            frame_end=frame_end,
            root_transform_only=True,
        )
        bpy.data.actions.remove(bpy.data.actions[-1])
        print(data_path + d + "/" + f + " processed.")
