import bpy


def add_floor(size):
    bpy.ops.mesh.primitive_plane_add(size=size, enter_editmode=False, location=(0, 0, 0))
    floor = bpy.context.object
    floor.name = 'floor'

    floor_mat = bpy.data.materials.new(name="floorMaterial")
    floor_mat.use_nodes = True
    bsdf = floor_mat.node_tree.nodes["Principled BSDF"]
    floor_text = floor_mat.node_tree.nodes.new("ShaderNodeTexChecker")
    floor_text.inputs[3].default_value = 150
    floor_mat.node_tree.links.new(bsdf.inputs['Base Color'], floor_text.outputs['Color'])
    bpy.context.object.rotation_euler[2] = 0.523599
    floor.data.materials.append(floor_mat)
    return floor


def add_camera(location, rotation):
    bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=location, rotation=rotation)
    camera = bpy.context.object
    return camera


def add_light(location):
    bpy.ops.object.light_add(type='SUN', location=location)
    sun = bpy.context.object
    bpy.context.object.rotation_euler[0] =  3.1416 # -1.5708
    bpy.context.object.rotation_euler[1] =  1.9189 # 2.2678 #  1.5708
    bpy.context.object.rotation_euler[2] =  3.1416 # -1.5708
    bpy.context.object.data.energy = 3.0

    # bpy.context.object.rotation_euler[1] = 1.5708
    # bpy.context.object.data.energy = 5
    # bpy.context.object.data.size = 10

    return sun



def make_scene(floor_size=1000, camera_position=(17.229, 0.5, 2.126), camera_rotation=(1.5359, 0, 1.57), light_position=(500, 0, 30)):    #  m37.54/7   -28.87/7   16.34/7
    floor = add_floor(floor_size)
    camera = add_camera(camera_position, camera_rotation)
    light = add_light(light_position)
    bpy.ops.object.select_all(action='DESELECT')
    floor.select_set(True)
    camera.select_set(True)
    light.select_set(True)
    bpy.ops.object.move_to_collection(collection_index=0, is_new=True, new_collection_name="Scene")
    bpy.ops.object.select_all(action='DESELECT')
    return [floor, camera, light]


def add_rendering_parameters(scene, args, camera):
    scene.render.resolution_x = args.resX
    scene.render.resolution_y = args.resY
    scene.render.fps = args.fps
    scene.frame_end = args.frame_end
    scene.camera = camera
    scene.render.filepath = args.save_path

    if args.render_engine == 'cycles':
        scene.render.engine = 'CYCLES'
        scene.cycles.device = 'GPU'
    elif args.render_engine == 'eevee':
        scene.render.engine = 'BLENDER_EEVEE'

    scene.render.image_settings.file_format = 'AVI_JPEG'
    return scene


def add_material_for_character(objs, color, alpha):
    char_mat = bpy.data.materials.new(name="characterMaterial")
    char_mat.use_nodes = True
    bsdf = char_mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs[0].default_value = color # (0.021219, 0.278894, 1, 1)   # character material color
    # bsdf.inputs[0].alpha = alpha
    for obj in objs:
        obj.data.materials.append(char_mat)
