blender -P visualize.py --dcopy_result0_bvh './bvh_files/Robot_to_Mousey_Gangnam Style.bvh'\
                        --dcopy_result1_bvh './bvh_files/Robot_to_Ortiz_Gangnam Style.bvh'\
                        --dcopy_result2_bvh './bvh_files/Robot_to_Sporty_Granny_Gangnam Style.bvh'\
                        --dcopy_fbx_file0 './skin/Mousey.fbx'\
                        --dcopy_fbx_file2 './skin/Ortiz.fbx'\
                        --dcopy_fbx_file2 './skin/Sporty_granny.fbx'\
                        --render_engine eevee --render\
                        --frame_end 313 --fps 30 --r_front
