import argparse


class Options:
    def __init__(self, argv):
        if "--" not in argv:
            self.argv = []  # as if no args are passed
        else:
            self.argv = argv[argv.index("--") + 1 :]

        usage_text = (
            "Run blender in background mode with this script:"
            "  blender --background --python [main_python_file] -- [options]"
        )

        self.parser = argparse.ArgumentParser(description=usage_text)
        self.initialize()
        self.args = self.parser.parse_args(self.argv)

    def initialize(self):
        self.parser.add_argument(
            '--dcopy_result0_bvh',
            type=str,
            default='./bvh_files/Robot_to_Mousey_Gangnam Style.bvh',
            help='path of bvh file',
        )
        self.parser.add_argument(
            '--dcopy_result1_bvh',
            type=str,
            default='./bvh_files/Robot_to_Ortiz_Gangnam Style.bvh',
            help='path of bvh file',
        )
        self.parser.add_argument(
            '--dcopy_result2_bvh',
            type=str,
            default='./bvh_files/Robot_to_Sporty_Granny_Gangnam Style.bvh',
            help='path of bvh file',
        )
        self.parser.add_argument(
            '--dcopy_fbx_file0',
            type=str,
            default='./skin/Mousey.fbx',
            help='path of skinned model fbx file',
        )
        self.parser.add_argument(
            '--dcopy_fbx_file1',
            type=str,
            default='./skin/Ortiz.fbx',
            help='path of skinned model fbx file',
        )
        self.parser.add_argument(
            '--dcopy_fbx_file2',
            type=str,
            default='./skin/Sporty_granny.fbx',
            help='path of skinned model fbx file',
        )

        self.parser.add_argument(
            '--save_path',
            type=str,
            default='./videos/',
            help='path of output video file',
        )

        # scene parameters
        self.parser.add_argument(
            '--view', type=str, default='front', help='view of the camera'
        )

        # rendering parameters
        self.parser.add_argument(
            '--render', action='store_true', default=True, help='render an output video'
        )
        self.parser.add_argument(
            '--render_engine',
            type=str,
            default='eevee',
            help='name of preferable render engine: cycles, eevee',
        )
        self.parser.add_argument(
            '--frame_end',
            type=int,
            default=2,
            help='the index of the last rendered frame',
        )
        self.parser.add_argument(
            '--fps', type=int, default=25, help='the frame rate of rendered video'
        )
        self.parser.add_argument('--resX', type=int, default=1920, help='x resolution')
        self.parser.add_argument('--resY', type=int, default=1080, help='y resolution')

    def parse(self):
        return self.args
