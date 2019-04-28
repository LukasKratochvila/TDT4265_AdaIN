from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """
    This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        # Additional options
        parser.add_argument('--content_size', type=int, default=512,
                            help='New (minimum) size for the content image, \
                            keeping the original size if set to 0 [default: 512]')
        parser.add_argument('--style_size', type=int, default=512,
                            help='New (minimum) size for the style image, \
                            keeping the original size if set to 0 [default: 512]')
        parser.add_argument('--crop', action='store_true',
                            help='Do center crop to create squared image [default: False]')
        parser.add_argument('--save_ext', default='.jpg',
                            help='The extension name of the output image [default: .jpg]')
        parser.add_argument('--output', type=str, default='output',
                            help='Directory to save the output image(s) [default: output]')
        parser.add_argument('--only_loss', action='store_true',
                            help='Not save output, only print loss [default: False]')
        # Advanced options
        parser.add_argument('--preserve_color', action='store_true',
                            help='If specified, preserve color of the content image')
        parser.add_argument('--alpha', type=float, default=1.0,
                            help='The weight that controls the degree of \
                                 stylization. Should be between 0 and 1')
        parser.add_argument('--style_interpolation_weights', type=str, default='',
                            help='The weight for blending the style of multiple style images')
        self.isTrain = False
        return parser
