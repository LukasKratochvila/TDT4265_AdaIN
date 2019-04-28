import argparse
import os



class BaseOptions():
    """
    This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """
        Reset the class; indicates the class hasn't been initailized
        """
        self.initialized = False

    def initialize(self, parser):
        """
        Define the common options that are used in both training and test.
        """
        # Basic parameters
        parser.add_argument('--content', type=str,
                            help='File path to the content image or multiple content \
                            images separated by commas')
        parser.add_argument('--content_dir', type=str,
                            help='Directory path to directory of content images')
        parser.add_argument('--style', type=str,
                            help='File path to the style image, or multiple style \
                            images separated by commas')
        parser.add_argument('--style_dir', type=str,
                            help='Directory path to directory of style images')
        parser.add_argument('--enc_w', type=str, default='weights/vgg_normalised.pth',
                            help='Specify encoder weights [default: weights/vgg_normalised.pth]')
        parser.add_argument('--dec', type=str, default='VGG19',
                            help='Specify decoder architecture [VGG19 | resnet18 | inceptionv3] [default: VGG19]')
        parser.add_argument('--dec_w', type=str, default='weights/decoder.pth', 
                            help='Specify decoder weights [default: weights/decoder.pth]')
        parser.add_argument('--dec_BN', action='store_true',
                            help='Use decoder with batch normalization [default: False]')
        self.initialized = True
        return parser

    def gather_options(self):
        """
        Initialize our parser with basic options(only once).
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        # save and return the parser
        self.parser = parser
        
        return parser.parse_args()

    def print_options(self, opt):
        """
        Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in vars(opt).items():
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>27}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # If trainig save to the disk
        if opt.isTrain:
            if isinstance(opt.expr_dir, list) and not isinstance(opt.expr_dir, str):
                for path in opt.expr_dir:
                    if not os.path.exists(path):
                        os.mkdir(path)
                else:
                    if not os.path.exists(path):
                        os.mkdir(path)
            else:
                if not os.path.exists(opt.expr_dir):
                    os.mkdir(opt.expr_dir)
        
            file_name = os.path.join(opt.expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write(message)
                opt_file.write('\n')

    def parse(self):
        """
        Parse our options. If train then check dirs. If test parse the contents and styles paths and weights of interpolation.
        """
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test
        self.print_options(opt)
        if opt.isTrain:
            opt.expr_dir = os.path.join(opt.save_dir, opt.name)
            if not os.path.exists(opt.save_dir):
                os.mkdir(opt.save_dir)                
            assert (opt.content_dir or opt.style_dir), "Missing content or style dir"
        else: 
            if opt.content:
                opt.content_paths = opt.content.split(',')
                if len(opt.content_paths) == 1:
                    opt.content_paths = [opt.content]
            else:
                opt.content_paths = [os.path.join(opt.content_dir, f) for f in os.listdir(opt.content_dir)]
            opt.do_interpolation = False
            if opt.style:
                opt.style_paths = opt.style.split(',')
                if len(opt.style_paths) == 1:
                    opt.style_paths = [opt.style]
                elif opt.style_interpolation_weights != '':
                    opt.do_interpolation = True
                    weights = [int(i) for i in opt.style_interpolation_weights.split(',')]
                    opt.interpolation_weights = [w / sum(weights) for w in weights]
            else:
                opt.style_paths = [os.path.join(opt.style_dir, f) for f in
                                       os.listdir(opt.style_dir)]
            if not os.path.exists(opt.output):
                os.mkdir(opt.output)
        # Either --content or --contentDir should be given.
        assert (opt.content or opt.content_dir), "Missing content image/s"
        # Either --style or --styleDir should be given.
        assert (opt.style or opt.style_dir), "Missing style image/s"

        self.opt = opt
        return self.opt
