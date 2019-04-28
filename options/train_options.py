from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """
    This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # training options
        parser.add_argument('--save_dir', default='./experiments',
                            help='Directory to save the model [default: ./experiments]')
        parser.add_argument('--name', type=str, default='experiment_name',
                            help='name of the experiment. It decides where to store samples and models')
        
        parser.add_argument('--lr', type=float, default=1e-4,
                            help='Learning rate [default: 1e-4]')
        parser.add_argument('--lr_decay', type=float, default=5e-5,
                             help='Learning rate decay [default: 5e-5]')
        parser.add_argument('--max_iter', type=int, default=160000,
                             help='Max iteration [default: 160000]')
        parser.add_argument('--batch_size', type=int, default=8,
                             help='Image batch size [default: 8]')
        parser.add_argument('--style_weight', type=float, default=10.0,
                             help='Weight for style during learning [default: 10]')
        parser.add_argument('--content_weight', type=float, default=1.0,
                            help='Weight for content during learning [default: 1.0]')
        parser.add_argument('--n_threads', type=int, default=16,
                            help='Number of threads for DataLoader [default: 16]')
        parser.add_argument('--save_model_interval', type=int, default=10000,
                            help='Save interval [default: 10000]')
        parser.add_argument('--verbose', action='store_false',
                            help='if specified, print less debugging information')
        self.isTrain = True
        return parser
