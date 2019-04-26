from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # training options
        parser.add_argument('--save_dir', default='./experiments',
                            help='Directory to save the model')
        parser.add_argument('--name', type=str, default='experiment_name',
                            help='name of the experiment. It decides where to store samples and models')
        
        #parser.add_argument('--log_dir', default='./logs',
        #                    help='Directory to save the log')
        parser.add_argument('--lr', type=float, default=1e-4,
                            help='Learning rate')
        parser.add_argument('--lr_decay', type=float, default=5e-5,
                             help='Learning rate decay')
        parser.add_argument('--max_iter', type=int, default=160000,
                             help='Max iteration')
        parser.add_argument('--batch_size', type=int, default=8,
                             help='Image batch size')
        parser.add_argument('--style_weight', type=float, default=10.0,
                             help='Weight for style during learning')
        parser.add_argument('--content_weight', type=float, default=1.0,
                            help='Weight for content during learning')
        parser.add_argument('--n_threads', type=int, default=16,
                            help='Number of threads for DataLoader')
        parser.add_argument('--save_model_interval', type=int, default=10000,
                            help='Save interval')
        parser.add_argument('--verbose', action='store_false',
                            help='if specified, print less debugging information')
        self.isTrain = True
        return parser
