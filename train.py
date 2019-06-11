import os
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from networks import net
from networks import VGG19
from networks import ResNet
from networks import InceptionV3
from options.train_options import TrainOptions

from sampler import InfiniteSamplerWrapper

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated


def train_transform():
    """
    Definition of train transform - resize and crop.
    """
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class FlatFolderDataset(data.Dataset):
    """
    Definition of Dataset for DataLoader.
    """
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = os.listdir(self.root)
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


def adjust_learning_rate(optimizer, iteration_count):
    """
    Imitating the original implementation
    """
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Getting train options from command line
args = TrainOptions().parse()

# Choose encoder architecture and load weights
if args.enc_w == 'weights/vgg_normalised.pth':
    encoder = VGG19.vgg19(args.enc_w)
else:
    assert False,"Wrong encoder"

# Choose decoder architecture     
if args.dec == 'VGG19':
    decoder = VGG19.vgg19_dec(BN=args.dec_BN)
elif args.dec == 'VGG19B':
    decoder = VGG19.vgg19B_dec(BN=args.dec_BN)
elif args.dec == 'resnet18':
    decoder = ResNet.resnet18_dec()
elif args.dec == 'inceptionv3':
    decoder = InceptionV3.inception3_dec()
else:
    assert False,"Wrong decoder"

########################################################################           
#                        Initializaton                                 #
########################################################################
    
# Select cuda device for training, cpu is not option
device = torch.device('cuda')

# Initialization training architecture  
network = net.Net(encoder, decoder)
network.print_networks(args.expr_dir,args.verbose)
network.train()
network.to(device)

# Open log file
writer = SummaryWriter(log_dir=args.expr_dir)  

# Initialization of transformation
content_tf = train_transform()
style_tf = train_transform()

# Initialization of Datasets
content_dataset = FlatFolderDataset(args.content_dir, content_tf)
style_dataset = FlatFolderDataset(args.style_dir, style_tf)

# Initialization of DataLoader iter
content_iter = iter(data.DataLoader(
    content_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))
style_iter = iter(data.DataLoader(
    style_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.n_threads))

# Initialization of optimizer
optimizer = torch.optim.Adam(network.decoder.parameters(), lr=args.lr)

########################################################################           
#                        Training                                      #
########################################################################

for i in tqdm(range(args.max_iter)):
    adjust_learning_rate(optimizer, iteration_count=i)
    content_images = next(content_iter).to(device)
    style_images = next(style_iter).to(device)
    loss_c, loss_s = network(content_images, style_images)
    loss_c = args.content_weight * loss_c
    loss_s = args.style_weight * loss_s
    loss = loss_c + loss_s

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    writer.add_scalar('loss_content', loss_c.item(), i + 1)
    writer.add_scalar('loss_style', loss_s.item(), i + 1)

    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        state_dict = network.decoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/decoder_iter_{:d}.pth'.format(args.expr_dir, i + 1))
writer.close()
