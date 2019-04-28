import torch
import time
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image

from networks import net
from networks import VGG19
from networks import ResNet
from networks import InceptionV3
from options.test_options import TestOptions

from function import adaptive_instance_normalization
from function import coral


def test_transform(size, crop):
    """
    Definition of train transform - resize and crop.
    """
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def style_transfer(encoder, decoder, content, style, alpha=1.0,interpolation_weights=None):
    """
    Fuction that implements the Encode -> AdaIN -> Decode
    """
    assert (0.0 <= alpha <= 1.0)
    content_f = encoder(content)
    style_f = encoder(style)
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = adaptive_instance_normalization(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)

# Getting test options from command line
args = TestOptions().parse() 

# Choose encoder architecture and load weights
if args.enc_w == 'weights/vgg_normalised.pth':
    encoder = VGG19.vgg19(args.enc_w)
else:
    assert False,"Wrong encoder"

# Choose decoder architecture and load weights        
if args.dec == 'VGG19':
    decoder = VGG19.vgg19_dec(args.dec_w, args.dec_BN)
elif args.dec == 'VGG19B':
    decoder = VGG19.vgg19B_dec(args.dec_w)
elif args.dec == 'resnet18':
    decoder = ResNet.resnet18_dec(args.dec_w)
elif args.dec == 'inceptionv3':
    decoder = InceptionV3.inception3_dec(args.dec_w)
else:
    assert False,"Wrong decoder"

########################################################################           
#                        Initializaton                                 #
########################################################################   
    
# Select cuda or cpu device for  testing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialization testing architecture
network = net.Net(encoder, decoder)
network.decoder.eval()
for i in range(network.num_enc):
    getattr(network, 'enc_{:d}'.format(i + 1)).eval()
network.to(device)

# Initialization of transformation
content_tf = test_transform(args.content_size, args.crop)
style_tf = test_transform(args.style_size, args.crop)

# Initialization of losses tensors
content_losses = torch.FloatTensor(len(args.content_paths),\
                            1 if args.do_interpolation else len(args.style_paths),\
                                 network.num_enc+1).zero_()
style_losses = torch.FloatTensor(len(args.content_paths),\
                          1 if args.do_interpolation else len(args.style_paths),\
                          network.num_enc+1).zero_()
time_elapsed = torch.FloatTensor(len(args.content_paths), 1 if args.do_interpolation \
                                 else len(args.style_paths)).zero_()

########################################################################           
#                        Apply style transfer                          #
########################################################################

for i,content_path in enumerate(args.content_paths):
    start_time = time.time() # For count time
    if args.do_interpolation:  # one content image, N style image
        style = torch.stack([style_tf(Image.open(p).convert('RGB')) for p in args.style_paths])
        content = content_tf(Image.open(content_path).convert('RGB')) \
            .unsqueeze(0).expand_as(style)
        style = style.to(device)
        content = content.to(device)
        with torch.no_grad():
            output = style_transfer(network.encode, network.decoder, content, style,
                                    args.alpha, args.interpolation_weights)
        output = output.cpu()
        output_name = '{:s}/{:s}_interpolation{:s}'.format(
            args.output, splitext(basename(content_path))[0], args.save_ext)
        if not args.only_loss:
            save_image(output, output_name)
        
        time_elapsed[i] = time.time() - start_time # Count time
        
        # Getting losses
        style_c = torch.FloatTensor(style[:1].shape).zero_()
        for i, w in enumerate(args.interpolation_weights):
            style_c += w * style[i] 
        size = torch.min(torch.tensor((content.shape,output.shape,style_c.shape)),0)
        result = output[:size[0][0],:size[0][1],:size[0][2],:size[0][3]]
        result = result.to(device)
        content_c = content[:size[0][0],:size[0][1],:size[0][2],:size[0][3]]
        style_c = style[:size[0][0],:size[0][1],:size[0][2],:size[0][3]]
        content_losses[i] = network.losses(content_c,result).cpu()
        style_losses[i] = network.losses(style_c,result).cpu()
        
    else:  # process one content and one style
        for j,style_path in enumerate(args.style_paths):
            content = content_tf(Image.open(content_path).convert('RGB'))
            style = style_tf(Image.open(style_path).convert('RGB'))
            if args.preserve_color:
                style = coral(style, content)
            style = style.to(device).unsqueeze(0)
            content = content.to(device).unsqueeze(0)
            with torch.no_grad():
                output = style_transfer(network.encode, network.decoder,
                                        content, style, args.alpha)
            output = output.cpu()
            
            output_name = '{:s}/{:s}_stylized_{:s}{:s}'.format(
                args.output, splitext(basename(content_path))[0],
                splitext(basename(style_path))[0], args.save_ext)
            if not args.only_loss:
                save_image(output, output_name)
            time_elapsed[i][j] = time.time() - start_time # Count time
            
            # Getting losses
            size = torch.min(torch.tensor((content.shape,output.shape,style.shape)),0)
            result = output[:size[0][0],:size[0][1],:size[0][2],:size[0][3]]
            result = result.to(device)
            content_c = content[:size[0][0],:size[0][1],:size[0][2],:size[0][3]]
            style_c = style[:size[0][0],:size[0][1],:size[0][2],:size[0][3]]
            content_losses[i][j] = network.losses(content_c,result).cpu()
            style_losses[i][j] = network.losses(style_c,result).cpu()

########################################################################           
#                        Printing losses                               #
########################################################################

print("Content loss %.3f"%content_losses.transpose(0,2)[0].mean())
s_loss_sum = 0
for i in range(1,style_losses.shape[2]):
   s_loss_sum += style_losses.transpose(0,2)[i].mean()
print("Style loss: {:.3f}".format(s_loss_sum * 8)) # Fitting scale of original implementation
print("Time for one image {:.3f}sec, {:.3f} img per sec".format(time_elapsed.mean(),1/time_elapsed.mean()))