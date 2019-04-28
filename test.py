import os
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
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def style_transfer(encoder, decoder, content, style, alpha=1.0,
                   interpolation_weights=None):
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


args = TestOptions().parse()  # get test options

do_interpolation = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.content:
    content_paths = args.content.split(',')
    if len(content_paths) == 1:
        content_paths = [args.content]
else:
    content_paths = [os.path.join(args.content_dir, f) for f in
                     os.listdir(args.content_dir)]

if args.style:
    style_paths = args.style.split(',')
    if len(style_paths) == 1:
        style_paths = [args.style]
    elif args.style_interpolation_weights != '':
        do_interpolation = True
        weights = [int(i) for i in args.style_interpolation_weights.split(',')]
        interpolation_weights = [w / sum(weights) for w in weights]
else:
    style_paths = [os.path.join(args.style_dir, f) for f in
                   os.listdir(args.style_dir)]

if not os.path.exists(args.output):
    os.mkdir(args.output)

# if we don't get the model we use vgg
if args.enc_w == 'weights/vgg_normalised.pth':
    encoder = VGG19.vgg19(args.enc_w)
else:
    assert False,"Wrong encoder"
        
if args.dec == 'VGG19':
    decoder = VGG19.vgg19_dec(args.dec_w, args.dec_BN)
elif args.dec == 'VGG19B':
    decoder = VGG19.vgg19B_dec(args.dec_w, args.dec_BN)
elif args.dec == 'resnet18':
    decoder = ResNet.resnet18_dec(args.dec_w)
elif args.dec == 'inceptionv3':
    decoder = InceptionV3.inception3_dec(args.dec_w)
else:
    assert False,"Wrong decoder"
    
network = net.Net(encoder, decoder)

network.decoder.eval()
for i in range(network.num_enc):
    getattr(network, 'enc_{:d}'.format(i + 1)).eval()

network.to(device)

content_tf = test_transform(args.content_size, args.crop)
style_tf = test_transform(args.style_size, args.crop)

content_losses = torch.FloatTensor(len(content_paths),\
                            1 if do_interpolation else len(style_paths),\
                                 network.num_enc+1).zero_()
style_losses = torch.FloatTensor(len(content_paths),\
                          1 if do_interpolation else len(style_paths),\
                          network.num_enc+1).zero_()
time_elapsed = torch.FloatTensor(len(content_paths), 1 if do_interpolation \
                                 else len(style_paths)).zero_()
for i,content_path in enumerate(content_paths):
    start_time = time.time()
    if do_interpolation:  # one content image, N style image
        style = torch.stack([style_tf(Image.open(p).convert('RGB')) for p in style_paths])
        content = content_tf(Image.open(content_path).convert('RGB')) \
            .unsqueeze(0).expand_as(style)
        style = style.to(device)
        content = content.to(device)
        with torch.no_grad():
            output = style_transfer(network.encode, network.decoder, content, style,
                                    args.alpha, interpolation_weights)
        output = output.cpu()
        output_name = '{:s}/{:s}_interpolation{:s}'.format(
            args.output, splitext(basename(content_path))[0], args.save_ext)
        if not args.only_loss:
            save_image(output, output_name)
        
        time_elapsed[i] = time.time() - start_time
        
        style_c = torch.FloatTensor(style[:1].shape).zero_()
        for i, w in enumerate(interpolation_weights):
            style_c += w * style[i] 
        size = torch.min(torch.tensor((content.shape,output.shape,style_c.shape)),0)
        result = output[:size[0][0],:size[0][1],:size[0][2],:size[0][3]]
        result = result.to(device)
        content_c = content[:size[0][0],:size[0][1],:size[0][2],:size[0][3]]
        style_c = style[:size[0][0],:size[0][1],:size[0][2],:size[0][3]]
        content_losses[i] = network.losses(content_c,result).cpu()
        style_losses[i] = network.losses(style_c,result).cpu()
        
    else:  # process one content and one style
        for j,style_path in enumerate(style_paths):
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
            time_elapsed[i][j] = time.time() - start_time
            
            size = torch.min(torch.tensor((content.shape,output.shape,style.shape)),0)
            result = output[:size[0][0],:size[0][1],:size[0][2],:size[0][3]]
            result = result.to(device)
            content_c = content[:size[0][0],:size[0][1],:size[0][2],:size[0][3]]
            style_c = style[:size[0][0],:size[0][1],:size[0][2],:size[0][3]]
            content_losses[i][j] = network.losses(content_c,result).cpu()
            style_losses[i][j] = network.losses(style_c,result).cpu()
message = ""
for i in range(1,content_losses.shape[2]):
    message += " level {:d}: {:.3f}".format(i, content_losses.transpose(0,2)[i].mean())
print("Content img - content loss %.3f"%content_losses.transpose(0,2)[0].mean(),"style loss", message)
message = ""
for i in range(1,style_losses.shape[2]):
    message += " level {:d}: {:.3f}".format(i, style_losses.transpose(0,2)[i].mean())
print("Style img - content loss %.3f"%style_losses.transpose(0,2)[0].mean(),"style loss", message)
print("Time for one image {:.3f}sec, {:.3f} img per sec".format(time_elapsed.mean(),1/time_elapsed.mean()))