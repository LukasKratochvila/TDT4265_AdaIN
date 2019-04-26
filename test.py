import os
import torch
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

def style_transfer(vgg, decoder, content, style, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
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
    content_paths = [args.content]
else:
    content_paths = [os.path.join(args.content_dir, f) for f in
                     os.listdir(args.content_dir)]

if args.style:
    style_paths = args.style.split(',')
    if len(style_paths) == 1:
        style_paths = [args.style]
    else:
        do_interpolation = True
        assert (args.style_interpolation_weights != ''), \
            'Please specify interpolation weights'
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
    decoder = VGG19.vgg19_dec(args.dec_w)
elif args.dec == 'VGG19B':
    decoder = VGG19.vgg19B_dec(args.dec_w)
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

content_loss = torch.Tensor(len(content_paths),\
                            1 if do_interpolation else len(style_paths))
style_loss = torch.Tensor(len(content_paths),\
                          1 if do_interpolation else len(style_paths),\
                          network.num_enc)
for content_path in content_paths:
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
        
        result = output[:style.shape[0],:style.shape[1],:style.shape[2],:style.shape[3]]
        content_loss[content_paths.index(content_path)], \
        style_loss[content_paths.index(content_path)] = network.losses(style,result)
        
    else:  # process one content and one style
        for style_path in style_paths:
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
            
            result = output[:style.shape[0],:style.shape[1],:style.shape[2],:style.shape[3]]
            content_loss[content_paths.index(content_path)]\
            [style_paths.index(style_path)],\
            style_loss[content_paths.index(content_path)]\
            [style_paths.index(style_path)] = network.losses(style,result)

print("Content loss:%.3f" %content_loss.mean())
message = ""
for i in range(style_loss.shape[2]):
    message += " level {:d}: {:.3f}".format(i+1, style_loss.transpose(0,2)[i].mean())
print("Style loss", message)
