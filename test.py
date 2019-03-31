import argparse
import os
import torch
import torch.nn as nn
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image

import net
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


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content', type=str,
                    help='File path to the content image')
parser.add_argument('--content_dir', type=str,
                    help='Directory path to a batch of content images')
parser.add_argument('--style', type=str,
                    help='File path to the style image, or multiple style \
                    images separated by commas if you want to do style \
                    interpolation or spatial control')
parser.add_argument('--style_dir', type=str,
                    help='Directory path to a batch of style images')
parser.add_argument('--enc', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--dec', type=str, default='models/decoder.pth')

# Additional options
parser.add_argument('--content_size', type=int, default=512,
                    help='New (minimum) size for the content image, \
                    keeping the original size if set to 0')
parser.add_argument('--style_size', type=int, default=512,
                    help='New (minimum) size for the style image, \
                    keeping the original size if set to 0')
parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')
parser.add_argument('--save_ext', default='.jpg',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default='output',
                    help='Directory to save the output image(s)')

# Advanced options
parser.add_argument('--preserve_color', action='store_true',
                    help='If specified, preserve color of the content image')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='The weight that controls the degree of \
                             stylization. Should be between 0 and 1')
parser.add_argument(
    '--style_interpolation_weights', type=str, default='',
    help='The weight for blending the style of multiple style images')

args = parser.parse_args()

do_interpolation = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Either --content or --contentDir should be given.
assert (args.content or args.content_dir)
# Either --style or --styleDir should be given.
assert (args.style or args.style_dir)

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
if args.enc == 'models/vgg_normalised.pth':
    # define decoder and encoder
    decoder = net.decoder
    encoder = net.vgg
    # load encoder weights and cut end of it
    encoder.load_state_dict(torch.load(args.enc))
    encoder = nn.Sequential(*list(encoder.children())[:31])
else:
    # define decoder and encoder
    decoder = net.res_decoder
    encoder = net.res
    # load encoder weights and cut end of it
    encoder.load_state_dict(torch.load(args.enc))
    last_block_child=list(list(encoder.children())[7][1].children())
    encoder = nn.Sequential(*list(encoder.children())[:7], list(encoder.children())[7][0], *last_block_child[:3])

decoder.eval()
encoder.eval()

decoder.load_state_dict(torch.load(args.dec))

encoder.to(device)
decoder.to(device)

content_tf = test_transform(args.content_size, args.crop)
style_tf = test_transform(args.style_size, args.crop)

for content_path in content_paths:
    if do_interpolation:  # one content image, N style image
        style = torch.stack([style_tf(Image.open(p)) for p in style_paths])
        content = content_tf(Image.open(content_path)) \
            .unsqueeze(0).expand_as(style)
        style = style.to(device)
        content = content.to(device)
        with torch.no_grad():
            output = style_transfer(encoder, decoder, content, style,
                                    args.alpha, interpolation_weights)
        output = output.cpu()
        output_name = '{:s}/{:s}_interpolation{:s}'.format(
            args.output, splitext(basename(content_path))[0], args.save_ext)
        save_image(output, output_name)

    else:  # process one content and one style
        for style_path in style_paths:
            content = content_tf(Image.open(content_path))
            style = style_tf(Image.open(style_path))
            if args.preserve_color:
                style = coral(style, content)
            style = style.to(device).unsqueeze(0)
            content = content.to(device).unsqueeze(0)
            with torch.no_grad():
                output = style_transfer(encoder, decoder, content, style,
                                        args.alpha)
            output = output.cpu()

            output_name = '{:s}/{:s}_stylized_{:s}{:s}'.format(
                args.output, splitext(basename(content_path))[0],
                splitext(basename(style_path))[0], args.save_ext
            )
            save_image(output, output_name)
