import torch.nn as nn
import torchvision

from function import adaptive_instance_normalization as adain
from function import calc_mean_std

class Identity(nn.BatchNorm2d):
    def __init__(self, inplanes):
        super(Identity, self).__init__(inplanes)
    
    def forward(self, x):
        return x

def conv3x3_dec(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    if in_planes != out_planes:
        return nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                             nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, 
                                       padding=1, bias=False))
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        #out += identity
        out = self.relu(out)

        return out

    
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, torchvision.models.Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
class BasicBlock_dec(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock_dec, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
    # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.bn2 = norm_layer(inplanes)
        self.conv2 = conv3x3_dec(inplanes, planes)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = norm_layer(planes)        
        self.conv1 = conv3x3_dec(planes, planes, stride)
        self.downsample = downsample

        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.bn2(x)
        out = self.conv2(out)

        out = self.relu(out)
        out = self.bn1(out)
        out = self.conv1(out)
      
        if self.downsample is not None:
            identity = self.downsample(x)

        #out += identity
        out = self.relu(out)

        return out
    
class ResDec(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, norm_layer=None):
        super(ResDec, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.inplanes = 512      
        self.layer4 = self._make_layer(block, 512, layers[3], norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1, norm_layer=norm_layer)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, norm_layer=norm_layer)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = norm_layer(64)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3,
                               bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)

        x = self.up2(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.up1(x)
        x = self.conv1(x)

        return x

res_decoder = ResDec(BasicBlock_dec, [2, 2, 2, 2], norm_layer=Identity)
res = ResNet(BasicBlock, [2, 2, 2, 2], norm_layer=Identity)
"""nn.Sequential(
    nn.BatchNorm2d(512),
    nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
    nn.ReLU(),
    nn.BatchNorm2d(512),
    nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
    # block 4-2
    nn.BatchNorm2d(512),
    nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
    nn.ReLU(),
    nn.BatchNorm2d(512),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
    # block 4-1
    nn.BatchNorm2d(256), # 7
    nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
    nn.ReLU(),
    nn.BatchNorm2d(256), 
    nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
    # block 3-2
    nn.BatchNorm2d(256), 
    nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
    nn.ReLU(),
    nn.BatchNorm2d(256), 
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
    # block 3-1    
    nn.BatchNorm2d(128), # 6 
    nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
    nn.ReLU(),
    nn.BatchNorm2d(128), 
    nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
    # block 2-2
    nn.BatchNorm2d(128), 
    nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
    nn.ReLU(),    
    nn.BatchNorm2d(128), 
    nn.Upsample(scale_factor=1, mode='nearest'),
    nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
    # block 2-1  
    nn.BatchNorm2d(64),  #5
    nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
    nn.ReLU(),
    nn.BatchNorm2d(64), 
    nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
    # block 1-2
    nn.BatchNorm2d(64), 
    nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
    nn.ReLU(),
    nn.BatchNorm2d(64), 
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
    # block 1-1
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReLU(),
    nn.BatchNorm2d(64), 
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.Conv2d(64, 3, kernel_size=7, padding=3, bias=False),
)

resnet18 = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
    nn.BatchNorm2d(64),    
    nn.ReLU(), # relu0-1
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    # block 1-1
    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(64),    
    nn.ReLU(), # relu1-1   
    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(64),   
    # block 1-2
    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(64),    
    nn.ReLU(), # relu1-2  
    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(64),
    # block 2-1 
    nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),    
    nn.ReLU(), # relu2-1 
    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(128),   
    # block 2-2
    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(128),    
    nn.ReLU(), # relu2-2  
    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(128),
    # block 3-1 
    nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),    
    nn.ReLU(), # relu3-1   
    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(256),   
    # block 3-2
    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(256),    
    nn.ReLU(), # relu3-2  
    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(256), 
    # block 4-1 
    nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(512),    
    nn.ReLU(), # relu4-1   #last layer
    nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(512),   
    # block 4-2
    nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(512),    
    nn.ReLU(), # relu4-2  
    nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(512),
)"""

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)


class Net(nn.Module):
    def __init__(self, encoder, decoder, switch):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        if switch:
            #VGG
            self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
            self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
            self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
            self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
            self.decoder = decoder
        else: #resnet
            enc_layers_c_2=list(enc_layers[5][1].children())
            enc_layers_c_3=list(enc_layers[6][1].children())
            self.enc_1 = nn.Sequential(*enc_layers[:3])	
            self.enc_2 = nn.Sequential(*enc_layers[3:5], enc_layers[5][0], *enc_layers_c_2[:3])	
            self.enc_3 = nn.Sequential(*enc_layers_c_2[3:],enc_layers[6][0],*enc_layers_c_3[:3])
            self.enc_4 = nn.Sequential(*enc_layers_c_3[3:],*enc_layers[7:])
            dec_layers = list(decoder.children())
            dec_layers_c_0 = list(dec_layers[0][0].children())
            self.decoder = nn.Sequential(*dec_layers_c_0[2:],*dec_layers[1:])
        self.mse_loss = nn.MSELoss()

        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # extract relu4_1 from input image
    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def forward(self, content, style, alpha=1.0):
        assert 0 <= alpha <= 1
        style_feats = self.encode_with_intermediate(style)
        content_feat = self.encode(content)
        t = adain(content_feat, style_feats[-1])
        t = alpha * t + (1 - alpha) * content_feat
        
        g_t = self.decoder(t)
        g_t_feats = self.encode_with_intermediate(g_t)
        
        loss_c = self.calc_content_loss(g_t_feats[-1], t)
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        for i in range(1, 4):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])
        return loss_c, loss_s
