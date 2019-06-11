import torch.nn as nn
import torch
import math
""""
Definition of models related to ResNet and.
Code mainly based on the Tensorflow implementation.
All bach normalization were taken away and the strides modified to avoid
downsampling. Upsampling layers added instead.

"""
########################################################################           
#                        ResNet18 decoder                              #
########################################################################
def resnet18_dec(decoder=None):
    model = ResNet_dec(BasicBlock,[2,2,2,2])
    if decoder != None:
        model.load_state_dict(torch.load(decoder))
    return model

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        #self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        #self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        #out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        #out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
        
class ResNet_dec(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 256
        super(ResNet_dec, self).__init__()
        self.conv1 = nn.Conv2d(512, 256, kernel_size=7, stride=1, padding=3,
                               bias=False)
        #self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.upSample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.layer1 = self._make_layer(block, 256, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1)
        self.upSample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.layer3 = self._make_layer(block, 64, layers[2], stride=1)
        self.layer4 = self._make_layer(block, 3, layers[3], stride=1)
        self.upSample3 = nn.Upsample(scale_factor=2, mode='nearest')
        #self.avgpool = nn.AvgPool2d(7, stride=1)
        #self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        #x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool(x)
        x = self.upSample1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.upSample2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.upSample3(x)

        #x = self.avgpool(x)
        #x = x.view(x.size(0), -1)
        #x = self.fc(x)

        return x

