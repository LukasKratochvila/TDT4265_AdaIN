import torch.nn as nn
import torch
import torchvision

class Inception3(nn.Module):

    def __init__(self, num_classes=1000, aux_logits=True, transform_input=False):
        super(Inception3, self).__init__()
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = torchvision.models.inception.BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = torchvision.models.inception.BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = torchvision.models.inception.BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = torchvision.models.inception.BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = torchvision.models.inception.BasicConv2d(80, 192, kernel_size=3)
        self.Mixed_5b = torchvision.models.inception.InceptionA(192, pool_features=32)
        self.Mixed_5c = torchvision.models.inception.InceptionA(256, pool_features=64)
        self.Mixed_5d = torchvision.models.inception.InceptionA(288, pool_features=64)
        self.Mixed_6a = torchvision.models.inception.InceptionB(288)
        self.Mixed_6b = torchvision.models.inception.InceptionC(768, channels_7x7=128)
        self.Mixed_6c = torchvision.models.inception.InceptionC(768, channels_7x7=160)
        self.Mixed_6d = torchvision.models.inception.InceptionC(768, channels_7x7=160)
        self.Mixed_6e = torchvision.models.inception.InceptionC(768, channels_7x7=192)
        if aux_logits:
            self.AuxLogits = torchvision.models.inception.InceptionAux(768, num_classes)
        self.Mixed_7a = torchvision.models.inception.InceptionD(768)
        self.Mixed_7b = torchvision.models.inception.InceptionE(1280)
        self.Mixed_7c = torchvision.models.inception.InceptionE(2048)
        self.fc = nn.Linear(2048, num_classes)
    """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    """
    def forward(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        if self.training and self.aux_logits:
            aux = self.AuxLogits(x)
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        # N x 2048 x 1 x 1
        x = nn.functional.dropout(x, training=self.training)
        # N x 2048 x 1 x 1
        x = x.view(x.size(0), -1)
        # N x 2048
        x = self.fc(x)
        # N x 1000 (num_classes)
        if self.training and self.aux_logits:
            return x, aux
        return x

class Inception3_dec(nn.Module):

    def __init__(self, num_classes=1000):
        super(Inception3_dec, self).__init__()
        
        self.trans = nn.Conv2d(512,768,kernel_size=3,padding=1)
        self.Mixed_6b = InceptionC_dec(768, channels_7x7=128)
        self.Mixed_6a = InceptionB_dec(288)
        self.Mixed_5d = InceptionA_dec(288, 288, pool_features=64)
        self.Mixed_5c = InceptionA_dec(288, 256, pool_features=64)
        self.Mixed_5b = InceptionA_dec(256, 192, pool_features=32)
        
        #self.up2 = nn.ConvTranspose2d(192, 192, kernel_size=3, stride=2)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')       
        
        self.Conv2d_4a_3x3 = BasicConv2d(192, 80, kernel_size=3)
        self.Conv2d_3b_1x1 = BasicConv2d(80, 64, kernel_size=1)
        
        #self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2)
        self.up1 = nn.Upsample(scale_factor=1, mode='nearest')

        self.Conv2d_2b_3x3 = BasicConv2d(64, 32, kernel_size=3, padding=1)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_1a_3x3 = BasicConv2d(32, 3, kernel_size=3, stride=2)
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.trans(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6a(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 192 x 35 x 35
        x = self.up2(x)
        # N x 192 x 71 x 71
        x = self.Conv2d_4a_3x3(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 64 x 73 x 73
        x = self.up1(x)
        # N x 64 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_1a_3x3(x)
        # N x 3 x 299 x 299

        return x
 
class InceptionA_dec(nn.Module):

    def __init__(self, in_channels, out_channels, pool_features):
        super(InceptionA_dec, self).__init__()
        self.in_channels = in_channels
        self.branch1x1 = BasicConv2d(64, out_channels, kernel_size=1)

        self.branch5x5_2 = BasicConv2d(64, 48, kernel_size=5, padding=2)
        self.branch5x5_1 = BasicConv2d(48, out_channels, kernel_size=1)

        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_2 = BasicConv2d(96, 64, kernel_size=3, padding=1)
        self.branch3x3dbl_1 = BasicConv2d(64, out_channels, kernel_size=1)

        self.branch_pool = BasicConv2d(pool_features, out_channels, kernel_size=1)

    def forward(self, x):
        num =self.in_channels//32
        splitted = torch.chunk(x, num, dim=1)
        branch1x1, branch5x5, branch3x3dbl, branch_pool=torch.cat(splitted[0:2],dim=1),torch.cat(splitted[2:4],dim=1),torch.cat(splitted[4:7],dim=1),torch.cat(splitted[7:],dim=1)

        branch1x1 = self.branch1x1(branch1x1)
        
        branch5x5 = self.branch5x5_2(branch5x5)
        branch5x5 = self.branch5x5_1(branch5x5)
        
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_1(branch3x3dbl)
        
        #branch_pool = nn.functional.avg_pool2d(branch_pool, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        
        output = branch1x1 + branch5x5 + branch3x3dbl + branch_pool
        return output
        
        #branch1x1 = self.branch1x1(x)

        #branch5x5 = self.branch5x5_1(x)
        #branch5x5 = self.branch5x5_2(branch5x5)

        #branch3x3dbl = self.branch3x3dbl_1(x)
        #branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        #branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        #branch_pool = nn.functional.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        #branch_pool = self.branch_pool(branch_pool)

        #outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        #return torch.cat(outputs, 1)

class InceptionB_dec(nn.Module):

    def __init__(self, in_channels):
        super(InceptionB_dec, self).__init__()
        self.branch3x3 = BasicConv2d(384, in_channels, kernel_size=3, stride=2)

        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)
        self.branch3x3dbl_2 = BasicConv2d(96, 64, kernel_size=3, padding=1)
        self.branch3x3dbl_1 = BasicConv2d(64, in_channels, kernel_size=1)
        
        self.branch_pool = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                         nn.Conv2d(288, in_channels, kernel_size=1, bias=False))

    def forward(self, x):
        splitted = torch.chunk(x, 8, dim=1)
        branch3x3, branch3x3dbl, branch_pool=torch.cat(splitted[0:4],dim=1),torch.cat(splitted[4:5],dim=1),torch.cat(splitted[5:],dim=1)
         
        branch3x3 = self.branch3x3(branch3x3)
        
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_1(branch3x3dbl)
        
        branch_pool = self.branch_pool(branch_pool)
        
        output = branch3x3 + branch3x3dbl + branch_pool
        return output
        #branch3x3 = self.branch3x3(x)

        #branch3x3dbl = self.branch3x3dbl_1(x)
        #branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        #branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        #branch_pool = nn.functional.max_pool2d(x, kernel_size=3, stride=2)

        #outputs = [branch3x3, branch3x3dbl, branch_pool]
        #return torch.cat(outputs, 1)

class InceptionC_dec(nn.Module):

    def __init__(self, in_channels, channels_7x7):
        super(InceptionC_dec, self).__init__()
        self.branch1x1 = BasicConv2d(192, in_channels, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_3 = BasicConv2d(192, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_1 = BasicConv2d(c7, in_channels, kernel_size=1)

        self.branch7x7dbl_5 = BasicConv2d(192, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_1 = BasicConv2d(c7, in_channels, kernel_size=1)

        self.branch_pool = BasicConv2d(192, in_channels, kernel_size=1)

    def forward(self, x):
        [branch1x1, branch7x7, branch7x7dbl, branch_pool] = torch.chunk(x, 4, dim=1)
        branch1x1 = self.branch1x1(branch1x1)
        
        branch7x7 = self.branch7x7_3(branch7x7)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_1(branch7x7)
        
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_1(branch7x7dbl)
        
        #branch_pool = nn.functional.avg_pool2d(branch_pool, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        
        output = branch1x1 + branch7x7 + branch7x7dbl + branch_pool
        return output

        #branch1x1 = self.branch1x1(x)

        #branch7x7 = self.branch7x7_1(x)
        #branch7x7 = self.branch7x7_2(branch7x7)
        #branch7x7 = self.branch7x7_3(branch7x7)

        #branch7x7dbl = self.branch7x7dbl_1(x)
        #branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        #branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        #branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        #branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        #branch_pool = nn.functional.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        #branch_pool = self.branch_pool(branch_pool)

        #outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        #return torch.cat(outputs, 1)

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, **kwargs):
        super(BasicConv2d, self).__init__()
        #self.bn = nn.BatchNorm2d(in_channels, eps=0.001)
        if stride != 1:
            self.conv = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(in_channels, out_channels, bias=False, stride=stride-1, kernel_size=kernel_size, padding=padding+1, **kwargs))
            #self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, stride=stride, kernel_size=kernel_size, padding=padding, **kwargs)
        else:
            if  padding == 0 and kernel_size > 1:
                self.conv = nn.Sequential(nn.Upsample(scale_factor=1, mode='nearest'), nn.Conv2d(in_channels, out_channels, bias=False, stride=stride, kernel_size=kernel_size, padding=kernel_size//2, **kwargs))
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, stride=stride, kernel_size=kernel_size, padding=padding, **kwargs)

    def forward(self, x):
        x = nn.functional.relu(x, inplace=True)
        #x = self.bn(x)
        x = self.conv(x)
        return x



    
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

        out += identity
        out = self.relu(out)

        return out
    
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


    
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
        self.up1 = nn.Upsample(scale_factor=1, mode='nearest')
        self.relu = nn.ReLU(inplace=True)
        #self.bn1 = norm_layer(64)
        self.conv1 = nn.Sequential(nn.Upsample(scale_factor=1, mode='nearest'),
                                   nn.Conv2d(64, 3, kernel_size=3, stride=1, 
                                        padding=1, bias=False))

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
                #norm_layer(self.inplanes),
                nn.Upsample(scale_factor=2, mode='nearest'),
                conv1x1(self.inplanes, planes * block.expansion, stride),                
                #nn.ConvTranspose2d(self.inplanes, planes * block.expansion, kernel_size=4, stride=2, 
                #                   padding=1, bias=False),
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

        x = self.up1(x)
        x = self.relu(x)
        #x = self.bn1(x)
        x = self.conv1(x)

        return x

class BasicBlock_dec(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock_dec, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
    # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        #self.bn2 = norm_layer(inplanes)
        self.conv2 = conv3x3_dec(inplanes, planes)
        self.relu = nn.ReLU(inplace=True)
        #self.bn1 = norm_layer(planes)        
        self.conv1 = conv3x3_dec(planes, planes, stride)
        self.downsample = downsample

        self.stride = stride

    def forward(self, x):
        identity = x

        #out = self.bn2(x)
        out = self.conv2(x)

        out = self.relu(out)
        #out = self.bn1(out)
        out = self.conv1(out)
      
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def conv3x3_dec(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    if in_planes != out_planes:
        return nn.Sequential(nn.Upsample(scale_factor=2,mode="nearest"),
                             nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, 
                                  padding=1, bias=False))
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class Identity(nn.BatchNorm2d):
    def __init__(self, inplanes):
        super(Identity, self).__init__(inplanes)
    
    def forward(self, x):
        return x



Res_dec=nn.Sequential(
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

Resnet = nn.Sequential(
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
)

# Default

Vgg19_dec = nn.Sequential(
    #nn.ReflectionPad2d((1, 1, 1, 1)),
    #nn.Conv2d(512, 512, (3, 3)),
    #nn.ReLU(),  # relu4-2
    #nn.ReflectionPad2d((1, 1, 1, 1)),
    #nn.Conv2d(512, 512, (3, 3)),
    #nn.ReLU(),  # relu4-3
    #nn.ReflectionPad2d((1, 1, 1, 1)),
    #nn.Conv2d(512, 512, (3, 3)),
    #nn.ReLU(),  # relu4-4
    #nn.Upsample(scale_factor=2, mode='nearest'),
    #nn.ReflectionPad2d((1, 1, 1, 1)),
    #nn.Conv2d(512, 512, (3, 3)),
    #nn.ReLU(),  # relu5-1
    #nn.ReflectionPad2d((1, 1, 1, 1)),
    #nn.Conv2d(512, 512, (3, 3)),
    #nn.ReLU(),  # relu5-2
    #nn.ReflectionPad2d((1, 1, 1, 1)),
    #nn.Conv2d(512, 512, (3, 3)),
    #nn.ReLU(),  # relu5-3
    #nn.ReflectionPad2d((1, 1, 1, 1)),
    #nn.Conv2d(512, 512, (3, 3)),
    #nn.ReLU(),  # relu5-4
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

Vgg19 = nn.Sequential(
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