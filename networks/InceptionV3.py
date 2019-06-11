import torch.nn as nn
import torch
""""
Definition of models related to InceptionV3.
"""

def inception3_dec(decoder=None):
    model = Inception3_dec()
    if decoder != None:
        model.load_state_dict(torch.load(decoder))
    return model

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

