import torch.nn as nn
import torch

import net_def

from function import adaptive_instance_normalization as adain
from function import calc_mean_std

def vgg19(encoder, test, decoder=None):
    vgg19 = net_def.Vgg19
    # load encoder weights and cut end of it
    vgg19.load_state_dict(torch.load(encoder))
    vgg19 = nn.Sequential(*list(vgg19.children())[:31])
    vgg19_dec = net_def.Vgg19_dec
    
    if test:
        vgg19_dec.load_state_dict(torch.load(decoder))

    return vgg19, vgg19_dec

def resnet18(encoder, test, decoder=None):
    resnet18 = net_def.ResNet(net_def.BasicBlock,[2,2,2,2])#,norm_layer=net_def.Identity)
    resnet18_dec = net_def.ResDec(net_def.BasicBlock_dec,[2,2,2,2])#,norm_layer=net_def.Identity)
    # load encoder weights and cut end of it
    resnet18.load_state_dict(torch.load(encoder))
    #last_block_child=list(list(encoder.children())[7][1].children())
    #resnet18 = nn.Sequential(*list(encoder.children())[:7], list(encoder.children())[7][0], *last_block_child[:3])
    resnet18 = nn.Sequential(*list(resnet18.children())[:8])

    if test:
        resnet18_dec.load_state_dict(torch.load(decoder))
    
    return resnet18, resnet18_dec

def inception3(encoder, test, decoder=None):    
    inception3 = net_def.Inception3()
    inception3_dec = net_def.Inception3_dec()
    # load encoder weights and cut end of it
    inception3.load_state_dict(torch.load(encoder))
    inception3 = nn.Sequential(*list(inception3.children())[:3],
                               nn.MaxPool2d(kernel_size=3,stride=2),
                               *list(inception3.children())[3:4])#,
                               #nn.MaxPool2d(kernel_size=3,stride=2),
                               #*list(inception3.children())[5:6])
    inception3_dec = nn.Sequential(*list(inception3_dec.children())[7:])
    
    if test:
        inception3_dec.load_state_dict(torch.load(decoder))
    
    return inception3, inception3_dec

class Net(nn.Module):
    def __init__(self, encoder, decoder, switch):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        if switch == 0:
            #VGG
            self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
            self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
            self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
            self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
            
            self.decoder = decoder
            self.num_enc = 4 
        else:
            if switch == 1:
                #resnet
                """
                #enc_layers_c_0=list(enc_layers[4][0].children())
                #enc_layers_c_1=list(enc_layers[4][1].children()) 
                #self.enc_1 = nn.Sequential(*enc_layers[:3])
                #self.enc_2 = nn.Sequential(enc_layers[3],*enc_layers_c_0[:3])
                #self.enc_3 = nn.Sequential(*enc_layers_c_0[3:],*enc_layers_c_1[:3])
                #enc_layers_c_0=list(enc_layers[5][0].children())
                #self.enc_4 = nn.Sequential(*enc_layers_c_1[3:],*enc_layers_c_0[:3])
                #enc_layers_c_1=list(enc_layers[5][1].children())
                #self.enc_5 = nn.Sequential(*enc_layers_c_0[3:-1],*enc_layers_c_1[:3])
                #enc_layers_c_0=list(enc_layers[6][0].children())
                #self.enc_6 = nn.Sequential(*enc_layers_c_1[3:],*enc_layers_c_0[:3])
                #enc_layers_c_1=list(enc_layers[6][1].children())
                #self.enc_7 = nn.Sequential(*enc_layers_c_0[3:-1],*enc_layers_c_1[:3])
                #enc_layers_c_0=list(enc_layers[7].children())
                #self.enc_8 = nn.Sequential(*enc_layers_c_1[3:],*enc_layers_c_0[:3])
                #self.enc_9 = nn.Sequential(*enc_layers_c_0[3:-1],*enc_layers[8:])
                    
                #enc_layers_c_2=list(enc_layers[5][1].children())
                #enc_layers_c_3=list(enc_layers[6][1].children())
                #self.enc_1 = nn.Sequential(*enc_layers[:3])	
                #self.enc_2 = nn.Sequential(*enc_layers[3:5], enc_layers[5][0], *enc_layers_c_2[:3])	
                #self.enc_3 = nn.Sequential(*enc_layers_c_2[3:],enc_layers[6][0],*enc_layers_c_3[:3])
                #self.enc_4 = nn.Sequential(*enc_layers_c_3[3:],*enc_layers[7:])
                """
                self.enc_1 = nn.Sequential(*enc_layers[:3])
                self.enc_2 = nn.Sequential(enc_layers[3], enc_layers[4][0])
                self.enc_3 = nn.Sequential(enc_layers[4][1])
                self.enc_4 = nn.Sequential(enc_layers[5][0])
                self.enc_5 = nn.Sequential(enc_layers[5][1])
                self.enc_6 = nn.Sequential(enc_layers[6][0])
                self.enc_7 = nn.Sequential(enc_layers[6][1])
                self.enc_8 = nn.Sequential(enc_layers[7])
                self.enc_9 = nn.Sequential(*enc_layers[8:])
            
                #dec_layers = list(decoder.children())
                #dec_layers_c_0 = list(dec_layers[0][0].children())
                self.decoder = decoder#nn.Sequential(*dec_layers_c_0[2:],*dec_layers[1:])
                self.num_enc = 9
            else:
                # Inception
                self.enc_1 = nn.Sequential(enc_layers[0])
                self.enc_2 = nn.Sequential(enc_layers[1])
                self.enc_3 = nn.Sequential(enc_layers[2])
                self.enc_4 = nn.Sequential(*enc_layers[3:5])
                #self.enc_5 = nn.Sequential(enc_layers[5])
                #self.enc_6 = nn.Sequential(enc_layers[6:8])
                #self.enc_7 = nn.Sequential(enc_layers[8])
                #self.enc_8 = nn.Sequential(enc_layers[10])
                #self.enc_9 = nn.Sequential(enc_layers[8])
                #self.enc_10 = nn.Sequential(enc_layers[9])
                self.num_enc = 4
                self.decoder = decoder
                        
        self.mse_loss = nn.MSELoss()

        # fix the encoder
        for i in range(self.num_enc):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(self.num_enc):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # extract relu4_1 from input image
    def encode(self, input):
        for i in range(self.num_enc):
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
        #print(g_t_feats[-1].shape, t.shape)
        loss_c = self.calc_content_loss(g_t_feats[-1], t)
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        for i in range(1, len(style_feats)):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])
        return loss_c, loss_s
