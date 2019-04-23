import torch.nn as nn
import os

from function import adaptive_instance_normalization as adain
from function import calc_mean_std

class Net(nn.Module):
    def __init__(self, encoder, decoder):#, switch):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        #self.enc_5 = nn.Sequential(*enc_layers[31:40])  # relu4_1 -> relu5_1
        #self.enc_6 = nn.Sequential(*enc_layers[40:53])  # relu5_1 -> relu5_4
        
        self.num_enc = 4#6   
            
        self.decoder = decoder

        self.mse_loss = nn.MSELoss()
        
        # Name for printing
        self.model_names = ['enc_{:d}'.format(i + 1) for i in range(self.num_enc)]
        self.model_names.append('decoder')

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
    
    def print_networks(self, expr_dir, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        message = '\n'
        message += '---------- Networks initialized -------------\n'
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    message += str(net) + '\n'
                message += '[Network %s] Total number of parameters : %.3f M\n' % (name, num_params / 1e6)
        message += '-----------------------------------------------'
        print(message)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'a') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')