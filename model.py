import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.utils import data
from collections import OrderedDict
from torch.nn.parameter import Parameter


class Residual_block(nn.Module):
    def __init__(self, nb_filts, first = False):
        super(Residual_block, self).__init__()
        self.first = first
        
        if not self.first:
            self.bn1 = nn.BatchNorm1d(num_features = nb_filts[0])
        self.lrelu = nn.LeakyReLU()
        self.lrelu_keras = nn.LeakyReLU(negative_slope=0.3)
        
        self.conv1 = nn.Conv1d(in_channels = nb_filts[0],
			out_channels = nb_filts[1],
			kernel_size = 3,
			padding = 1,
			stride = 1)
        self.bn2 = nn.BatchNorm1d(num_features = nb_filts[1])
        self.conv2 = nn.Conv1d(in_channels = nb_filts[1],
			out_channels = nb_filts[1],
			padding = 1,
			kernel_size = 3,
			stride = 1)
        
        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv1d(in_channels = nb_filts[0],
				out_channels = nb_filts[1],
				padding = 0,
				kernel_size = 1,
				stride = 1)
            
        else:
            self.downsample = False
        self.mp = nn.MaxPool1d(3)
        
    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.lrelu_keras(out)
        else:
            out = x
            
        out = self.conv1(x)
        out = self.bn2(out)
        out = self.lrelu_keras(out)
        out = self.conv2(out)
        
        if self.downsample:
            identity = self.conv_downsample(identity)
            
        out += identity
        out = self.mp(out)
        return out


class RawNet(nn.Module):
    def __init__(self, d_args, device):
        super(RawNet, self).__init__()
        self.first_conv = nn.Conv1d(
            in_channels = d_args['in_channels'], #1
			out_channels = d_args['filts'][0], #128
			kernel_size = d_args['first_conv'], #3
			stride = d_args['first_conv']) #3
        self.first_bn = nn.BatchNorm1d(
            num_features = d_args['filts'][0]) #128
        self.lrelu_keras = nn.LeakyReLU(
            negative_slope = 0.3)
        
        self.block0 = self._make_layer(
            nb_blocks = d_args['blocks'][0], #2
			nb_filts = d_args['filts'][1], #128
			first = True)
        self.block1 = self._make_layer(
            nb_blocks = d_args['blocks'][1], #4
			nb_filts = d_args['filts'][2]) #256
        
        self.bn_before_gru = nn.BatchNorm1d(
            num_features = d_args['filts'][2][-1]) #256
        self.gru = nn.GRU(
            input_size = d_args['filts'][2][-1], #256
			hidden_size = d_args['gru_node'], #1024
			num_layers = d_args['nb_gru_layer'], #1
			batch_first = True)
        self.fc1_gru = nn.Linear(
            in_features = d_args['gru_node'], #1024
			out_features = d_args['nb_fc_node']) #256
        self.fc2_gru = nn.Linear(
            in_features = d_args['nb_fc_node'], #256
			out_features = d_args['nb_classes'], #6112
			bias = True)
        
        
    def forward(self, x, y = 0, is_test=False, is_TS=False):
        x = self.first_conv(x)
        x = self.first_bn(x)
        x = self.lrelu_keras(x)
        
        x = self.block0(x)
        x = self.block1(x)
        
        x = self.bn_before_gru(x)
        x = self.lrelu_keras(x)
        #(batch, filt, time) >> (batch, time, filt)
        x = x.permute(0, 2, 1)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = x[:,-1,:]
        code = self.fc1_gru(x)
        if is_test: return code
        
        code_norm = code.norm(p=2,dim=1, keepdim=True) / 10.
        code = torch.div(code, code_norm)
        out = self.fc2_gru(code)
        if is_TS: return out, code
        else: return out


    def _make_layer(self, nb_blocks, nb_filts, first = False):
        layers = []
        #def __init__(self, nb_filts, first = False):
        for i in range(nb_blocks):
            first = first if i == 0 else False
            layers.append(Residual_block(nb_filts = nb_filts,
		        first = first))
            if i == 0: nb_filts[0] = nb_filts[1]
        
        return nn.Sequential(*layers)



class RawNetWithSA(nn.Module):
    def __init__(self, d_args, device):
        super(RawNetWithSA, self).__init__()         

        self.first_conv = nn.Conv1d(in_channels = d_args['in_channels'],
			out_channels = d_args['filts'][0],#128
			kernel_size = d_args['first_conv'],#3
			padding = 0,
			stride = d_args['first_conv'])
        self.first_bn = nn.BatchNorm1d(num_features = d_args['filts'][0])
        self.lrelu_keras = nn.LeakyReLU(negative_slope = 0.3)

        self.block0 = self._make_layer(
            nb_blocks = d_args['blocks'][0], #2
			nb_filts = d_args['filts'][1], #128
			first = True)
        self.block1 = self._make_layer(
            nb_blocks = d_args['blocks'][1], #4
			nb_filts = d_args['filts'][2]) #256

        self.bn_before_gru = nn.BatchNorm1d(num_features = d_args['filts'][2][-1])
        self.gru = nn.GRU(input_size = d_args['filts'][2][-1],
			hidden_size = d_args['gru_node'],
			num_layers = d_args['nb_gru_layer'],
			batch_first = True)
        
        self.fc_embed = nn.Linear(in_features = d_args['gru_node'],
			out_features = d_args['nb_fc_node'])
        self.output_1 = nn.Linear(in_features = d_args['nb_fc_node'],
			out_features = d_args['nb_classes'],
			bias = True)
        self.output_2 = nn.Linear(in_features = d_args['nb_fc_node'],
			out_features = d_args['nb_classes'],
			bias = True)


    def forward(self, x, y = 0, nb_ta_samp = 32076, is_test=False, is_TS=False):

        l_x = []
        nb_time = x.size(-1) 
        window_size = int(nb_ta_samp / 10)
        if nb_time > nb_ta_samp:
            step = nb_ta_samp - window_size
            iter = int( (nb_time - window_size) / step ) + 1
            for i in range(iter):
                if i == 0:
                    l_x.append(x[:, :, :nb_ta_samp])
                elif i < iter - 1:
                    l_x.append(x[:, :, i*step : i*step + nb_ta_samp])
                else:
                    l_x.append(x[:, :, -nb_ta_samp:])
        else:
            iter = 1
            l_x.append(x)
        
        for i in range(iter):
            x = l_x[i]
            x = self.first_conv(x)
            x = self.first_bn(x)
            x = self.lrelu_keras(x)

            x = self.block0(x)
            x = self.block1(x)

            g = self.bn_before_gru(x)
            g = self.lrelu_keras(g)
            g = g.permute(0, 2, 1)#(batch, filt, time) >> (batch, time, filt)
            self.gru.flatten_parameters()
            g, _ = self.gru(g)
            g = g[:,-1,:]
            
            code = self.fc_embed(g)
            if i == 0: s_code = code.view(x.size(0), -1, 1)
            else: s_code = torch.cat([s_code, code.view(x.size(0), -1, 1)], dim = 2)
            
            code_norm = code.norm(p=2,dim=1, keepdim=True) / 10.
            code = torch.div(code, code_norm)
            o = self.output_2(code)
            if i == 0: s_out = o.view(x.size(0), -1, 1)
            else: s_out = torch.cat([s_out, o.view(x.size(0), -1, 1)], dim = 2)

        code = torch.mean(s_code, dim = 2).view(x.size(0), -1)
        if is_test: return code
        
        code_norm = code.norm(p=2,dim=1, keepdim=True) / 10.
        code = torch.div(code, code_norm)
        out = self.output_1(code)
        
        if is_TS: return code, out 
        else: return out, s_out, code, s_code 
    

    def _make_layer(self, nb_blocks, nb_filts, first = False):
        layers = []
        #def __init__(self, nb_filts, first = False):
        for i in range(nb_blocks):
            first = first if i == 0 else False
            layers.append(Residual_block(nb_filts = nb_filts,
		        first = first))
            if i == 0: nb_filts[0] = nb_filts[1]
        
        return nn.Sequential(*layers)