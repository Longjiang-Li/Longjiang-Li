
import torch.nn as nn
from Nets import *

class CNN_LLJ(nn.Module):
    def __init__(self, in_channels=0, out_channels=0, midle_channels = 1000):
        super(CNN_LLJ, self).__init__()

        self.conv_met1 = DoubleConv(in_channels, midle_channels).double().cuda()
        self.conv_met2 = DoubleConv(midle_channels, midle_channels*2).double().cuda()
        #self.conv_met3 = DoubleConv(midle_channels*2, midle_channels*4).double().cuda()
        #self.conv_met4 = DoubleConv(midle_channels*4, midle_channels*4).double().cuda()
        self.output    = OutConv(midle_channels*2, out_channels).double().cuda()

    def forward(self, x):
        x1 = self.conv_met1(x)
        x1 = self.conv_met2(x1)
        #x1 = self.conv_met3(x1)
        #x1 = self.conv_met4(x1)
        x_out = self.output(x1)
        return x_out.squeeze()