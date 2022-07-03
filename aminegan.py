# Define Model
import torch.nn as nn

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels,stride=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3,stride=stride, 
                               groups=in_channels, bias=bias, padding=1)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 
                               kernel_size=1,stride=1, bias=bias)
        #self.apply(weights_init)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,  kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding)
        self.inst_norm = nn.InstanceNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU()
        #self.apply(weights_init)

    def forward(self, x):
        #_, c, h, w = x.size()
        out = self.conv(x)
        out = self.inst_norm(out)
        out = self.lrelu(out)
        return out 

class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels,stride=1):
        super(DSConv, self).__init__()
        self.sep_conv = SeparableConv2d(in_channels,out_channels,stride=stride)
        self.inst_norm = nn.InstanceNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU()
        #self.conv_block = ConvBlock(in_channels,out_channels,kernel_size=1,stride=stride)

    def forward(self, x):
        #_, _, h, w = x.size()
        out = self.sep_conv(x)
        out = self.inst_norm(out)
        out = self.lrelu(out)
        #out = self.conv_block(out)
        return out  

class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownConv, self).__init__()
        self.dsconv1 = DSConv(in_channels,out_channels,stride=2)
        #self.resize = F.interpolate(x, size=(200, 200), mode='bilinear')
        self.dsconv2 = DSConv(in_channels,out_channels,stride=1)       

    def forward(self, x):
        _, _, h, w = x.size()
        out1 = self.dsconv1(x)
        #size=(h//2, w//2)
        resize = F.interpolate(x,size=(h//2, w//2), mode='bilinear') #,scale_factor=0.5
        out2 = self.dsconv2(resize)
        final = out1 + out2
        return final

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.dsconv = DSConv(in_channels, out_channels,stride=1)       

    def forward(self, x):
        _, _, h, w = x.size()
        resize = F.interpolate(x, size=(2*h, 2*w), mode='bilinear') #,scale_factor=2.0
        out = self.dsconv(resize)
        return out 

class IRB_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(IRB_Block, self).__init__()
        self.convblock = ConvBlock(in_channels,out_channels,kernel_size=1,stride=1,padding=0)
        self.depthwise = nn.Conv2d(out_channels, out_channels, kernel_size=3,stride=1,groups=out_channels, padding=1)
        #SeparableConv2d(out_channels,out_channels,stride=1) 
        self.inst_norm = nn.InstanceNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU()
        self.pointwize = nn.Conv2d(out_channels,in_channels,kernel_size=1,stride=1)
        #self.apply(weights_init)

    def forward(self, x):
        #bz, c, h, w = x.size()
        out = self.convblock(x)
        out = self.depthwise(out)
        out = self.inst_norm(out)
        out = self.lrelu(out)
        out = self.pointwize(out)
        out = self.inst_norm(out)
        final = out + x
        return final                       

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,32,3,1),
            nn.LeakyReLU(),

            nn.Conv2d(32,64,3,2),
            nn.LeakyReLU(),

            nn.Conv2d(64,128,3,1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(),

            nn.Conv2d(128,128,3,2),
            nn.LeakyReLU(),

            nn.Conv2d(128,256,3,1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(),

            nn.Conv2d(256,256,3,1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(),

            nn.Conv2d(256,1,3,1),
        )

    def forward(self, input):
        return self.model(input)

'''class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
                    
    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x'''
class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.model = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 128),
            DownConv(128,128),
            ConvBlock(128,128),
            DSConv(128,128),
            DownConv(128,256),
            ConvBlock(256,256),
            
            IRB_Block(256,512),
            IRB_Block(256,512),
            IRB_Block(256,512),
            IRB_Block(256,512),
            IRB_Block(256,512),
            IRB_Block(256,512),
            IRB_Block(256,512),
            IRB_Block(256,512),

            ConvBlock(256,128),
            UpConv(128,128),
            DSConv(128,128),
            ConvBlock(128,128),

            UpConv(128,128),
            ConvBlock(128,64),
            ConvBlock(64,64),
            nn.Conv2d(64,3,1,1,padding=0),
            nn.Tanh(),
            
        )

    def forward(self, input):
        return self.model(input)

