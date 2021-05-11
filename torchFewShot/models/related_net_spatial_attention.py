import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from cam import CAM
from channel_wise_attention import SELayer
#from fuse_net import resnet_fuse
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, kernel=3, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        if kernel == 1:
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        elif kernel == 3:
            self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if kernel == 1:
            self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        elif kernel == 3:
            self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, kernel=1, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class RelaNet(nn.Module):

    def __init__(self,input_channel, layer_channels, kernel=3):
        self.inplanes = 64
        self.kernel = kernel
        self.input_channel=input_channel
        self.layer_channels=layer_channels
        super(RelaNet, self).__init__()
        self.convs=[]
        for i in range(len(layer_channels)):
            if i==0:
                self.conv1=nn.Conv2d(self.input_channel, self.layer_channels[i], kernel_size=kernel, stride=1, padding=0, bias=False)
                self.bn1 = nn.BatchNorm2d(self.layer_channels[i])
            elif i==1:  
                self.conv2=nn.Conv2d(self.layer_channels[i-1], self.layer_channels[i], kernel_size=kernel, stride=1, padding=0, bias=False)
                self.bn2 = nn.BatchNorm2d(self.layer_channels[i])                
            else:
                self.conv3=nn.Conv2d(self.layer_channels[i-1], 128, kernel_size=kernel, stride=1, padding=0, bias=False)
                self.bn3 = nn.BatchNorm2d(128)
        for j in range(len(layer_channels)):
            if j==0:
                self.conv1_attention=nn.Conv2d(128, self.layer_channels[j], kernel_size=kernel, stride=1, padding=0, bias=False)
                self.bn1_attention = nn.BatchNorm2d(self.layer_channels[j])
            elif j==1:  
                self.conv2_attention=nn.Conv2d(self.layer_channels[j-1], self.layer_channels[j], kernel_size=kernel, stride=1, padding=0, bias=False)
                self.bn2_attention = nn.BatchNorm2d(self.layer_channels[j])                
            else:
                self.conv3_attention=nn.Conv2d(self.layer_channels[j-1],1, kernel_size=kernel, stride=1, padding=0, bias=False)
                #self.bn3 = nn.BatchNorm2d(self.layer_channels[i])     
        self.conv3_channel_attention=nn.Conv2d(128*5,128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3_channel_attention = nn.BatchNorm2d(128)        
        #self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.cam_support = CAM()
        self.channel_att=SELayer(128,4)
        #self.sigmoid=torch.Sigmoid()
    def forward(self, x,label_1shot):
        shape=x.shape
        #print(x.shape)
        #exit(0)        
        #ftrain = self.cam_support(x, x)
        #print(ftrain.shape)
        #exit(0)
        #print(x.shape,(shape[0],shape[1],1,shape[2],shape[3],shape[4]))
        #print(x.shape,(shape[0],1,shape[1],shape[2],shape[3],shape[4]))  
        #exit(0)        
        shape=x.shape
        #print(x.shape)
        #exit(0)
        right=x.reshape(shape[0],shape[1],1,shape[2],shape[3],shape[4]).repeat(1,1,shape[1],1,1,1)
        left=x.reshape(shape[0],1,shape[1],shape[2],shape[3],shape[4]).repeat(1,shape[1],1,1,1,1)  
        #concat_feature=torch.cat((left, right), 3).reshape( shape[0]* shape[1]*shape[1],2*shape[2],shape[3],shape[4] ) 
        #print(x.shape)
        #exit(0)
        concat_feature=(left-right).reshape( shape[0]* shape[1]*shape[1],shape[2],shape[3],shape[4] ) 
        #concat_feature1= concat_feature
        #print(concat_feature.shape)
        #exit(0)
        for i in range(3):
            if i==0:
                concat_feature = self.conv1(concat_feature)
                concat_feature= self.bn1(concat_feature)
                concat_feature = self.relu(concat_feature)
                #print(concat_feature.shape,'llllllllllllllllllll')
            elif i==1:
                #print(concat_feature.shape,'looooooooooooooo')            
                concat_feature = self.conv2(concat_feature)
                concat_feature= self.bn2(concat_feature)
                concat_feature = self.relu(concat_feature)
            else:
                #print(concat_feature.shape,'pppppppppppppp')            
                concat_feature = self.conv3(concat_feature)
                concat_feature= self.bn3(concat_feature) 
                concat_feature = self.relu(concat_feature)                
        #concat_feature=concat_feature.reshape( shape[0],shape[1],shape[1],shape[2],shape[3],shape[4] )
        concat_feature=concat_feature.reshape( shape[0],shape[1],shape[1],128,shape[3],shape[4] )
        concat_feature=concat_feature.mean(2).reshape( -1,128,shape[3],shape[4])
        # channel-wise attention
        concat_feature_gm=concat_feature.reshape(shape[0],shape[1]*128,shape[3],shape[4] )#.mean(1)
        concat_feature_gm= self.conv3_channel_attention( concat_feature_gm)
        concat_feature_gm=self.bn3_channel_attention(  concat_feature_gm)
        concat_feature_gm = self.relu(concat_feature_gm) 
        #print(concat_feature_gm.shape)
        #exit(0)        
        #print(concat_feature.shape)
        channel_attention=self.channel_att(concat_feature_gm).unsqueeze(1)
        #print(channel_attention.shape,x.shape)
        #exit(0)
        for i in range(3):
            if i==0:
                concat_feature = self.conv1_attention(concat_feature)
                concat_feature= self.bn1_attention(concat_feature)
                concat_feature = self.relu(concat_feature)
                #print(concat_feature.shape,'llllllllllllllllllll')
            elif i==1:
                #print(concat_feature.shape,'looooooooooooooo')            
                concat_feature = self.conv2_attention(concat_feature)
                concat_feature= self.bn2_attention(concat_feature)
                concat_feature = self.relu(concat_feature)
            else:
                #print(concat_feature.shape,'pppppppppppppp')            
                concat_feature = self.conv3_attention(concat_feature)
                #concat_feature= self.bn3_attention(concat_feature) 
                #concat_feature = self.relu(concat_feature)           
        #spatial=torch.sigmoid( concat_feature).reshape( shape[0],shape[1],1,shape[3],shape[4] )   
        spatial=torch.tanh( concat_feature).reshape( shape[0],shape[1],1,shape[3],shape[4] )   
        #print(spatial.shape,'pppp tanh')
        #print(spatial[:2])
        #exit(0)
        concat_feature=(0.5*spatial+1.5)*x*(1+channel_attention )
        #concat_feature=  ftrain      
        #print(concat_feature.shape)
        #exit(0)
        return  concat_feature,spatial,channel_attention


def fusenet():
    model = RelaNet(512,[128,128,512], kernel=1)
    return model
