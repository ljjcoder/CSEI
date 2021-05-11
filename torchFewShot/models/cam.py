from __future__ import absolute_import
from __future__ import division

import torch
import math
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    """Basic convolutional block:
    convolution + batch normalization.

    Args (following http://pytorch.org/docs/master/nn.html#torch.nn.Conv2d):
    - in_c (int): number of input channels.
    - out_c (int): number of output channels.
    - k (int or tuple): kernel size.
    - s (int or tuple): stride.
    - p (int or tuple): padding.
    """
    def __init__(self, in_c, out_c, k, s=1, p=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn(self.conv(x))


class CAM(nn.Module):
    def __init__(self):
        super(CAM, self).__init__()
        self.conv1 = ConvBlock(36, 6, 1)
        self.conv2 = nn.Conv2d(6, 36, 1, stride=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def get_attention(self, a):
        input_a = a#[4,5,30,36,36]
        #print(torch.mean(input_a[:,:,:,0,0], -1).shape)
        a = a.mean(3) #[4,5,30,36]
        #print(a.shape)
        #exit(0)
        a = a.transpose(1, 3) #[4,36,30,5]
        a = F.relu(self.conv1(a))#[4,6,30,5]
        a = self.conv2(a) #[4,36,30,5]
        a = a.transpose(1, 3)#[4,5,30,36]
        a = a.unsqueeze(3) #[4,5,30,1,36]
        
        a = torch.mean(input_a * a, -1)#[4,5,30,36] 
        a = F.softmax(a / 0.025, dim=-1) + 1
        return a 

    def forward(self, f1, f2,test=False):
        b, n1, c, h, w = f1.size()
        n2 = f2.size(1)

        f1 = f1.view(b, n1, c, -1) #[4,5,512,36]
        f2 = f2.view(b, n2, c, -1) #[4,30,512,36]
        #print(f1.shape,f2.shape)
        #exit(0)
        f1_norm = F.normalize(f1, p=2, dim=2, eps=1e-12)
        f2_norm = F.normalize(f2, p=2, dim=2, eps=1e-12)
        
        f1_norm = f1_norm.transpose(2, 3).unsqueeze(2)#[4,5,1,36,512]
        
        f2_norm = f2_norm.unsqueeze(1)#[4,1,30,512,36]
        #print(f1_norm.shape,f2_norm.shape)
        #exit(0)
        a1 = torch.matmul(f1_norm, f2_norm) #[4,5,30,36,36]
        #print('The shape of a1 before get_attention: ', a1.shape,'located in cam.py at 72')
        #exit(0)
        a2 = a1.transpose(3, 4)  #[4,5,30,36,36]

        a1 = self.get_attention(a1)#[4,5,30,36]
        a2 = self.get_attention(a2)#[4,5,30,36] 
        #print('The shape of a1 after get_attention: ', a1.shape,'located in cam.py at 72')
        #print('The shape of a2 after get_attention: ', a2.shape,'located in cam.py at 72')
        #print(f1.unsqueeze(2).shape,a1.unsqueeze(3).shape)#[4,5,1,512,36],#[4,5,30,1,36]
        #exit(0)
        f1 = f1.unsqueeze(2) * a1.unsqueeze(3)
        f1 = f1.view(b, n1, n2, c, h, w)#[4,5,30,512,6,6]
        f2 = f2.unsqueeze(1) * a2.unsqueeze(3)
        f2 = f2.view(b, n1, n2, c, h, w)#[4,5,30,512,6,6]
        #print(f1.shape,f2.shape,'located in cam.py at 88')
        #exit(0)
        if test:
            return f1.transpose(1, 2), f2.transpose(1, 2),a1.view(b, n1, n2, h, w),a2.view(b, n1, n2, h, w)
        else:
            return f1.transpose(1, 2), f2.transpose(1, 2)
class CAM_similarity(nn.Module):
    def __init__(self):
        super(CAM_similarity, self).__init__()
        self.conv1 = ConvBlock(18468, 6, 1)
        self.conv2 = nn.Conv2d(6, 36, 1, stride=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def get_attention(self, a,features):
        input_a = a#[4,5,30,36,36]
        #print(torch.mean(input_a[:,:,:,0,0], -1).shape)
        a = a.mean(3) #[4,5,30,36]
        #print(a.shape,'located in cam at 109 in CAM_similarity')
        #exit(0)
        b,train_n,test_n,c,sptial=features.size()
        a=a.unsqueeze(3)
        a=torch.cat([a,features],3)
        a=a.view(b,train_n,test_n,-1)#[4,5,30,512,6,6]
        #print(a.shape)
        #exit(0)
        a = a.transpose(1, 3) #[4,36,30,5]
        a = F.relu(self.conv1(a))#[4,6,30,5]
        a = self.conv2(a) #[4,36,30,5]
        a = a.transpose(1, 3)#[4,5,30,36]
        a = a.unsqueeze(3) #[4,5,30,1,36]
        
        a = torch.mean(input_a * a, -1)#[4,5,30,36] 
        a = F.softmax(a / 0.025, dim=-1) + 1
        return a 

    def forward(self, f1, f2,test=False):
        b, n1, c, h, w = f1.size()
        n2 = f2.size(1)

        f1 = f1.view(b, n1, c, -1) #[4,5,512,36]
        f2 = f2.view(b, n2, c, -1)
        #print(f1.shape,f2.shape)
        #exit(0)
        f1_norm = F.normalize(f1, p=2, dim=2, eps=1e-12)#[4,5,512,36]
        f2_norm = F.normalize(f2, p=2, dim=2, eps=1e-12)#[4,30,512,36]
        #print(f1_norm.shape,f2_norm.shape)
        #added by ljj
        f1_expand=f1_norm.unsqueeze(2).expand(f1_norm.shape[0],f1_norm.shape[1],f2_norm.shape[1],f1_norm.shape[2],f1_norm.shape[3])
        f2_expand=f2_norm.unsqueeze(1).expand(f1_norm.shape[0],f1_norm.shape[1],f2_norm.shape[1],f1_norm.shape[2],f1_norm.shape[3])   
        #print(f1_norm.shape,f2_norm.shape)
        #for i in range(30):
            #print((f1_expand[:,:,i,:,:]-f1_norm).abs().sum())
        #for i in range(5):
           # print((f2_expand[:,i,:,:,:]-f2_norm).abs().sum())            
        #exit(0)
        #print(f1_expand.shape,f2_expand.shape)
        f1_f2=f2_expand#-f2_expand
        f2_f1=f1_expand#-f1_expand
        #added end
        #print(f1_f2.shape)
        #exit(0)
        f1_norm = f1_norm.transpose(2, 3).unsqueeze(2) #
        
        f2_norm = f2_norm.unsqueeze(1)
        #print(f1_norm.shape,f2_norm.shape)
        #exit(0)
        a1 = torch.matmul(f1_norm, f2_norm) #[4,5,30,36,36]
        #print('The shape of a1 before get_attention: ', a1.shape,'located in cam.py at 72')
        #exit(0)
        a2 = a1.transpose(3, 4)  #[4,5,30,36,36]

        a1 = self.get_attention(a1,f1_f2)#[4,5,30,36]
        a2 = self.get_attention(a2,f2_f1)#[4,5,30,36] 
        #print('The shape of a1 after get_attention: ', a1.shape,'located in cam.py at 72')
        #print('The shape of a2 after get_attention: ', a2.shape,'located in cam.py at 72')
        #print(f1.unsqueeze(2).shape,a1.unsqueeze(3).shape)#[4,5,1,512,36],#[4,5,30,1,36]
        #exit(0)
        f1 = f1.unsqueeze(2) * a1.unsqueeze(3)
        f1 = f1.view(b, n1, n2, c, h, w)#[4,5,30,512,6,6]
        f2 = f2.unsqueeze(1) * a2.unsqueeze(3)
        f2 = f2.view(b, n1, n2, c, h, w)#[4,5,30,512,6,6]
        #print(f1.shape,f2.shape,'located in cam.py at 88')
        #exit(0)
        if test:
            return f1.transpose(1, 2), f2.transpose(1, 2),a1.view(b, n1, n2, h, w),a2.view(b, n1, n2, h, w)
        else:
            return f1.transpose(1, 2), f2.transpose(1, 2)