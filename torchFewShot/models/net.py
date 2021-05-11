import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.append(r"./torchFewShot/models/") 

from resnet12 import resnet12
from cam import CAM,CAM_similarity


class Model(nn.Module):
    def __init__(self, scale_cls, num_classes=64):
        super(Model, self).__init__()
        self.scale_cls = scale_cls

        self.base = resnet12()
        self.cam = CAM()

        self.nFeat = self.base.nFeat
        self.clasifier = nn.Conv2d(self.nFeat, num_classes, kernel_size=1) 

    def test(self, ftrain, ftest):
        ftest = ftest.mean(4)
        ftest = ftest.mean(4)
        ftest = F.normalize(ftest, p=2, dim=ftest.dim()-1, eps=1e-12)
        ftrain = F.normalize(ftrain, p=2, dim=ftrain.dim()-1, eps=1e-12)
        scores = self.scale_cls * torch.sum(ftest * ftrain, dim=-1)
        return scores

    def forward(self, xtrain, xtest, ytrain, ytest,test_fg=False):
        batch_size, num_train = xtrain.size(0), xtrain.size(1)
        #print(xtrain.shape,xtest.shape)#[4,25,3,84,84],[4,30,3,84,84]
        #exit(0)
        num_test = xtest.size(1)
        K = ytrain.size(2)
        ytrain = ytrain.transpose(1, 2)#[4,5,25]
        #print(batch_size)
        xtrain = xtrain.view(-1, xtrain.size(2), xtrain.size(3), xtrain.size(4))#[100,3,84,84]
        xtest = xtest.view(-1, xtest.size(2), xtest.size(3), xtest.size(4))#[120,3,84,84]
        x = torch.cat((xtrain, xtest), 0)#(220,3,84,84)
        #print(x.shape,xtrain.shape,ytrain.shape,xtest.shape,num_train,'llll',)
        #exit(0)
        f = self.base(x)#[220,512,6,6]
        #print(f.shape,'located in net.py at 42')#[220,512,6,6]
        #exit(0)
        #print(ytrain[0,:,1:10])

        #exit(0)
        ftrain = f[:batch_size * num_train]
        ftrain = ftrain.view(batch_size, num_train, -1)#[4,25,18432] 
        #print(ftrain.shape)
        #exit(0)        
        ftrain = torch.bmm(ytrain, ftrain)#(4,5,18432),it is matrix multiply [4,5,25],[4,25,18432]
        #print(ytrain.sum(dim=2, keepdim=True).shape)
        #print(ftrain.div(ytrain.sum(dim=2, keepdim=True))
        #print(ytrain.sum(dim=2, keepdim=True).expand_as(ftrain)[0,:3,:10])
        #exit(0)
        ftrain = ftrain.div(ytrain.sum(dim=2, keepdim=True).expand_as(ftrain))# each row is the mean value of all support images of crospond class [4,5,18432]
        ftrain = ftrain.view(batch_size, -1, *f.size()[1:])#[4,5,512,6,6]
        #print(ftrain.shape)
        #exit(0)
        ftest = f[batch_size * num_train:]
        ftest = ftest.view(batch_size, num_test, *f.size()[1:])#[4,30,512,6,6] 
        #print(ftest.shape,ftrain.shape,'lllllllllllllll')
        #exit(0)  
        if not test_fg:
            ftrain, ftest = self.cam(ftrain, ftest,test_fg)##[4,30,5,512,6,6],[4,30,5,512,6,6]
        else:
            ftrain, ftest,a1,a2 = self.cam(ftrain, ftest,test_fg)
        ftrain = ftrain.mean(4)
        ftrain = ftrain.mean(4)#[4,30,5,512]

        if not self.training:
            if test_fg:
                return self.test(ftrain, ftest),a1,a2
            else:
                return self.test(ftrain, ftest)

        ftest_norm = F.normalize(ftest, p=2, dim=3, eps=1e-12)#[4,30,5,512,6,6]
        ftrain_norm = F.normalize(ftrain, p=2, dim=3, eps=1e-12)#[4,30,5,512]
        #print(ftest_norm.shape,ftrain_norm.shape,'located in net.py at 74')
        #exit(0)
        ftrain_norm = ftrain_norm.unsqueeze(4)
        ftrain_norm = ftrain_norm.unsqueeze(5)#[4,30,5,512,1,1]
        #print(ftest_norm.shape,self.scale_cls,K)
        #exit(0)
        cls_scores = self.scale_cls * torch.sum(ftest_norm * ftrain_norm, dim=3)#[4,30,5,6,6]
        #print(cls_scores.shape,'located in net.py at 79')
        #exit(0)        
        cls_scores = cls_scores.view(batch_size * num_test, *cls_scores.size()[2:])#[120,5,6,6]
        #print(cls_scores.shape,'located in net.py at 79')
        #exit(0)
        ftest = ftest.view(batch_size, num_test, K, -1)#[4,30,5,18432]
        ftest = ftest.transpose(2, 3) #[4,30,18432,5]
        ytest = ytest.unsqueeze(3)#[4,30,5,1] 
        #print(ytest.shape,ftest.shape)
        #exit(0)
        ftest = torch.matmul(ftest, ytest) 
        ftest = ftest.view(batch_size * num_test, -1, 6, 6)
        #print(ftest.shape)
        #exit(0)        
        ytest = self.clasifier(ftest)

        return ytest, cls_scores
        
class Model_mltizhixin(nn.Module):
    def __init__(self, scale_cls, num_classes=64):
        super(Model_mltizhixin, self).__init__()
        self.scale_cls = scale_cls

        self.base = resnet12()
        self.cam = CAM()

        self.nFeat = self.base.nFeat
        self.clasifier = nn.Conv2d(self.nFeat, num_classes, kernel_size=1) 

    def test(self, ftrain, ftest):
        ftest = ftest.mean(4)
        ftest = ftest.mean(4)
        ftest = F.normalize(ftest, p=2, dim=ftest.dim()-1, eps=1e-12)
        ftrain = F.normalize(ftrain, p=2, dim=ftrain.dim()-1, eps=1e-12)
        scores = self.scale_cls * torch.sum(ftest * ftrain, dim=-1)
        return scores

    def forward(self, xtrain, xtest, ytrain, ytest,test_fg=False):
        batch_size, num_train = xtrain.size(0), xtrain.size(1)
        #print(xtrain.shape,xtest.shape)#[4,25,3,84,84],[4,30,3,84,84]
        #exit(0)
        #cls_scores=0
        num_test = xtest.size(1)
        K = ytrain.size(2)
        ytest = ytest.unsqueeze(3)
        ytrain = ytrain.transpose(1, 2)#[4,5,25]
        #print(batch_size)
        xtrain = xtrain.view(-1, xtrain.size(2), xtrain.size(3), xtrain.size(4))#[100,3,84,84]
        xtest = xtest.view(-1, xtest.size(2), xtest.size(3), xtest.size(4))#[120,3,84,84]
        x = torch.cat((xtrain, xtest), 0)#(220,3,84,84)
        #print(x.shape,xtrain.shape,ytrain.shape,xtest.shape,num_train,'llll',)
        #exit(0)
        f = self.base(x)#[220,512,6,6]
        #print(f.shape,'located in net.py at 42')#[220,512,6,6]
        #exit(0)
        #print(ytrain[0,:,1:10])

        #exit(0)
        ftrain_all = f[:batch_size * num_train].view(batch_size, 5, -1,512,6,6)
        ftest_all = f[batch_size * num_train:]
        ftest_all = ftest_all.view(batch_size, num_test, *f.size()[1:])#[4,30,512,6,6]         
        #print(ftrain.shape)
        #exit(0)
        ytest_p=[]
        for i in range(5):
          ftrain = ftrain_all[:,:,i].view(-1,512,6,6)
          ftrain = ftrain.view(batch_size, 5, -1)#[4,25,18432] 
        #print(ftrain.shape)
        #exit(0)        
          ftrain = torch.bmm(ytrain, ftrain)#(4,5,18432),it is matrix multiply [4,5,25],[4,25,18432]
        #print(ytrain.sum(dim=2, keepdim=True).shape)
        #print(ftrain.div(ytrain.sum(dim=2, keepdim=True))
        #print(ytrain.sum(dim=2, keepdim=True).expand_as(ftrain)[0,:3,:10])
        #exit(0)
          ftrain = ftrain.div(ytrain.sum(dim=2, keepdim=True).expand_as(ftrain))# each row is the mean value of all support images of crospond class [4,5,18432]
          ftrain = ftrain.view(batch_size, -1, *f.size()[1:])#[4,5,512,6,6]
        #print(ftrain.shape)
        #exit(0)
          #ftest = f[batch_size * num_train:]
          #ftest = ftest.view(batch_size, num_test, *f.size()[1:])#[4,30,512,6,6] 
          ftest =ftest_all
        #print(ftest.shape,ftrain.shape,'lllllllllllllll')
        #exit(0)  
          if not test_fg:
              ftrain, ftest_att = self.cam(ftrain, ftest,test_fg)##[4,30,5,512,6,6],[4,30,5,512,6,6]
          else:
              ftrain, ftest_att ,a1,a2 = self.cam(ftrain, ftest,test_fg)
          ftrain = ftrain.mean(4)
          ftrain = ftrain.mean(4)#[4,30,5,512]

          if not self.training:
              if test_fg:
                  return self.test(ftrain, ftest_att),a1,a2
              else:
                  return self.test(ftrain, ftest_att)

          ftest_norm = F.normalize(ftest_att, p=2, dim=3, eps=1e-12)#[4,30,5,512,6,6]
          ftrain_norm = F.normalize(ftrain, p=2, dim=3, eps=1e-12)#[4,30,5,512]
        #print(ftest_norm.shape,ftrain_norm.shape,'located in net.py at 74')
        #exit(0)
          ftrain_norm = ftrain_norm.unsqueeze(4)
          ftrain_norm = ftrain_norm.unsqueeze(5)#[4,30,5,512,1,1]
        #print(ftest_norm.shape,self.scale_cls,K)
        #exit(0)
          if i==0:
              cls_scores =self.scale_cls * torch.sum(ftest_norm * ftrain_norm, dim=3)#[4,30,5,6,6]
        #print(cls_scores.shape,'located in net.py at 79')
        #exit(0)        
          #cls_scores = cls_scores.view(batch_size * num_test, *cls_scores.size()[2:])#[120,5,6,6]
        #print(cls_scores.shape,'located in net.py at 79')
        #exit(0)
          ftest_att = ftest_att.view(batch_size, num_test, K, -1)#[4,30,5,18432]
          ftest_att = ftest_att.transpose(2, 3) #[4,30,18432,5]
          #ytest = ytest.unsqueeze(3)#[4,30,5,1] 
          #print(ytest.shape,ftest.shape)
        #exit(0)
          
          ftest_att = torch.matmul(ftest_att, ytest) 
          ftest_att = ftest_att.view(batch_size * num_test, -1, 6, 6)
        #print(ftest.shape)
        #exit(0)        
          ytest_p.append( self.clasifier(ftest_att))
          #print(ytest_p.shape,cls_scores.shape,'llll')
          
          #exit(0)
        cls_scores = cls_scores.view(batch_size * num_test, *cls_scores.size()[2:])#[120,5,6,6]
        y_p = torch.stack(ytest_p, dim=0).view(5*120,64,6,6) 
        #print(y_p.shape)
        #exit(0)
        return y_p, cls_scores
                
class Model_tradi(nn.Module):
    def __init__(self, scale_cls, num_classes=64):
        super(Model_tradi, self).__init__()

        self.scale_cls = scale_cls

        self.base = resnet12()
        self.cam = CAM()

        self.nFeat = self.base.nFeat
        self.clasifier = nn.Conv2d(self.nFeat, num_classes, kernel_size=1) 
        #print(self.training)
        #exit(0)
    def test(self, ftrain, ftest):
        ftest = ftest.mean(4)
        ftest = ftest.mean(4)
        ftest = F.normalize(ftest, p=2, dim=ftest.dim()-1, eps=1e-12)
        ftrain = F.normalize(ftrain, p=2, dim=ftrain.dim()-1, eps=1e-12)
        scores = self.scale_cls * torch.sum(ftest * ftrain, dim=-1)
        return scores

    def forward(self, xtrain, xtest, ytrain, ytest,test_fg=False):
        batch_size, num_train = xtrain.size(0), xtrain.size(1)
        #print(xtrain.shape,xtest.shape)#[4,25,3,84,84],[4,30,3,84,84]
        #exit(0)
        num_test = xtest.size(1)
        K = ytrain.size(2)
        ytrain = ytrain.transpose(1, 2)#[4,5,25]
        #print(batch_size)
        xtrain = xtrain.view(-1, xtrain.size(2), xtrain.size(3), xtrain.size(4))#[100,3,84,84]
        
        xtest = xtest.view(-1, xtest.size(2), xtest.size(3), xtest.size(4))#[120,3,84,84]
        #x = torch.cat((xtrain, xtest), 0)#(220,3,84,84)
        #print(x.shape,xtrain.shape,xtest.shape,num_train)
        #exit(0)
        f = self.base(xtest)#[220,512,6,6]
        #print(f.shape,'located in net.py at 42')#[220,512,6,6]
        #exit(0)
        #print(ytrain[0,:,1:10])

        #exit(0)
        #ftrain = f[:batch_size * num_train]
        #ftrain = ftrain.view(batch_size, num_train, -1)#[4,25,18432] 
        #print(ftrain.shape)
        #exit(0)        
        #ftrain = torch.bmm(ytrain, ftrain)#(4,5,18432),it is matrix multiply [4,5,25],[4,25,18432]
        #print(ytrain.sum(dim=2, keepdim=True).shape)
        #print(ftrain.div(ytrain.sum(dim=2, keepdim=True))
        #print(ytrain.sum(dim=2, keepdim=True).expand_as(ftrain)[0,:3,:10])
        #exit(0)
        #ftrain = ftrain.div(ytrain.sum(dim=2, keepdim=True).expand_as(ftrain))# each row is the mean value of all support images of crospond class [4,5,18432]
        #ftrain = ftrain.view(batch_size, -1, *f.size()[1:])#[4,5,512,6,6]
        #print(ftrain.shape)
        #exit(0)
        ftest = f#[batch_size * num_train:]
        ftest = ftest.view(batch_size, num_test, *f.size()[1:])#[4,30,512,6,6] 
        #print(ftest.shape,ftrain.shape,'lllllllllllllll##########')
        #exit(0)  
        #if not test_fg:
            #ftrain, ftest = self.cam(ftrain, ftest,test_fg)##[4,30,5,512,6,6],[4,30,5,512,6,6]
        #else:
            #ftrain, ftest,a1,a2 = self.cam(ftrain, ftest,test_fg)
        #ftrain = ftrain.mean(4)
        #ftrain = ftrain.mean(4)#[4,30,5,512]
        #print(ftest.shape,ftrain.shape,self.training,'lllllllllllllll##########')
        #if not self.training:
            #if test_fg:
                #return self.test(ftrain, ftest),a1,a2
            #else:
                #return self.test(ftrain, ftest)
        #print(ftest.shape,ftrain.shape,'lllllllllllllll##########')
        #exit(0)
        #ftest_norm = F.normalize(ftest, p=2, dim=3, eps=1e-12)#[4,30,5,512,6,6]
        #ftrain_norm = F.normalize(ftrain, p=2, dim=3, eps=1e-12)#[4,30,5,512]
        #print(ftest_norm.shape,ftrain_norm.shape,'located in net.py at 74')
        #exit(0)
        #ftrain_norm = ftrain_norm.unsqueeze(4)
        #ftrain_norm = ftrain_norm.unsqueeze(5)#[4,30,5,512,1,1]
        #print(ftest_norm.shape,self.scale_cls,K)
        #exit(0)
        #cls_scores = self.scale_cls * torch.sum(ftest_norm * ftrain_norm, dim=3)#[4,30,5,6,6]
        #print(cls_scores.shape,'located in net.py at 79')
        #exit(0)        
        #cls_scores = cls_scores.view(batch_size * num_test, *cls_scores.size()[2:])#[120,5,6,6]
        #print(cls_scores.shape,'located in net.py at 79')
        #exit(0)
        #ftest = ftest.view(batch_size, num_test, K, -1)#[4,30,5,18432]
        #ftest = ftest.transpose(2, 3) #[4,30,18432,5]
        #ytest = ytest.unsqueeze(3)#[4,30,5,1] 
        #print(ytest.shape,ftest.shape)
        #exit(0)
        #ftest = torch.matmul(ftest, ytest) 
        ftest = ftest.view(batch_size * num_test, -1, 6, 6)
        feature=ftest        
        ftest = F.avg_pool2d(ftest, kernel_size=6, stride=1)
        #print(ftest.shape,'using avg_pool')
        #print(ftest.shape,'no using avg_pool')        
        #exit(0)        
        ytest = self.clasifier(ftest)
        #print(ytest)
        #exit(0)

        return ytest,feature#, cls_scores        