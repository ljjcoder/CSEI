import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.append(r"./torchFewShot/models/") 

from resnet12 import resnet12
#from related_net import fusenet
from related_net_spatial_attention import fusenet
from cam import CAM,CAM_similarity
from torchFewShot.utils.torchtools import one_hot_36
#from contrast_loss import TripletLoss


class Model(nn.Module):
    def __init__(self, scale_cls,only_CSEI, num_classes=64):
        super(Model, self).__init__()
        self.scale_cls = scale_cls

        self.base = resnet12()
        self.only_CSEI=only_CSEI
        if not self.only_CSEI:
            self.fusenet=fusenet()
        self.cam = CAM()

        self.nFeat = self.base.nFeat
        self.clasifier = nn.Conv2d(self.nFeat, num_classes, kernel_size=1) 
        #self.clasifier1 = nn.Conv2d(self.nFeat, num_classes, kernel_size=1) 
        #self.contrastLoss=TripletLoss()
        
    def test_ori(self, ftrain, ftest):
        ftest = ftest.mean(4)
        ftest = ftest.mean(4)
        ftest = F.normalize(ftest, p=2, dim=ftest.dim()-1, eps=1e-12)
        ftrain = F.normalize(ftrain, p=2, dim=ftrain.dim()-1, eps=1e-12)
        scores = self.scale_cls * torch.sum(ftest * ftrain, dim=-1)
        print(ftest.shape,ftrain.shape)
        exit(0)        
        return scores
        
    def test(self, ftrain, ftest):
        ftest_mean = ftest.mean(4)
        ftest_mean = ftest_mean.mean(4).unsqueeze(3)
        ftrain_mean = ftrain.mean(4)
        ftrain_mean = ftrain_mean.mean(4).unsqueeze(3)        
        ftest=ftest.view(4, 75, 5, 512,-1)
        ftrain=ftrain.view(4, 75, 5, 512,-1)     
        ftest=ftest.transpose(3, 4)
        ftrain=ftrain.transpose(3, 4)         
        ftest = F.normalize(ftest, p=2, dim=ftest.dim()-1, eps=1e-12)
        ftrain = F.normalize(ftrain, p=2, dim=ftrain.dim()-1, eps=1e-12)
        ftest_mean = F.normalize(ftest_mean, p=2, dim=ftest_mean.dim()-1, eps=1e-12)
        ftrain_mean = F.normalize(ftrain_mean, p=2, dim=ftrain_mean.dim()-1, eps=1e-12)        
        #print(ftest_mean.shape,ftrain_mean.shape)#[4,75,5,1,512]
        #print(ftest.shape,ftrain.shape)#[4, 75, 5, 36, 512]       
        #exit(0)
        #print(ftest.shape,ftrain.shape,ftest_mean.shape,ftrain_mean.shape)        
        scores = self.scale_cls * torch.sum((ftest_mean * ftrain_mean).squeeze(), dim=-1)
        ftrain_scores = self.scale_cls * torch.sum(ftest_mean * ftrain, dim=-1)        
        ftest_scores = self.scale_cls * torch.sum(ftest * ftrain_mean, dim=-1)
        #print(ftest.shape,ftrain.shape)
        #print(ftrain_scores.shape,ftest_scores.shape)
        
        _,train_ind=torch.max(ftrain_scores,3)
        _,test_ind=torch.max(ftest_scores,3)
        #print(train_ind)
        #exit(0)        
        train_ind=train_ind.view(-1)
        test_ind=test_ind.view(-1)
        train_one_hot=one_hot_36(train_ind).view(4,75,5,-1).unsqueeze(4)  
        test_one_hot=one_hot_36(test_ind).view(4,75,5,-1).unsqueeze(4)  
        scores_final = self.scale_cls * torch.sum(((ftest*test_one_hot.cuda()).sum(3) * ((ftrain* train_one_hot.cuda()).sum(3))), dim=-1)        
        #print( test_one_hot.shape,train_one_hot.shape) #[4, 75, 5, 36]    
        #exit(0)        
        return scores,scores_final
        
    def test_topK(self, ftrain, ftest,K):
        shape_train=ftrain.shape
        #print(ftrain.shape,ftest.shape)
        #exit(0)
        ftest_mean = ftest.mean(4)
        ftest_mean = ftest_mean.mean(4).unsqueeze(3)
        ftrain_mean = ftrain.mean(4)
        ftrain_mean = ftrain_mean.mean(4).unsqueeze(3)  
        #print(ftrain_mean.shape)
        ftest=ftest.view(shape_train[0], 75, 5, 512,-1)
        ftrain=ftrain.view(shape_train[0], 75, 5, 512,-1)     
        ftest=ftest.transpose(3, 4)
        ftrain=ftrain.transpose(3, 4)
        #print(ftrain.shape)
        #exit(0)        
        ftest = F.normalize(ftest, p=2, dim=ftest.dim()-1, eps=1e-12)
        ftrain = F.normalize(ftrain, p=2, dim=ftrain.dim()-1, eps=1e-12)
        ftest_mean = F.normalize(ftest_mean, p=2, dim=ftest_mean.dim()-1, eps=1e-12)
        ftrain_mean = F.normalize(ftrain_mean, p=2, dim=ftrain_mean.dim()-1, eps=1e-12)        
        #print(ftest_mean.shape,ftrain_mean.shape)#[4,75,5,1,512]
        #print(ftest.shape,ftrain.shape)#[4, 75, 5, 36, 512]       
        #exit(0)
        #print(ftest.shape,ftrain.shape,ftest_mean.shape,ftrain_mean.shape)        
        scores = self.scale_cls * torch.sum((ftest_mean * ftrain_mean).squeeze(), dim=-1)
        ftrain_scores = self.scale_cls * torch.sum(ftest_mean * ftrain, dim=-1)        
        ftest_scores = self.scale_cls * torch.sum(ftest * ftrain_mean, dim=-1)
        #print(ftest.shape,ftrain.shape)
        #print(ftrain_scores.shape,ftest_scores.shape)
        #print(self.scale_cls)
        #print(scores.shape,'scores')
        #K=1
        #_,train_ind=torch.max(ftrain_scores,3)
        #_,test_ind=torch.max(ftest_scores,3)
        _,train_ind=torch.topk(ftrain_scores, K, dim=3 )
        _,test_ind=torch.topk(ftest_scores, K, dim=3 )        
        #print(train_ind)
        #print(train_ind.shape)
        #exit(0)        
        train_ind=train_ind.view(-1)
        test_ind=test_ind.view(-1)

        train_one_hot=one_hot_36(train_ind).view(shape_train[0],75,5,K,-1).sum(3).unsqueeze(4)  
        test_one_hot=one_hot_36(test_ind).view(shape_train[0],75,5,K,-1).sum(3).unsqueeze(4)  
        #print(train_one_hot.shape)
        #print(train_one_hot[0,0,0,:])
        #exit(0)
        ftest_fuse=((ftest*test_one_hot.cuda()).sum(3))/K
        ftrain_fuse=((ftrain*train_one_hot.cuda()).sum(3))/K  
        ftest_fuse_mean = F.normalize(ftest_fuse, p=2, dim=ftest_fuse.dim()-1, eps=1e-12)
        ftrain_fuse_mean = F.normalize(ftrain_fuse, p=2, dim=ftrain_fuse.dim()-1, eps=1e-12)         
        #print(ftest_fuse.shape)
        #print(ftrain_fuse.shape)
        #exit(0)
        #scores_final = self.scale_cls * torch.sum(((ftest*test_one_hot.cuda()).sum(3) * ((ftrain* train_one_hot.cuda()).sum(3))), dim=-1)    
        scores_final = self.scale_cls * torch.sum((ftest_fuse_mean * ftrain_fuse_mean), dim=-1)        
        #print( test_one_hot.shape,train_one_hot.shape) #[4, 75, 5, 36]    
        #exit(0)        
        return scores,scores_final
        
    def test_select_topK(self, ftrain, ftest,K):
        ftest_mean = ftest.mean(4)
        ftest_mean = ftest_mean.mean(4).unsqueeze(3)
        ftrain_mean = ftrain.mean(4)
        ftrain_mean = ftrain_mean.mean(4).unsqueeze(3)  
        #print(ftrain_mean.shape)
        ftest=ftest.view(4, 75, 5, 512,-1)
        ftrain=ftrain.view(4, 75, 5, 512,-1)     
        ftest=ftest.transpose(3, 4)
        ftrain=ftrain.transpose(3, 4)
        #print(ftrain.shape)
        #exit(0)        
        ftest = F.normalize(ftest, p=2, dim=ftest.dim()-1, eps=1e-12)
        ftrain = F.normalize(ftrain, p=2, dim=ftrain.dim()-1, eps=1e-12)
        ftest_mean = F.normalize(ftest_mean, p=2, dim=ftest_mean.dim()-1, eps=1e-12)
        ftrain_mean = F.normalize(ftrain_mean, p=2, dim=ftrain_mean.dim()-1, eps=1e-12)        
        #print(ftest_mean.shape,ftrain_mean.shape)#[4,75,5,1,512]
        #print(ftest.shape,ftrain.shape)#[4, 75, 5, 36, 512]       
        #exit(0)
        #print(ftest.shape,ftrain.shape,ftest_mean.shape,ftrain_mean.shape)        
        scores = self.scale_cls * torch.sum((ftest_mean * ftrain_mean).squeeze(), dim=-1)
        ftrain_scores = self.scale_cls * torch.sum(ftest_mean * ftrain, dim=-1)        
        ftest_scores = self.scale_cls * torch.sum(ftest * ftrain_mean, dim=-1)
        #print(ftest.shape,ftrain.shape)
        #print(ftrain_scores.shape,ftest_scores.shape)
        #print(self.scale_cls)
        #print(scores.shape,'scores')
        #K=1
        #_,train_ind=torch.max(ftrain_scores,3)
        #_,test_ind=torch.max(ftest_scores,3)
        sorces_list=[]
        for i in range(35):
            print(i)
            _,train_ind=torch.topk(ftrain_scores, i+1, dim=3 )
            for j in range(35):  
                print(j)
                _,test_ind=torch.topk(ftest_scores, j+1, dim=3 )        
        #print(train_ind)
        #print(train_ind.shape)
        #exit(0)        
                train_ind=train_ind.view(-1)
                test_ind=test_ind.view(-1)

                train_one_hot=one_hot_36(train_ind).view(4,75,5,i+1,-1).sum(3).unsqueeze(4)  
                test_one_hot=one_hot_36(test_ind).view(4,75,5,j+1,-1).sum(3).unsqueeze(4)  
        #print(train_one_hot.shape)
        #print(train_one_hot[0,0,0,:])
        #exit(0)
                #print(ftest.shape,test_one_hot.shape)
                ftest_fuse=((ftest*test_one_hot.cuda()).sum(3))/(j+1)
                ftrain_fuse=((ftrain*train_one_hot.cuda()).sum(3))/(i+1)  
                ftest_fuse_mean = F.normalize(ftest_fuse, p=2, dim=ftest_fuse.dim()-1, eps=1e-12)
                ftrain_fuse_mean = F.normalize(ftrain_fuse, p=2, dim=ftrain_fuse.dim()-1, eps=1e-12)         
        #print(ftest_fuse.shape)
        #print(ftrain_fuse.shape)
        #exit(0)
        #scores_final = self.scale_cls * torch.sum(((ftest*test_one_hot.cuda()).sum(3) * ((ftrain* train_one_hot.cuda()).sum(3))), dim=-1)    
                scores_final = self.scale_cls * torch.sum((ftest_fuse_mean * ftrain_fuse_mean), dim=-1)  
                sorces_list.append(scores_final.view(4,75,5,1))
                #print(scores_final.shape)
                #exit(0)
        #print( test_one_hot.shape,train_one_hot.shape) #[4, 75, 5, 36] 
        scores_final,_=torch.cat(sorces_list,3).max(3)
        #print(scores_final.shape)
        #exit()
        #exit(0)        
        return scores,scores_final
        
    def forward(self, xtrain, xtest, ytrain, ytest,topK=28,test_fg=False):
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
        channel,wide,height=ftrain.shape[1],ftrain.shape[2],ftrain.shape[3]
        #print(channel,wide,height)
        #exit(0)
        #ftrain_temp=ftrain.view(batch_size, num_train,ftrain.shape[1],ftrain.shape[2],ftrain.shape[3])
        #ftrain=self.fusenet(ftrain_temp,ytrain)        
        #exit(0)
        ftrain = ftrain.view(batch_size, num_train, -1)#[4,25,18432] 
        
        #print(ftrain.shape)
        #exit(0)        
        ftrain = torch.bmm(ytrain, ftrain)#(4,5,18432),it is matrix multiply [4,5,25],[4,25,18432]
        #print(ftrain.shape,';;;;;;;')
        #print(ytrain.shape)
        N_class=ytrain.shape[1]
        #exit(0)
        
        #one-shot miniimagenet 65.29 use this code
        #ftrain_temp=ftrain.view(batch_size, N_class,channel,wide,height)
        #print(ftrain_temp
        #ftrain=self.fusenet(ftrain_temp,ytrain).view(batch_size, N_class, -1)  

        
        #print(ftrain.shape)
        #exit(0)
        #print(ytrain.sum(dim=2, keepdim=True).shape)
        #print(ftrain.div(ytrain.sum(dim=2, keepdim=True))
        #print(ytrain.sum(dim=2, keepdim=True).expand_as(ftrain)[0,:3,:10])
        #exit(0)
        ftrain = ftrain.div(ytrain.sum(dim=2, keepdim=True).expand_as(ftrain))# each row is the mean value of all support images of crospond class [4,5,18432]
        #print(ftrain.shape)
        #exit(0)
        ftrain_temp=ftrain.view(batch_size, N_class,channel,wide,height)
        #print(ftrain_temp[0,0,0])
        #print(ftrain_temp
        if not self.only_CSEI:
            ftrain,spatial,channel_attention=self.fusenet(ftrain_temp,ytrain)
        else:
            spatial=0
            channel_attention=0
        #ftrain_related=ftrain.view(-1, 512, 6,6)
        #print(ftrain[0,0,0])
        #exit(0)
        ftrain = ftrain.view(batch_size, N_class, -1)        
        ftrain = ftrain.view(batch_size, -1, *f.size()[1:])#[4,5,512,6,6]
        ftrain_class=ftrain.view( -1, *f.size()[1:])
        #ftrain_class=ftrain_class.view(4,5,512,-1).transpose(2,3).contiguous()
        #print(ftrain_class.shape)
        #ftrain_class=ftrain_class.view(4,-1,512)
        #loss=self.contrastLoss(ftrain_class,ytrain)
         
        #print(ftrain_class.shape,loss)
        #exit(0)
        #print(ftrain_class.shape,'ftrain')
        
        
        #exit(0)
        ftest = f[batch_size * num_train:]
        ftest = ftest.view(batch_size, num_test, *f.size()[1:])#[4,30,512,6,6] 
        #ftest=(spatial+1)*ftest
        ftest=(channel_attention+1)*ftest
        #print(ftest.shape,ftrain.shape,'use ftest_spatial')
        #exit(0)  
        if not test_fg:
            ftrain, ftest = self.cam(ftrain, ftest,test_fg)##[4,30,5,512,6,6],[4,30,5,512,6,6]
        else:
            ftrain, ftest,a1,a2 = self.cam(ftrain, ftest,test_fg)
        ftrain_ori=ftrain
        ftrain = ftrain.mean(4)
        ftrain = ftrain.mean(4)#[4,30,5,512]
        #print(test_fg)
        #exit()
        if not self.training:
            if test_fg:
                return self.test(ftrain, ftest),a1,a2
            else:
                return self.test_topK(ftrain_ori, ftest,topK)
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
        #ytrain_class = self.clasifier1(ftrain_class)
        
        params = list(self.clasifier.parameters())  
        #print(ytest.shape,len(params),params[0].shape,ftest.shape)
        #exit(0)        

        return ytest, cls_scores,ftest,params[0],spatial#,loss#,ytrain_class 
        