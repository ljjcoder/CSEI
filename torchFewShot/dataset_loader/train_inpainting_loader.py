from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import cv2
from PIL import Image
import numpy as np
import os.path as osp
import sys
sys.path.append("/ghome/lijj/python_package/")
import lmdb
import io
import random

import torch
from torch.utils.data import Dataset


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            #img=img.resize((100, 100),Image.ANTIALIAS)
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class FewShotDataset_train_inpainting(Dataset):
    """Few shot epoish Dataset

    Returns a task (Xtrain, Ytrain, Xtest, Ytest, Ycls) to classify'
        Xtrain: [nKnovel*nExpemplars, c, h, w].
        Ytrain: [nKnovel*nExpemplars].
        Xtest:  [nTestNovel, c, h, w].
        Ytest:  [nTestNovel].
        Ycls: [nTestNovel].
    """

    def __init__(self,
                 dataset, # dataset of [(img_path, cats), ...].
                 labels2inds, # labels of index {(cats: index1, index2, ...)}.
                 labelIds, # train labels [0, 1, 2, 3, ...,].
                 nKnovel=5, # number of novel categories.
                 nExemplars=1, # number of training examples per novel category.
                 nTestNovel=6*5, # number of test examples for all the novel categories.
                 epoch_size=2000, # number of tasks per eooch.
                 transform=None,
                 load=False,
                 **kwargs
                 ):
        
        self.dataset = dataset
        #print(self.dataset)
        #exit(0)
        self.labels2inds = labels2inds
        self.labelIds = labelIds
        self.nKnovel = nKnovel
        self.transform = transform
        #print(len(self.labels2inds) ,len(self.labelIds),self.nKnovel,nExemplars)
        #exit(0)

        self.nExemplars = nExemplars#5
        self.nTestNovel = nTestNovel
        self.epoch_size = epoch_size
        self.load = load

    def __len__(self):
        return self.epoch_size

    def _sample_episode(self):
        """sampels a training epoish indexs.
        Returns:
            Tnovel: a list of length 'nTestNovel' with 2-element tuples. (sample_index, label)
            Exemplars: a list of length 'nKnovel * nExemplars' with 2-element tuples. (sample_index, label)
        """

        Knovel = random.sample(self.labelIds, self.nKnovel)
        nKnovel = len(Knovel)
        assert((self.nTestNovel % nKnovel) == 0)
        nEvalExamplesPerClass = int(self.nTestNovel / nKnovel) # 6

        Tnovel = []
        Exemplars = []
        for Knovel_idx in range(len(Knovel)):
            ids = (nEvalExamplesPerClass + self.nExemplars)
            img_ids = random.sample(self.labels2inds[Knovel[Knovel_idx]], ids) 
            #print(img_ids)
            #exit(0)

            imgs_tnovel = img_ids[:nEvalExamplesPerClass]
            imgs_emeplars = img_ids[nEvalExamplesPerClass:]

            Tnovel += [(img_id, Knovel_idx) for img_id in imgs_tnovel]
            Exemplars += [(img_id, Knovel_idx) for img_id in imgs_emeplars]
        assert(len(Tnovel) == self.nTestNovel)
        assert(len(Exemplars) == nKnovel * self.nExemplars)
        #print(Exemplars)
        #exit(0)
        random.shuffle(Exemplars)
        random.shuffle(Tnovel)

        return Tnovel, Exemplars

    def _creatExamplesTensorData(self, examples):
        """
        Creats the examples image label tensor data.

        Args:
            examples: a list of 2-element tuples. (sample_index, label).

        Returns:
            images: a tensor [nExemplars, c, h, w]
            labels: a tensor [nExemplars]
            cls: a tensor [nExemplars]
        """

        images = []
        images1 = []
        images2 = []
        images3 = []
        images4 = [] 

        images5 = []
        images6 = []
        images7 = []
        images8 = []         
        labels = []
        cls = []
        for (img_idx, label) in examples:
            img, ids = self.dataset[img_idx]
            #print(img)
            #exit(0)
            img1=img.replace('train','train_1')
            img2=img.replace('train','train_2')
            img3=img.replace('train','train_3')
            img4=img.replace('train','train_4') 
            img5=img.replace('train','train_5/train_5')
            img6=img.replace('train','train_6/train_6')
            img7=img.replace('train','train_7/train_7')
            img8=img.replace('train','train_8/train_8') 
            
            #img5=img.replace('train','train_5')
            #img6=img.replace('train','train_6')
            #img7=img.replace('train','train_7')
            #img8=img.replace('train','train_8') 

            
           # print(img1,img2,img3,img4)
            #exit(0)
            #print(img8)
            
            #imtest=cv2.imread(img8)
            #print(imtest.shape)
            #exit(0)            
            if self.load:
                img = Image.fromarray(img)
            else:
                img = read_image(img)
                img1 = read_image(img1)                
                img2 = read_image(img2)                
                img3 = read_image(img3)
                img4 = read_image(img4)  

                img5 = read_image(img5)                
                img6 = read_image(img6)                
                img7 = read_image(img7)
                img8 = read_image(img8) 
                #img=img.resize((100, 100),Image.ANTIALIAS) 
                
                #img1=img1.resize((100, 100),Image.ANTIALIAS)  
                #img2=img2.resize((100, 100),Image.ANTIALIAS)  
                #img3=img3.resize((100, 100),Image.ANTIALIAS)  
                #img4=img4.resize((100, 100),Image.ANTIALIAS)  
                #img5=img5.resize((100, 100),Image.ANTIALIAS)  
                #img6=img6.resize((100, 100),Image.ANTIALIAS) 
                #img7=img7.resize((100, 100),Image.ANTIALIAS)  
                #img8=img8.resize((100, 100),Image.ANTIALIAS) 
                #print('pppppppppp')
                #exit(0)
            if self.transform is not None:
                img = self.transform(img)
                img1 = self.transform(img1)
                img2 = self.transform(img2)
                img3 = self.transform(img3)
                img4 = self.transform(img4)

                img5 = self.transform(img5)
                img6 = self.transform(img6)
                img7 = self.transform(img7)
                img8 = self.transform(img8)                
            images.append(img)
            images1.append(img1)
            images2.append(img2)
            images3.append(img3)
            images4.append(img4)
            
            images5.append(img5)
            images6.append(img6)
            images7.append(img7)
            images8.append(img8)            
            labels.append(label)
            cls.append(ids)
        images = torch.stack(images, dim=0)
        images1 = torch.stack(images1, dim=0)
        images2 = torch.stack(images2, dim=0)
        images3 = torch.stack(images3, dim=0)
        images4 = torch.stack(images4, dim=0) 

        images5 = torch.stack(images5, dim=0)
        images6 = torch.stack(images6, dim=0)
        images7 = torch.stack(images7, dim=0)
        images8 = torch.stack(images8, dim=0)         
        labels = torch.LongTensor(labels)
        cls = torch.LongTensor(cls)
        return images,images1,images2,images3,images4,images5,images6,images7,images8, labels, cls


    def __getitem__(self, index):
        Tnovel, Exemplars = self._sample_episode()
        Xt,Xt1,Xt2,Xt3,Xt4,Xt5,Xt6,Xt7,Xt8, Yt, Ytc = self._creatExamplesTensorData(Exemplars)
        Xe, Xe1, Xe2, Xe3, Xe4, Xe5, Xe6, Xe7, Xe8, Ye, Yec = self._creatExamplesTensorData(Tnovel)
        return Xt,Xt1,Xt2,Xt3,Xt4,Xt5,Xt6,Xt7,Xt8, Yt, Ytc, Xe, Xe1, Xe2, Xe3, Xe4, Ye, Yec
