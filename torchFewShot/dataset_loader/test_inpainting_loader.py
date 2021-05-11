from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
from PIL import Image
import numpy as np
import transforms as T
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
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class FewShotDataset_test_inpainting(Dataset):
    """Few shot epoish Dataset

    Returns a task (Xtrain, Ytrain, Xtest, Ytest) to classify'
        Xtrain: [nKnovel*nExpemplars, c, h, w].
        Ytrain: [nKnovel*nExpemplars].
        Xtest:  [nTestNovel, c, h, w].
        Ytest:  [nTestNovel].
    """

    def __init__(self,
                 dataset, # dataset of [(img_path, cats), ...].
                 labels2inds, # labels of index {(cats: index1, index2, ...)}.
                 labelIds, # train labels [0, 1, 2, 3, ...,].
                 nKnovel=5, # number of novel categories.
                 nExemplars=1, # number of training examples per novel category.
                 nTestNovel=2*5, # number of test examples for all the novel categories.
                 epoch_size=2000, # number of tasks per eooch.
                 transform=None,
                 load=True,
                 seed=223,
                 **kwargs
                 ):
        
        self.dataset = dataset
        self.labels2inds = labels2inds
        self.labelIds = labelIds
        #print(labelIds)
        #print(len(labelIds))
        #exit(0)
        self.nKnovel = nKnovel
        self.transform = transform

        self.nExemplars = nExemplars
        self.nTestNovel = nTestNovel
        self.epoch_size = epoch_size
        self.load = load
        #print(self.nExemplars,self.nTestNovel,self.epoch_size) #5,75,2000
        #exit(0)
        seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.transform_test = T.Compose([
                T.Resize((84, 84), interpolation=3),
                T.RandomCrop(84, padding=8),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        self.Epoch_Exemplar = []
        self.Epoch_Tnovel = []
        for i in range(epoch_size):
            Tnovel, Exemplar = self._sample_episode()
            self.Epoch_Exemplar.append(Exemplar)
            self.Epoch_Tnovel.append(Tnovel)

    def __len__(self):
        return self.epoch_size

    def _sample_episode(self):
        """sampels a training epoish indexs.
        Returns:
            Tnovel: a list of length 'nTestNovel' with 2-element tuples. (sample_index, label)
            Exemplars: a list of length 'nKnovel * nExemplars' with 2-element tuples. (sample_index, label)
        """

        Knovel = random.sample(self.labelIds, self.nKnovel)
        #print(Knovel)
        #exit(0)
        nKnovel = len(Knovel)
        assert((self.nTestNovel % nKnovel) == 0)
        nEvalExamplesPerClass = int(self.nTestNovel / nKnovel)
        #print(nEvalExamplesPerClass)
        #exit(0)
        Tnovel = []
        Exemplars = []
        for Knovel_idx in range(len(Knovel)):
            ids = (nEvalExamplesPerClass + self.nExemplars)
            img_ids = random.sample(self.labels2inds[Knovel[Knovel_idx]], ids) 

            imgs_tnovel = img_ids[:nEvalExamplesPerClass]
            imgs_emeplars = img_ids[nEvalExamplesPerClass:]
            #print(imgs_tnovel)
            #exit(0)
            Tnovel += [(img_id, Knovel_idx) for img_id in imgs_tnovel]
            Exemplars += [(img_id, Knovel_idx) for img_id in imgs_emeplars]
        assert(len(Tnovel) == self.nTestNovel)
        assert(len(Exemplars) == nKnovel * self.nExemplars)
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
        """

        images = []
        
        images2 = []
        images3 = []
        images4 = []
        images5 = []        
        labels = []
        for (img_idx, label) in examples:
            img = self.dataset[img_idx][0]
            #print(img)
            ##exit(0)
            if self.load:
                img = Image.fromarray(img)
            else:
                img = read_image(img)
                #print(img.size)
                #print(np.array(img).shape)
                #exit(0)
            if self.transform is not None:
                img1 = self.transform(img)

                img2 = self.transform_test(img)
                img3 = self.transform_test(img)
                img4 = self.transform_test(img)
                img5 = self.transform_test(img)  
                #print((img2-img1).abs().sum(),(img3-img1).abs().sum(),(img2-img3).abs().sum())
            #print(img.shape,'located in test_loader.py at 146')
            #exit(0)
            images.append(img1)
            
            images2.append(img2)
            images3.append(img3)
            images4.append(img4)
            images5.append(img5)            
            labels.append(label)
        images = torch.stack(images, dim=0)

        images2 = torch.stack(images2, dim=0)
        images3 = torch.stack(images3, dim=0)
        images4 = torch.stack(images4, dim=0)
        images5 = torch.stack(images5, dim=0)        
        labels = torch.LongTensor(labels)
        return images, images2,images3,images4,images5,labels

    def __getitem__(self, index):
        Tnovel = self.Epoch_Tnovel[index]
        Exemplars = self.Epoch_Exemplar[index]
        Xt, Xt2,Xt3,Xt4,Xt5,Yt = self._creatExamplesTensorData(Exemplars)
        Xe,Xe2,Xe3,Xe4,Xe5, Ye = self._creatExamplesTensorData(Tnovel)
        return Xt,  Xt2,Xt3,Xt4,Xt5,Yt, Xe, Ye

