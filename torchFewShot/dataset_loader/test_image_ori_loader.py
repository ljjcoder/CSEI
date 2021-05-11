from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
from PIL import Image
import numpy as np
import os.path as osp
import lmdb
import io
import random
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

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


class FewShotDataset_test_imgori(Dataset):
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
        self.edge=1
        self.sigma=2        
        #print(self.nExemplars,self.nTestNovel,self.epoch_size) #5,75,2000
        #exit(0)
        seed = 112
        random.seed(seed)
        np.random.seed(seed)

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
    def load_edge(self, img, mask):
        sigma = self.sigma
        index=1
        # in test mode images are masked (with masked regions),
        # using 'mask' parameter prevents canny to detect edges for the masked regions
        #mask = None if self.training else (1 - mask / 255).astype(np.bool)

        # canny
        if self.edge == 1:
            # no edge
            if sigma == -1:
                return np.zeros(img.shape).astype(np.float)

            # random sigma
            if sigma == 0:
                sigma = random.randint(1, 4)

            return canny(img, sigma=sigma, mask=mask).astype(np.float)

        # external
        else:
            imgh, imgw = img.shape[0:2]
            edge = imread(self.edge_data[index])
            edge = self.resize(edge, imgh, imgw)

            # non-max suppression
            if self.nms == 1:
                edge = edge * canny(img, sigma=sigma, mask=mask)

            return edge
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
        labels = []
        images_ori=[]
        edges=[]
        imgs_gray=[]
        masks=[]
        cls = []        
        for (img_idx, label) in examples:
            img_ori = self.dataset[img_idx][0]
            #print(img)
            #exit(0)
            if self.load:
                img_ori = Image.fromarray(img_ori)
            else:
                img_ori = read_image(img_ori)
                #print(img.size)
                #print(np.array(img).shape)
                #exit(0)
            if self.transform is not None:
                img = self.transform(img_ori)
            img_gray = rgb2gray(np.array(img_ori))
            #edge = self.load_edge(img_gray, None) 
            #edge_tensor=Image.fromarray( edge)  
            #edge_tensor=F.to_tensor( edge).float()            
            #print(img.shape,'located in test_loader.py at 146')
            #exit(0)
            img_gray=Image.fromarray(img_gray)
            img_gray_tensor=F.to_tensor(img_gray).float()            
            imgs_gray.append(img_gray_tensor)             
            images.append(img)
            labels.append(label)
            
            masked_img=np.array(img_ori)#*(1-mask_3)+mask_3*255
            masked_img=Image.fromarray(masked_img)
            masked_img_tensor=F.to_tensor(masked_img).float()           
            images_ori.append(masked_img_tensor)
            
            #edges.append(edge)            
        images = torch.stack(images, dim=0)
        labels = torch.LongTensor(labels)
        images_ori = torch.stack(images_ori, dim=0)  
        #edges = torch.stack(edges, dim=0)
        imgs_gray = torch.stack(imgs_gray, dim=0) 
        
        return images, labels,images_ori,imgs_gray

    def __getitem__(self, index):
        Tnovel = self.Epoch_Tnovel[index]
        Exemplars = self.Epoch_Exemplar[index]
        Xt, Yt,xtori,xt_imgs_gray = self._creatExamplesTensorData(Exemplars)
        Xe, Ye,xeori,xe_imgs_gray  = self._creatExamplesTensorData(Tnovel)
        return Xt, Yt , xtori ,xt_imgs_gray, Xe, Ye#,xeori,xe_edges 

