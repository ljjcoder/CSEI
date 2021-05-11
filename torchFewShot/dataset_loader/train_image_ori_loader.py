from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import cv2
import os
from PIL import Image
import numpy as np
import os.path as osp
import lmdb
import io
import random
from scipy.misc import imread
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


class FewShotDataset_train_imgori(Dataset):
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
        self.edge=1
        self.sigma=2        
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
    def load_edge(self, img,  mask):
        sigma = self.sigma
        index=1
        # in test mode images are masked (with masked regions),
        # using 'mask' parameter prevents canny to detect edges for the masked regions
        mask = None #if self.training else (1 - mask / 255).astype(np.bool)
        #mask =(1 - mask / 255).astype(np.bool)
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
            cls: a tensor [nExemplars]
        """

        images = []
        labels = []
        images_ori=[]
        edges=[]
        imgs_gray=[]
        masks=[]
        cls = []
        #self.mask_root="/home/lijunjie/edge-connect-master/examples/fuse.png"            
            #self.mask_root="/home/lijunjie/Pconv/PConv-Keras_mask01/stroke_4/img00000266_mask.png" 
            #self.mask_root="/home/lijunjie/Pconv/PConv-Keras_mask01/stroke_4/img00021416_mask.png"             
            #self.mask_root="/home/lijunjie/edge-connect-master/examples/Places365_val_00006822_mask.png"              
            #index_mask=self.data[index].rfind('/')
            #name_mask=self.data[index][index_mask+1:len(self.data[index])]          
            #self.mask_root='/home/lijunjie/generative_inpainting-master_global_local/data/random_rec_places_10000/'+name_mask           
            #mask_root="/home/lijunjie/Pconv/PConv-Keras_mask01/stroke_4/"+mask_name[self.mask_id]+'.png'             
            #mask = imread(self.mask_data[index])
        
        #mask_3 = (imread(self.mask_root)/255).astype(dtype=np.uint8)
        #if len(mask_3.shape) < 3:
            #mask_3 = gray2rgb(mask_3)
        #print(mask_3.shape)
        
        #cv2.imwrite('./test_masked.png',img)        
        for (img_idx, label) in examples:
            img_ori, ids = self.dataset[img_idx]
            #img_ori='/home/lijunjie/edge-connect-master/examples/test_result/input_000.png'
            #exit(0)
            if self.load:
                img_ori = Image.fromarray(img_ori)
            else:
                img_ori = read_image(img_ori)
            if self.transform is not None:
                img = self.transform(img_ori)
            img_gray = rgb2gray(np.array(img_ori))
            #print(img_gray.shape)
            #exit(0)
            masked_img=np.array(img_ori)#*(1-mask_3)+mask_3*255
            masked_img=Image.fromarray(masked_img)
            masked_img_tensor=F.to_tensor(masked_img).float()
            #print(masked_img_tensor.shape)
            #exit(0)           
            images_ori.append(masked_img_tensor)
            #cv2.imwrite('./test_masked.png',np.array(img_ori)*(1-mask_3)+mask_3*255)             
            #mask = rgb2gray(mask_3)
            #mask = (mask > 0).astype(np.uint8) * 255
            #print(mask.shape)#(84,84)
            #exit(0)
            #mask_tensor=F.to_tensor(Image.fromarray(mask)).float()
            #masks.append(mask_tensor)
            #edge = self.load_edge(img_gray, None)
            #edge = self.load_edge(img_gray, mask)
            #print(edge.dtype,'lllkkkk')
            #exit(0)
            
            #edge_tensor=Image.fromarray( edge)  
            #edge_tensor=F.to_tensor( edge).float()
            img_gray=Image.fromarray(img_gray)
            img_gray_tensor=F.to_tensor(img_gray).float()            
            imgs_gray.append(img_gray_tensor)             
            images.append(img)
            labels.append(label)  
            #edges.append(edge_tensor)            
            cls.append(ids)
        #print(type(images[0]))
        images = torch.stack(images, dim=0)
        #masks = torch.stack(masks, dim=0)  
        #print(masks.shape,'llll')#(5,1,84,84)
        
        #print(images.shape)
        #exit(0)
        labels = torch.LongTensor(labels)
        images_ori = torch.stack(images_ori, dim=0) 
        #edges = torch.stack(edges, dim=0)         
        cls = torch.LongTensor(cls)
        imgs_gray = torch.stack(imgs_gray, dim=0)
        #print(imgs_gray.shape,'ljj')
        #exit(0)        
        return images, labels, cls,images_ori,imgs_gray#,masks


    def __getitem__(self, index):
        Tnovel, Exemplars = self._sample_episode()
        Xt, Yt, Ytc,Xt_img_ori,Xt_img_gray = self._creatExamplesTensorData(Exemplars)
        Xe, Ye, Yec,Xe_img_ori,Xe_img_gray= self._creatExamplesTensorData(Tnovel)
        return Xt, Yt,Ytc,Xt_img_ori,Xt_img_gray , Xe, Ye, Yec#,Xe_img_ori,Xe_edges,Xe_img_gray,Xe_masks
