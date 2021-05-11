from __future__ import print_function
from __future__ import division

import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np
import random

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch.nn.functional as F
sys.path.append('./torchFewShot')

from torchFewShot.models.net_related import Model
from torchFewShot.data_manager_image_inpainting_data import DataManager
from torchFewShot.losses import CrossEntropyLoss
from torchFewShot.optimizers import init_optimizer

from torchFewShot.utils.iotools import save_checkpoint, check_isfile
from torchFewShot.utils.avgmeter import AverageMeter
from torchFewShot.utils.logger import Logger
from torchFewShot.utils.torchtools import one_hot, adjust_learning_rate

parser = argparse.ArgumentParser(description='Test image model with 5-way classification')
# Datasets
parser.add_argument('-d', '--dataset', type=str, default='miniImageNet')
parser.add_argument('--load', default=False)
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--height', type=int, default=84,
                    help="height of an image (default: 84)")
parser.add_argument('--width', type=int, default=84,
                    help="width of an image (default: 84)")
# Optimization options
parser.add_argument('--train-batch', default=4, type=int,
                    help="train batch size")
parser.add_argument('--test-batch', default=8, type=int,
                    help="test batch size")
# Architecture
parser.add_argument('--num_classes', type=int, default=64)
parser.add_argument('--scale_cls', type=int, default=7)
parser.add_argument('--save-dir', type=str, default='')
parser.add_argument('--resume', type=str, default='', metavar='PATH')
# FewShot settting
parser.add_argument('--nKnovel', type=int, default=5,
                    help='number of novel categories')
parser.add_argument('--nExemplars', type=int, default=1,
                    help='number of training examples per novel category.')
parser.add_argument('--train_nTestNovel', type=int, default=6 * 5,
                    help='number of test examples for all the novel category when training')
parser.add_argument('--train_epoch_size', type=int, default=1200,
                    help='number of episodes per epoch when training')
parser.add_argument('--nTestNovel', type=int, default=15 * 5,
                    help='number of test examples for all the novel category')
parser.add_argument('--epoch_size', type=int, default=600,
                    help='number of batches per epoch')
# Miscs
parser.add_argument('--phase', default='test', type=str)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--gpu-devices', default='1', type=str)

args = parser.parse_args()


def main():
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()

    #sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    #print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")
        
    print('Initializing image data manager')
    dm = DataManager(args, use_gpu)
    trainloader, testloader = dm.return_dataloaders()

    model = Model(scale_cls=args.scale_cls, num_classes=args.num_classes)
    # load the model
    #best_path
    checkpoint = torch.load("/home/yfchen/ljj_code/spatial_test/result/miniImageNet/CAM/1-shot-seed112_inpaint_support_fuse_Cam_surport_from_64.99_test_fixed_GPU0_2333/best_model.pth.tar")
    model.load_state_dict(checkpoint['state_dict'])
    print("Loaded checkpoint from '{}'".format(args.resume))

    if use_gpu:
        model = model.cuda()

    acc_5 = test_ori_5(model, testloader, use_gpu)


def test_ori_5(model, testloader, use_gpu,topK=28):
    accs = AverageMeter()
    test_accuracies = []
    final_accs = AverageMeter()
    final_test_accuracies = [] 
    #params = torch.load(best_path)
    #model.load_state_dict(params['state_dict'], strict=True)            
    model.eval()

    with torch.no_grad():
        #for batch_idx , (images_train, labels_train,Xt_img_ori,Xt_img_gray, images_test, labels_test) in enumerate(testloader):
        for batch_idx , (images_train, images_train2,images_train3,images_train4,images_train5,labels_train, images_test, labels_test) in enumerate(testloader):   
            shape_test=images_train.shape[0]
            images_train1=images_train.reshape(shape_test,-1,1,3,84,84)
            images_train2=images_train2.reshape(shape_test,-1,1,3,84,84)
            images_train3=images_train3.reshape(shape_test,-1,1,3,84,84)
            images_train4=images_train4.reshape(shape_test,-1,1,3,84,84) 
            images_train5=images_train5.reshape(shape_test,-1,1,3,84,84) 

            labels_train_5 = labels_train.reshape(shape_test,-1,1)#[:,:,0]

            labels_train_5 = labels_train_5.repeat(1,1,5) 
            labels_train = labels_train_5.reshape(shape_test,-1)             
            images_train_5=torch.cat((images_train1, images_train2,images_train3,images_train4,images_train5), 2)   
            images_train=images_train_5.reshape(shape_test,-1,3,84,84)           
            if use_gpu:
                images_train = images_train.cuda()
                #images_train_5 = images_train_5.cuda()
                images_test = images_test.cuda()

            end = time.time()
            #print(images_train.shape,labels_train.shape)
            #exit()
            batch_size, num_train_examples, channels, height, width = images_train.size()
            num_test_examples = images_test.size(1)

            labels_train_1hot = one_hot(labels_train).cuda()
            labels_test_1hot = one_hot(labels_test).cuda()
            #print(images_train.shape,images_test.shape)
            cls_scores ,cls_scores_final= model(images_train, images_test, labels_train_1hot, labels_test_1hot,topK)
            #print(cls_scores.shape,cls_scores_final.shape)
            #exit(0)
            cls_scores = cls_scores.view(batch_size * num_test_examples, -1)
            cls_scores_final = cls_scores_final.view(batch_size * num_test_examples, -1)            
            labels_test = labels_test.view(batch_size * num_test_examples)

            _, preds = torch.max(cls_scores.detach().cpu(), 1)
            _, preds_final = torch.max(cls_scores_final.detach().cpu(), 1)            
            acc = (torch.sum(preds == labels_test.detach().cpu()).float()) / labels_test.size(0)
            accs.update(acc.item(), labels_test.size(0))

            acc_final = (torch.sum(preds_final == labels_test.detach().cpu()).float()) / labels_test.size(0)
            final_accs.update(acc_final.item(), labels_test.size(0))
            
            gt = (preds == labels_test.detach().cpu()).float()
            gt = gt.view(batch_size, num_test_examples).numpy() #[b, n]
            
            gt_final = (preds_final == labels_test.detach().cpu()).float()
            gt_final = gt_final.view(batch_size, num_test_examples).numpy() #[b, n]
            
            acc = np.sum(gt, 1) / num_test_examples
            acc = np.reshape(acc, (batch_size))
            test_accuracies.append(acc)

            acc_final = np.sum(gt_final, 1) / num_test_examples
            acc_final = np.reshape(acc_final, (batch_size))
            final_test_accuracies.append(acc_final)
            
    accuracy = accs.avg
    test_accuracies = np.array(test_accuracies)
    test_accuracies = np.reshape(test_accuracies, -1)
    stds = np.std(test_accuracies, 0)
    ci95 = 1.96 * stds / np.sqrt(args.epoch_size)
    
    accuracy_final = final_accs.avg
    test_accuracies_final = np.array(final_test_accuracies)
    test_accuracies_final = np.reshape(test_accuracies_final, -1)
    stds_final = np.std(test_accuracies_final, 0)
    ci95_final = 1.96 * stds_final / np.sqrt(args.epoch_size)    
    print('Accuracy: {:.2%}, std: :{:.2%}'.format(accuracy, ci95))
    print('Accuracy: {:.2%}, std: :{:.2%}'.format(accuracy_final, ci95_final))
    return accuracy


if __name__ == '__main__':
    main()
