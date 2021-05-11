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
import cv2

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch.nn.functional as F
sys.path.append('./torchFewShot')

#from args_tiered import argument_parser
from args_xent import argument_parser
#from torchFewShot.models.net import Model

from torchFewShot.models.models_gnn import create_models
from torchFewShot.data_manager import DataManager
from torchFewShot.losses import CrossEntropyLoss
from torchFewShot.optimizers import init_optimizer

from torchFewShot.utils.iotools import save_checkpoint, check_isfile
from torchFewShot.utils.avgmeter import AverageMeter
from torchFewShot.utils.logger import Logger
from torchFewShot.utils.torchtools import one_hot, adjust_learning_rate

#from inpainting import EC


parser = argument_parser()
args = parser.parse_args()
#print(args.use_similarity)
#exit(0)
if args.use_similarity:
    from torchFewShot.models.net_similary import Model
else:
    from torchFewShot.models.net import Model    
only_test=False
def main():
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()

    sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")
        
    print('Initializing image data manager')
    dm = DataManager(args, use_gpu)
    trainloader, testloader = dm.return_dataloaders()
    #print(args.scale_cls,args.num_classes)
    #exit(0)
    GNN_model=create_models(args,512)
    if args.use_similarity:
        model = Model(args,GNN_model,scale_cls=args.scale_cls, num_classes=args.num_classes)
    else:
        model = Model(scale_cls=args.scale_cls, num_classes=args.num_classes)
    if only_test:
        params = torch.load('result/%s/CAM/1-shot-seed112/%s' % (args.dataset, 'best_model.pth.tar'))
        print(type(params))
    #exit(0)
    #for key in params.keys():
        #print(type(key))
    #exit(0)
        model.load_state_dict(params['state_dict'])
    #exit(0)
    criterion = CrossEntropyLoss()
    optimizer = init_optimizer(args.optim, model.parameters(), args.lr, args.weight_decay)

    if use_gpu:
        model = model.cuda()

    start_time = time.time()
    train_time = 0
    best_acc = -np.inf
    best_epoch = 0
    print("==> Start training")

    for epoch in range(args.max_epoch):
        learning_rate = adjust_learning_rate(optimizer, epoch, args.LUT_lr)

        start_train_time = time.time()
        #exit(0)
        #print(not True)
        #exit(0)
        if not only_test:
            #print(';;;;;;;;;;;')
            #exit(0)
            train(epoch, model, criterion, optimizer, trainloader, learning_rate, use_gpu)
            train_time += round(time.time() - start_train_time)
        
        if epoch == 0 or epoch > (args.stepsize[0]-1) or (epoch + 1) % 10 == 0:
            print('enter test code')
            #exit(0)
            acc = test(model, testloader, use_gpu)
            is_best = acc > best_acc
            #print(acc)
            #exit(0)
            if is_best:
                best_acc = acc
                best_epoch = epoch + 1
            if not only_test:            
                save_checkpoint({
                    'state_dict': model.state_dict(),
                    'acc': acc,
                    'epoch': epoch,
                }, is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch + 1) + '.pth.tar'))

            print("==> Test 5-way Best accuracy {:.2%}, achieved at epoch {}".format(best_acc, best_epoch))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))
    print("==========\nArgs:{}\n==========".format(args))


def train(epoch, model, criterion, optimizer, trainloader, learning_rate, use_gpu):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()

    end = time.time()
    for batch_idx, (images_train, labels_train, images_test, labels_test, pids) in enumerate(trainloader):
        data_time.update(time.time() - end)
        #pids is the all class id
        #print(labels_train.shape)
        #exit(0)        
        if use_gpu:
            images_train, labels_train = images_train.cuda(), labels_train.cuda()
            images_test, labels_test = images_test.cuda(), labels_test.cuda()
            pids = pids.cuda()

        batch_size, num_train_examples, channels, height, width = images_train.size()
        num_test_examples = images_test.size(1)

        labels_train_1hot = one_hot(labels_train).cuda()
        labels_test_1hot = one_hot(labels_test).cuda()

        ytest, cls_scores = model(images_train, images_test, labels_train_1hot, labels_test_1hot)#ytest is all class classification
                                                                                                 #cls_scores is N-way classifation

        loss1 = criterion(ytest, pids.view(-1))#
        loss2 = criterion(cls_scores, labels_test.view(-1))
        loss = loss1 + 0.5 * loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), pids.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

    print('Epoch{0} '
          'lr: {1} '
          'Time:{batch_time.sum:.1f}s '
          'Data:{data_time.sum:.1f}s '
          'Loss:{loss.avg:.4f} '.format(
           epoch+1, learning_rate, batch_time=batch_time, 
           data_time=data_time, loss=losses))


def test(model, testloader, use_gpu):
    accs = AverageMeter()
    test_accuracies = []
    model.eval()

    with torch.no_grad():
        for batch_idx , (images_train, labels_train, images_test, labels_test) in enumerate(testloader):
            if use_gpu:
                images_train = images_train.cuda()
                images_test = images_test.cuda()
            #print(images_test.shape, 'located in train.py at 177' )
            #print(images_test.shape[0],'located in train.py at 177')
            #exit(0)
            std=np.expand_dims(np.array([0.229, 0.224, 0.225]),axis=1)
            std=np.expand_dims(std,axis=2) 
            mean=np.expand_dims(np.array([0.485, 0.456, 0.406]),axis=1)             
            mean=np.expand_dims(mean,axis=2)  
            #print(std.shape,mean.shape)
            #exit(0)
            #for i in range(images_test.shape[0]):
                #for j in range(images_test.shape[1]):
                    #images_temp=images_test[i,j,:,:].cpu().numpy()
                    #print(images_temp.shape)
                    
                    #images_temp=images_temp*std+mean
                    #images_ori=images_temp.transpose((1,2,0))
                    #print(images_ori.shape)
                    #print(images_ori.max(0).max(0).max(0),images_ori.min(0).min(0).min(0))
                    #exit(0)
                    #images_ori=np.uint8(images_ori*255)
                    #cv2.imwrite('./result/vis_images/images_ori.jpg',images_ori)
                    #exit(0)
            end = time.time()

            batch_size, num_train_examples, channels, height, width = images_train.size()
            num_test_examples = images_test.size(1)

            labels_train_1hot = one_hot(labels_train).cuda()
            labels_test_1hot = one_hot(labels_test).cuda()

            cls_scores,a1,a2 = model(images_train, images_test, labels_train_1hot, labels_test_1hot,True)
            #a1.
            #print(a1.shape,a2.shape,'located in train.py at 209',(a1-1).max(),(a1-1).min())#[4,5,75,6,6]
            #print(type(a1.max(3)))
            #exit(0)
            max_a1=a1.max(3)[0].max(3)[0].unsqueeze(3).unsqueeze(3)
            min_a1=a1.min(3)[0].min(3)[0].unsqueeze(3).unsqueeze(3)
            max_a2=a2.max(3)[0].max(3)[0].unsqueeze(3).unsqueeze(3)
            min_a2=a2.min(3)[0].min(3)[0].unsqueeze(3).unsqueeze(3)
            #print(min_a1.shape,min_a1[0,0,0],max_a1[0,0,0])
            #exit(0)
            #print(std.shape,mean.shape)
            #exit(0)
            scale_a1=torch.div((a1-min_a1),(max_a1-min_a1))
            scale_a2=torch.div((a2-min_a2),(max_a2-min_a2)) 
            #print(images_train.shape[1],images_test.shape[1],'located in train.py at 224')
            #exit(0)
            #print(scale_a1[0,0,1],scale_a2[0,0,1])
            #exit(0)
            result_surpport_imgs=np.zeros((84*5+8*4,84*4+8*3,3)).astype(dtype=np.uint8)
            #print(labels_test[0])
            #exit(0)
            #result_test_imgs=np.zeros((84+3)*20,(84+3)*75,3)
        
            cls_scores = cls_scores.view(batch_size * num_test_examples, -1)
            labels_test = labels_test.view(batch_size * num_test_examples)

            _, preds = torch.max(cls_scores.detach().cpu(), 1)
            #print(labels_test.numpy()[:75])            
            #print(preds.numpy()[:75])
            #exit(0)
            acc = (torch.sum(preds == labels_test.detach().cpu()).float()) / labels_test.size(0)
            accs.update(acc.item(), labels_test.size(0))
            #print(images_train.shape,images_test.shape)
            #print(scale_a1.shape)
            #exit(0)
            if only_test:
                for i in range(images_test.shape[0]):
                    for k in range(images_test.shape[1]):
                        for j in range(images_train.shape[1]):
                            images_temp_test=images_test[i,k,:,:].cpu().numpy()
                            images_temp_train=images_train[i,j,:,:].cpu().numpy()                        
                            #print(images_temp.shape)
                            index_support=labels_train[i,j]
                            index_test= labels_test[i*num_test_examples+k]                           
                            #print(label_gt,label_pred)
                            #exit(0)
                            images_temp_test=images_temp_test*std+mean
                            images_ori_test=images_temp_test.transpose((1,2,0))[:,:,::-1]
                        
                            images_temp_train=images_temp_train*std+mean
                            images_ori_train=images_temp_train.transpose((1,2,0))[:,:,::-1]                    
                            #print(images_ori.shape)
                            #print(images_ori.max(0).max(0).max(0),images_ori.min(0).min(0).min(0))
                            #exit(0)
                            hot_a1=cv2.resize(np.uint8(scale_a1[i,index_support,k].cpu().numpy()*255),(84,84))
                            hot_a2=cv2.resize(np.uint8(scale_a2[i,index_support,k].cpu().numpy()*255),(84,84))
                            heatmap_a1 = cv2.applyColorMap(hot_a1, cv2.COLORMAP_JET)
                            heatmap_a2 = cv2.applyColorMap(hot_a2, cv2.COLORMAP_JET)                        
                            #print(heatmap_a1.shape)
                        
                            #exit(0)
                            images_ori_test=np.uint8(images_ori_test*255)
                            images_ori_train=np.uint8(images_ori_train*255)
                            vis_test=images_ori_test*0.7+heatmap_a2*0.3
                            #hot_a1=scale_a1[i,k,j]
                            #hot_a2=scale_a2[i,k,j]  
                            vis_train=images_ori_train*0.7+heatmap_a1*0.3                        
                            #cv2.imwrite('./result/vis_images/images_ori_test.jpg',images_ori_test)
                            #cv2.imwrite('./result/vis_images/images_test.jpg',vis_test)
                            #cv2.imwrite('./result/vis_images/images_ori_train.jpg',images_ori_train)
                            #cv2.imwrite('./result/vis_images/images_train.jpg',vis_train)  
                            result_surpport_imgs[84*index_support+8*index_support:84*(index_support+1)+8*index_support,:84,:]=images_ori_test
                            result_surpport_imgs[84*index_support+8*index_support:84*(index_support+1)+8*index_support,84+8:84+84+8,:]=images_ori_train  
                            result_surpport_imgs[84*index_support+8*index_support:84*(index_support+1)+8*index_support,84*2+8*2:84*3+8*2,:]=vis_test 
                            result_surpport_imgs[84*index_support+8*index_support:84*(index_support+1)+8*index_support,84*3+8*3:84*4+8*3,:]=vis_train
                        label_gt=int(labels_test.numpy()[k])
                        label_pred=int(preds.numpy()[k])    
                        cv2.imwrite('./result/vis_images/vis'+'_'+str(batch_idx)+'_'+str(i)+'_'+str(k)+'_'+str(label_gt)+'_'+str(label_pred)+'.jpg',result_surpport_imgs)                            
                #exit(0)
            if not True:                                
                if batch_idx>12:
                    break
            gt = (preds == labels_test.detach().cpu()).float()
            gt = gt.view(batch_size, num_test_examples).numpy() #[b, n]
            acc = np.sum(gt, 1) / num_test_examples
            acc = np.reshape(acc, (batch_size))
            test_accuracies.append(acc)
    #exit(0)
    accuracy = accs.avg
    test_accuracies = np.array(test_accuracies)
    test_accuracies = np.reshape(test_accuracies, -1)
    stds = np.std(test_accuracies, 0)
    ci95 = 1.96 * stds / np.sqrt(args.epoch_size)
    print('Accuracy: {:.2%}, std: :{:.2%}'.format(accuracy, ci95))
    #exit(0)
    return accuracy


if __name__ == '__main__':
    main()
