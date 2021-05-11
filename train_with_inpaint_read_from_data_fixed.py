from __future__ import print_function
from __future__ import division

import os
import sys
import time
from PIL import Image
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
import torchvision.transforms.functional as Funljj
sys.path.append('./torchFewShot')


from args_mini import argument_parser
#from torchFewShot.models.net import Model

import losses
#from torchFewShot.models.models_gnn import create_models
#from torchFewShot.data_manager_imageori import DataManager
from torchFewShot.data_manager_image_inpainting_data import DataManager
#from torchFewShot.data_manager import DataManager
from torchFewShot.losses import CrossEntropyLoss
from torchFewShot.optimizers import init_optimizer
import transforms as T
from torchFewShot.utils.iotools import save_checkpoint, check_isfile
from torchFewShot.utils.avgmeter import AverageMeter
from torchFewShot.utils.logger import Logger
from torchFewShot.utils.torchtools import one_hot, adjust_learning_rate




parser = argument_parser()
args = parser.parse_args()
#print(args.use_similarity)
#exit(0)
if args.use_similarity:   
    print('similarity net removed')  
    exit(0)    
else: 
    from torchFewShot.models.net_related import Model   
      
    
only_test=False
save_best=True
only_CSEI=False
def main():

    os.system('cp ./train_with_inpaint_read_from_data_fixed.py ' +args.save_dir + 'train_with_inpaint_read_from_data_fixed.py')
    #exit()

    
    loss_fn = losses.GenericLoss('batchsgm', 0.02, 64)    
    torch.manual_seed(args.seed)
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    #print(use_gpu)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #torch.cuda.manual_seed(args.seed)    
    #torch.manual_seed(config.SEED)
    
    #torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(args.seed)
    random.seed(args.seed)
    sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = False
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)
        #exit(0)
    else:
        print("Currently using CPU (GPU is highly recommended)")
        
    print('Initializing image data manager')
    dm = DataManager(args, use_gpu)
    trainloader, testloader = dm.return_dataloaders()
    #model_edge = EdgeConnect(config)
    #model_edge.load()        
    print('\nstart testing...\n')
    #model_edge.test()
    #print(args.scale_cls,args.num_classes)
    #exit(0)
    #GNN_model=create_models(args,512)
    #print(args.use_similarity)
    #exit(0)
    if args.use_similarity:
        #GNN_model=create_models(args,512)    
        #model = Model(args,GNN_model,scale_cls=args.scale_cls, num_classes=args.num_classes)
        print('similarity remove')
        exit()
    else:
        model = Model(scale_cls=args.scale_cls,only_CSEI=only_CSEI, num_classes=args.num_classes)
        #model_tradclass = Model_tradi(scale_cls=args.scale_cls, num_classes=args.num_classes)
        #params_tradclass = torch.load('result/%s/CAM/1-shot-seed112_classic_classifier_avg_nouse_CAN/%s' % (args.dataset, 'best_model.pth.tar'))        
        #model_tradclass.load_state_dict(params_tradclass['state_dict'])  
        #params = torch.load('result/%s/CAM/5-shot-seed112_inpaint_batchsgmregular_begin_70epoch/%s' % (args.dataset, 'best_model.pth.tar'))      
        #model.load_state_dict(params['state_dict'])          
        #print('enter model_tradclass')
        #exit(0)
    if not only_CSEI:
        #params = torch.load('result/%s/CAM/5-shot-seed112_inpaint_batchsgmregular_begin_70epoch/%s' % (args.dataset, 'best_model.pth.tar'))
        #params = torch.load('../fewshot-CAN-master/result/%s/CAM/1-shot-seed112_inpaint_use_CAM_argumenttest_nouse_similarity_read_from_data1/%s' % (args.dataset, 'best_model.pth.tar'))
        params = torch.load('/home/yfchen/ljj_code/trained_model/mini/only_inpaint/1-shot/%s' % ('best_model.pth.tar'))
        #params = torch.load("/home/yfchen/ljj_code/spatial_test/result/miniImageNet/CAM/1-shot_only_augment_test_fixed_GPU0_2333/best_model.pth.tar")
        #params = torch.load('./result/%s/CAM/1-shot-seed112_inpaint_support_fuse_Cam_surport_from_65.3_spatial_tanh_8argument_traingloballabelargurement_contrast_normal/%s' % (args.dataset, 'best_model.pth.tar'))        
        #params = torch.load('result/%s/CAM/1-shot-seed112_inpaint_support_fuse_Cam_surport_from_65.3_debug/%s' % (args.dataset, 'best_model.pth.tar'))        
        #params = torch.load('result/%s/CAM/1-shot-seed112_inpaint_support_fuse_Cam_surport/%s' % (args.dataset, 'best_model.pth.tar'))
        #params_tradclass = torch.load('result/%s/CAM/1-shot-seed112_classic_classifier_global_avg/%s' % (args.dataset, 'checkpoint_inpaint67.pth.tar'))        
        print(type(params))

        model.load_state_dict(params['state_dict'], strict=False)
        #model_tradclass.load_state_dict(params_tradclass['state_dict'])        

    criterion = CrossEntropyLoss()
    optimizer = init_optimizer(args.optim, model.parameters(), args.lr, args.weight_decay)


    if use_gpu:
        model = model.cuda()
        #model_tradclass = model_tradclass.cuda()        

    start_time = time.time()
    train_time = 0
    best_acc = -np.inf
    best_epoch = 0
    print("==> Start training")
    for i in range(1):
        #print(i+14)
        acc = test_ori(model, testloader, use_gpu,28)
        #acc_5 = test_ori_5(model, testloader, use_gpu)
    #print("==> Test 5-way Best accuracy {:.2%}, achieved at epoch {}".format( acc, 0)) 
    #exit()  
    #print(args.save_dir)
    #exit()
    best_path=args.save_dir+'best_model.pth.tar'
    if only_CSEI:
        args.max_epoch=70
    for epoch in range(args.max_epoch):
        if not args.Classic:
            learning_rate = adjust_learning_rate(optimizer, epoch, args.LUT_lr)
        else:
            optimizer_tradclass = init_optimizer(args.optim, model_tradclass.parameters(), args.lr, args.weight_decay)
            learning_rate = adjust_learning_rate(optimizer_tradclass, epoch, args.LUT_lr)  

        start_train_time = time.time()
        #model.base.eval()
        #exit(0)
        #print(not True)
        #exit(0)
        if not only_test:
            #print(';;;;;;;;;;;')
            #exit(0)
            if not args.Classic:
                print('enter train code')
                train(epoch, model, criterion,loss_fn, optimizer, trainloader, learning_rate, use_gpu)
                #print('oooo')
            else:
                acc=train(epoch,model_edge, model_tradclass, criterion, optimizer_tradclass, trainloader, learning_rate, use_gpu)
                
            train_time += round(time.time() - start_train_time)
        
        if epoch == 0 or epoch > (args.stepsize[0]-1) or (epoch + 1) % 10 == 0:
            print('enter test code')
            #exit(0)
            if not args.Classic:
                #acc = test(model_edge, model, model_tradclass,weight_softmax, testloader, use_gpu)
                acc = test_ori(model, testloader, use_gpu)
                #acc_5 = test_ori_5(model, testloader, use_gpu)
            is_best = acc > best_acc
            #else:
            
            
            #print(acc)
            #exit(0)
            if is_best:
                best_acc = acc
                best_epoch = epoch + 1
            if not only_test:
                if not args.Classic:
                    if save_best:
                        save_checkpoint({
                            'state_dict': model.state_dict(),
                            'acc': acc,
                            'epoch': epoch,
                        }, is_best, osp.join(args.save_dir, 'checkpoint_inpaint' + str(epoch + 1) + '.pth.tar'))
                    else:
                        print('not save')
                if  args.Classic:                
                    save_checkpoint({
                        'state_dict': model_tradclass.state_dict(),
                        'acc': acc,
                        'epoch': epoch,
                    }, is_best, osp.join(args.save_dir, 'checkpoint_classic' + str(epoch + 1) + '.pth.tar'))                

            print("==> Test 5-way Best accuracy {:.2%}, achieved at epoch {}".format(best_acc, best_epoch))
    acc_5 = test_ori_5(model,best_path, testloader, use_gpu)
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))
    print("==========\nArgs:{}\n==========".format(args))



        
             
def train(epoch, model, criterion,loss_fn, optimizer, trainloader, learning_rate, use_gpu):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    std=np.expand_dims(np.array([0.229, 0.224, 0.225]),axis=1)
    std=np.expand_dims(std,axis=2) 
    mean=np.expand_dims(np.array([0.485, 0.456, 0.406]),axis=1)             
    mean=np.expand_dims(mean,axis=2) 
    model.train()
    #model_edge.eval()
    #model_tradclass.eval()
    end = time.time()
    #print('llllllllllllll','located in train_with_inpaint_final.py at 264')
    #exit(0)
    #for batch_idx, (images_train, labels_train,tpids,Xt_img_ori,Xt_img_gray,images_test, labels_test, pids) in enumerate(trainloader):
    for batch_idx, (images_train,images_train1,images_train2,images_train3,images_train4,images_train5,images_train6,images_train7,images_train8, labels_train,tpids, images_test,images_test1,images_test2,images_test3,images_test4, labels_test, pids) in enumerate(trainloader):    
        data_time.update(time.time() - end)
        #print(Xt_img_ori.shape,Xt_img_gray.shape,images_train.shape,'lll')
        edges=[]
        if only_CSEI:
            augment_k=4
        else:
            augment_k=8
        tpids_4 = tpids.reshape(4,-1,1)#[:,:,0]

        tpids_4 = tpids_4.repeat(1,1,augment_k).reshape(4,-1)  

        K_shot=images_train.shape[1]/5
        images_train1=images_train1.reshape(4,-1,1,3,84,84)
        images_train2=images_train2.reshape(4,-1,1,3,84,84)
        images_train3=images_train3.reshape(4,-1,1,3,84,84)
        images_train4=images_train4.reshape(4,-1,1,3,84,84) 
        images_train5=images_train5.reshape(4,-1,1,3,84,84)
        images_train6=images_train6.reshape(4,-1,1,3,84,84)
        images_train7=images_train7.reshape(4,-1,1,3,84,84)
        images_train8=images_train8.reshape(4,-1,1,3,84,84)         
        #print(images_test.shape)
        #exit(0)
        #images_test1=images_test1.reshape(4,30,1,3,84,84)
        #images_test2=images_test2.reshape(4,30,1,3,84,84)
        #images_test3=images_test3.reshape(4,30,1,3,84,84)
        #images_test4=images_test4.reshape(4,30,1,3,84,84)
       
        #if cuda  memory enough use follow code
        if only_CSEI:
            images_train_4=torch.cat((images_train1, images_train2,images_train3,images_train4), 2)
        else:
            images_train_4=torch.cat((images_train1, images_train2,images_train3,images_train4,images_train5,images_train6,images_train7,images_train8), 2)   
        #if cuda  memory not enough use follow this code
        #images_train_4=torch.cat((images_train1, images_train2,images_train3), 2)
        #images_train_fuse=   torch.cat((images_train.reshape(4,-1,1,3,84,84), images_train1, images_train2,images_train3), 2)     
        #images_test=images_test.reshape(4,30,1,3,84,84)        
        #images_test_4=torch.cat((images_test,images_test1, images_test2,images_test3, images_test4), 2)
        #images_test_4=torch.cat((images_test,images_test3, images_test4), 2)        
        #images_test=images_test_4.reshape(4,-1,3,84,84)       
        labels_train_4 = labels_train.reshape(4,-1,1)#[:,:,0]

        labels_train_4 = labels_train_4.repeat(1,1,augment_k)
        labels_test_4=labels_train_4[:,:,:augment_k]
        labels_train_4 = labels_train_4.reshape(4,-1)  
        labels_test_4=labels_test_4.reshape(4,-1)       
       
        if use_gpu:
            images_train, labels_train,images_train_4 = images_train.cuda(), labels_train.cuda(),images_train_4.cuda()
            #images_train_fuse=images_train_fuse.cuda()
            
            images_test, labels_test = images_test.cuda(), labels_test.cuda()
            pids = pids.cuda()
            labels_train_4=labels_train_4.cuda()
            labels_test_4=labels_test_4.cuda()
            tpids_4 = tpids_4.cuda()
            tpids=tpids.cuda()
        pids_con=torch.cat((pids, tpids_4), 1)
        labels_test_4=torch.cat((labels_test, labels_test_4), 1)
        #tpids

        batch_size, num_train_examples, channels, height, width = images_train.size()
        num_test_examples = images_test.size(1)
        
        labels_train_1hot = one_hot(labels_train).cuda()
    
        train_pid=torch.matmul(labels_train_1hot.transpose(1, 2),tpids.unsqueeze(2).float()).squeeze()
        train_pid=(train_pid/K_shot).long()

        
        #exit()
        labels_train_1hot_4 = one_hot(labels_train_4).cuda()        
        #labels_train = labels_train.view(batch_size * num_train_examples)   
        #print( labels_train)
        #exit(0)        
        labels_test_1hot = one_hot(labels_test).cuda()
        labels_test_1hot_4 = one_hot(labels_test_4).cuda()
 
        #support set
        switch=np.random.uniform(0,1)
        if switch>-1:
            images_train=images_train.reshape(4,-1,3,84,84)
        else:
            images_train=images_train1.cuda().reshape(4,-1,3,84,84) 
            #images_train1            
        images_train_4=images_train_4.reshape(4,-1,3,84,84)
        #inpaint_tensor=torch.from_numpy(inpaint_img_np).cuda().reshape(4,20,3,84,84).float()        
        images_test=torch.cat((images_test, images_train_4), 1).reshape(4,-1,3,84,84)#images_train
        

        
        
 



        

        ytest, cls_scores,features,params_classifier,spatial = model(images_train, images_test, labels_train_1hot, labels_test_1hot_4)#ytest is all class classification
                                                                                                #cls_scores is N-way classifation

        loss1 = criterion(ytest, pids_con.view(-1))                  
        loss2 = criterion(cls_scores, labels_test_4.view(-1))        

        if epoch>900: 
            loss3 = loss_fn(params_classifier,ytest, features,pids_con)        
            loss = loss1 + 0.5 * loss2+loss3
        else:
            loss= loss1 + 0.5 * loss2#+0.5*loss_contrast
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
def test_ori(model, testloader, use_gpu,topK=28):
    accs = AverageMeter()
    test_accuracies = []
    final_accs = AverageMeter()
    final_test_accuracies = []    
    model.eval()
    with torch.no_grad():
        #for batch_idx , (images_train, labels_train,Xt_img_ori,Xt_img_gray, images_test, labels_test) in enumerate(testloader):
        for batch_idx , (images_train, images_train2,images_train3,images_train4,images_train5,labels_train, images_test, labels_test) in enumerate(testloader):        
            if use_gpu:
                images_train = images_train.cuda()
                images_test = images_test.cuda()

            end = time.time()
    
            batch_size, num_train_examples, channels, height, width = images_train.size()
            num_test_examples = images_test.size(1)

            labels_train_1hot = one_hot(labels_train).cuda()
            labels_test_1hot = one_hot(labels_test).cuda()

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
    
def test_ori_5(model,best_path, testloader, use_gpu,topK=28):
    accs = AverageMeter()
    test_accuracies = []
    final_accs = AverageMeter()
    final_test_accuracies = [] 
    params = torch.load(best_path)
    model.load_state_dict(params['state_dict'], strict=True)            
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
def test_vis(model, testloader, use_gpu):
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
