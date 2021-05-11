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
from scipy.misc import imread
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torchvision.transforms.functional as Funljj
sys.path.append('./torchFewShot')

#from args_tiered import argument_parser
from args_xent import argument_parser
#from torchFewShot.models.net import Model

from torchFewShot.models.models_gnn import create_models
from torchFewShot.data_manager_imageori import DataManager
#from torchFewShot.data_manager import DataManager
from torchFewShot.losses import CrossEntropyLoss
from torchFewShot.optimizers import init_optimizer
import transforms as T
from torchFewShot.utils.iotools import save_checkpoint, check_isfile
from torchFewShot.utils.avgmeter import AverageMeter
from torchFewShot.utils.logger import Logger
from torchFewShot.utils.torchtools import one_hot, adjust_learning_rate

sys.path.append('/home/lijunjie/edge-connect-master')
from shutil import copyfile
from src.config import Config
from src.edge_connect_few_shot import EdgeConnect

#config = load_config(mode)
config_path = os.path.join('/home/lijunjie/edge-connect-master/checkpoints/places2_authormodel', 'config.yml')
config = Config(config_path)
config.TEST_FLIST = '/home/lijunjie/edge-connect-master/examples/test_result/'
config.TEST_MASK_FLIST = '/home/lijunjie/edge-connect-master/examples/places2/masks'
config.RESULTS = './checkpoints/EC_test'
config.MODE = 2
if config.MODE == 2:
    config.MODEL =  3
    config.INPUT_SIZE = 0
    config.mask_id=2
    #if args.input is not None:
        #config.TEST_FLIST = args.input

    #if args.mask is not None:
        #config.TEST_MASK_FLIST = args.mask

    #if args.edge is not None:
        #config.TEST_EDGE_FLIST = args.edge

    #if args.output is not None:
        #config.RESULTS = args.output
#exit(0)


parser = argument_parser()
args = parser.parse_args()
#print(args.use_similarity)
#exit(0)
if args.use_similarity:
    from torchFewShot.models.net_similary import Model
else:
    from torchFewShot.models.net import Model_mltizhixin , Model_tradi
    #print('enter ori net')
    #exit(0)
    
only_test=False
def returnCAM(feature_conv, weight_softmax, class_idx,output_cam ):
    # generate the class activation maps upsample to 256x256
    size_upsample = (84, 84)
    nc, h, w = feature_conv.shape
    #output_cam = []
    #print(class_idx)
    #exit(0)
    #print(class_idx, nc, h, w,weight_softmax[class_idx[0]].shape)
    #print(feature_conv.shape)
    #print(class_idx)
    #exit(0)
    cam_imgs_resize=[]
    for idx in class_idx[0]:
        #idx=int(idx)
        #print(idx)
        #exit(0)
        #print( weight_softmax[idx].shape,feature_conv.reshape((nc, h*w)).shape)
        #exit(0)
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        #cam_img = np.uint8((255 * cam_img)>200)*255
        cam_img = np.uint8(255 * cam_img)
        cam_img_resize=cv2.resize(cam_img, size_upsample)
        cam_img_resize = np.uint8((cam_img_resize)>200)*255
        #cv2.imwrite('./mask.jpg',cam_img*255)
        #exit(0)
        #print(cam_img.sum())
        #exit(0)
        #cam_img = np.uint8(255 * cam_img)
        mask_tensor=Funljj.to_tensor(Image.fromarray(cam_img_resize)).float()
        #print(mask_tensor.sum()) 
        #exit(0)        
        output_cam.append(mask_tensor)
        cam_imgs_resize.append(cam_img_resize)
    return output_cam,cam_imgs_resize
def main():
    #os.system('cp ./train_with_inpaint_read_from_data.py ' +args.save_dir + 'train_with_inpaint_read_from_data.py')
    ##os.system('cp ./net/network_ori.py '+config.tensorboard_folder + 'network_ori.py.backup')    
    #os.system('cp ./net/network_test.py '+config.tensorboard_folder + 'network_test.py.backup')    
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    config.DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True     
    #torch.manual_seed(config.SEED)
    
    #torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(args.seed)
    random.seed(args.seed)
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
    model_edge = EdgeConnect(config)
    model_edge.load()        
    print('\nstart testing...\n')
    #model_edge.test()
    #print(args.scale_cls,args.num_classes)
    #exit(0)
    #GNN_model=create_models(args,512)
    #print(args.use_similarity)
    #exit(0)
    if args.use_similarity:
        GNN_model=create_models(args,512)    
        model = Model(args,GNN_model,scale_cls=args.scale_cls, num_classes=args.num_classes)
    else:
        model = Model_mltizhixin(scale_cls=args.scale_cls, num_classes=args.num_classes)
        model_tradclass = Model_tradi(scale_cls=args.scale_cls, num_classes=args.num_classes)
        params_tradclass = torch.load('result/%s/CAM/1-shot-seed112_classic_classifier_avg_nouse_CAN/%s' % (args.dataset, 'best_model.pth.tar'))        
        model_tradclass.load_state_dict(params_tradclass['state_dict'])  
        #params = torch.load('result/%s/CAM/1-shot-seed112_inpaint_use_CAM/%s' % (args.dataset, 'checkpoint_inpaint67.pth.tar'))      
        #model.load_state_dict(params['state_dict'])          
        #print('enter model_tradclass')
        #exit(0)
    if False:
        params = torch.load('result/%s/CAM/1-shot-seed112/%s' % (args.dataset, 'best_model.pth.tar'))
        params_tradclass = torch.load('result/%s/CAM/1-shot-seed112_classic_classifier_global_avg/%s' % (args.dataset, 'checkpoint_inpaint67.pth.tar'))        
        print(type(params))
    #exit(0)
    #for key in params.keys():
        #print(type(key))
    #exit(0)
        #model.load_state_dict(params['state_dict'])
        model_tradclass.load_state_dict(params_tradclass['state_dict'])        
    #exit(0)
    #for ind,i in model.state_dict().items():
        #print (ind,i.shape)
    #exit(0)
    params = list(model_tradclass.parameters())    
    #fc_params=params[-2]
    weight_softmax = np.squeeze(params[-2].data.numpy())
    #print(weight_softmax.shape,type(params[-2]),params[-2].shape,params[-2].data.shape)
    #exit(0)
    criterion = CrossEntropyLoss()
    optimizer = init_optimizer(args.optim, model.parameters(), args.lr, args.weight_decay)
    #optimizer_tradclass = init_optimizer(args.optim, model_tradclass.parameters(), args.lr, args.weight_decay)    
    #model_tradclass

    if use_gpu:
        model = model.cuda()
        model_tradclass = model_tradclass.cuda()        

    start_time = time.time()
    train_time = 0
    best_acc = -np.inf
    best_epoch = 0
    print("==> Start training")

    for epoch in range(args.max_epoch):
        if not args.Classic:
            learning_rate = adjust_learning_rate(optimizer, epoch, args.LUT_lr)
        else:
            optimizer_tradclass = init_optimizer(args.optim, model_tradclass.parameters(), args.lr, args.weight_decay)
            learning_rate = adjust_learning_rate(optimizer_tradclass, epoch, args.LUT_lr)  
            #print('enter optimizer_tradclass')
            #exit(0)

        start_train_time = time.time()
        #exit(0)
        #print(not True)
        #exit(0)
        if not only_test:
            #print(';;;;;;;;;;;')
            #exit(0)
            if not args.Classic:
                print('enter train code')
                train(epoch,model_edge, model, model_tradclass,weight_softmax, criterion, optimizer, trainloader, learning_rate, use_gpu)
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
            is_best = acc > best_acc
            #else:
            
            
            #print(acc)
            #exit(0)
            if is_best:
                best_acc = acc
                best_epoch = epoch + 1
            if not only_test:
                if not args.Classic:
                    save_checkpoint({
                        'state_dict': model.state_dict(),
                        'acc': acc,
                        'epoch': epoch,
                    }, is_best, osp.join(args.save_dir, 'checkpoint_inpaint' + str(epoch + 1) + '.pth.tar'))
                if  args.Classic:                
                    save_checkpoint({
                        'state_dict': model_tradclass.state_dict(),
                        'acc': acc,
                        'epoch': epoch,
                    }, is_best, osp.join(args.save_dir, 'checkpoint_classic' + str(epoch + 1) + '.pth.tar'))                

            print("==> Test 5-way Best accuracy {:.2%}, achieved at epoch {}".format(best_acc, best_epoch))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))
    print("==========\nArgs:{}\n==========".format(args))

from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb
def load_edge( img,  mask):
    sigma = 2
    index=1
        # in test mode images are masked (with masked regions),
        # using 'mask' parameter prevents canny to detect edges for the masked regions
    mask = None if False else (1 - mask / 255).astype(np.bool)
        #mask =(1 - mask / 255).astype(np.bool)
        # canny
    if True:
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
transform_test = T.Compose([
                T.Resize((args.height, args.width), interpolation=3),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])         
def train(epoch,model_edge, model, model_tradclass,weight_softmax, criterion, optimizer, trainloader, learning_rate, use_gpu):
    
    if not os.path.isdir("/data4/lijunjie/mini-imagenet-tools/processed_images/mask/train_1"):
        os.mkdir("/data4/lijunjie/mini-imagenet-tools/processed_images/mask/train_1")
    if not os.path.isdir("/data4/lijunjie/mini-imagenet-tools/processed_images/mask/train_2"):        
        os.mkdir("/data4/lijunjie/mini-imagenet-tools/processed_images/mask/train_2")
    if not os.path.isdir("/data4/lijunjie/mini-imagenet-tools/processed_images/mask/train_3"):        
        os.mkdir("/data4/lijunjie/mini-imagenet-tools/processed_images/mask/train_3")
    if not os.path.isdir("/data4/lijunjie/mini-imagenet-tools/processed_images/mask/train_4"):        
        os.mkdir("/data4/lijunjie/mini-imagenet-tools/processed_images/mask/train_4")
    if not os.path.isdir("/data4/lijunjie/mini-imagenet-tools/processed_images/mask/train_full"):        
        os.mkdir("/data4/lijunjie/mini-imagenet-tools/processed_images/mask/train_full")
    if not os.path.isdir("/data4/lijunjie/mini-imagenet-tools/processed_images/train_full"):        
        os.mkdir("/data4/lijunjie/mini-imagenet-tools/processed_images/train_full")        
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    std=np.expand_dims(np.array([0.229, 0.224, 0.225]),axis=1)
    std=np.expand_dims(std,axis=2) 
    mean=np.expand_dims(np.array([0.485, 0.456, 0.406]),axis=1)             
    mean=np.expand_dims(mean,axis=2) 
    model.eval()
    #model_edge.eval()
    model_tradclass.eval()
    end = time.time()
    #print('llllllllllllll','located in train_with_inpaint_final.py at 264')
    #exit(0)
    for root, dirs, _ in os.walk('/data4/lijunjie/mini-imagenet-tools/processed_images/train'):
        #for f in files:
            #print(os.path.join(root, f))

        for d in dirs:
            path=os.path.join(root, d)
            path_1=path.replace('train','mask/train_1')
            path_2=path.replace('train','mask/train_2')
            path_3=path.replace('train','mask/train_3')
            path_4=path.replace('train','mask/train_4')
            path_5=path.replace('train','mask/train_full')   
            path_6=path.replace('train','train_full')              
            if not os.path.isdir(path_1):            
                os.mkdir(path_1)
                os.mkdir(path_2)
                os.mkdir(path_3)
                os.mkdir(path_4)
                os.mkdir(path_5)
                os.mkdir(path_6)                
            files = os.listdir(path) 
            #images=[]
            #imgs_gray=[]
            #Xt_img_ori=[]
            Paths=[]
            Paths.append(path_1)
            Paths.append(path_2)
            Paths.append(path_3)
            Paths.append(path_4)  
            Paths.append(path_5)
            Paths.append(path_6)             
            for file in files:
                images=[]
                imgs_gray=[]
                Xt_img_ori=[]            
                img_ori = read_image(os.path.join(path, file))
                #print(file)
                #exit(0)
                masked_img=np.array(img_ori)#*(1-mask_3)+mask_3*255
                masked_img=Image.fromarray(masked_img)
                masked_img_tensor=Funljj.to_tensor(masked_img).float()           
                Xt_img_ori.append(masked_img_tensor)
                img = transform_test(img_ori)
                img_gray = rgb2gray(np.array(img_ori))
                img_gray=Image.fromarray(img_gray)
                img_gray_tensor=Funljj.to_tensor(img_gray).float()            
                imgs_gray.append(img_gray_tensor)                
                images.append(img)
                images = torch.stack(images, dim=0)
                imgs_gray = torch.stack(imgs_gray, dim=0) 
                Xt_img_ori = torch.stack(Xt_img_ori, dim=0)
                if use_gpu:
                    images_train = images.cuda()
                    imgs_gray = imgs_gray.cuda()
                    Xt_img_ori = Xt_img_ori.cuda()
                    
                with torch.no_grad():
                    ytest,feature= model_tradclass(images_train.reshape(1,1,3,84,84), images_train.reshape(1,1,3,84,84),images_train.reshape(1,1,3,84,84), images_train.reshape(1,1,3,84,84))               
                feature_cpu=feature.detach().cpu().numpy()
                probs, idx = ytest.detach().sort(1, True)
                probs = probs.cpu().numpy()
                idx = idx.cpu().numpy() 
        #print(pids)
        #print(idx[:,0,0,0])
        #print(idx.shape)
        #exit(0)
        #print(feature.shape)
        #exit(0)
                masks=[]
                edges=[]
                mask_fuse=0
        #output_cam=[]
                for i in range(feature.shape[0]):
                    CAMs,masks_cpu=returnCAM(feature_cpu[i], weight_softmax, [idx[i,:4,0,0]],masks)
                    #for j in range(4):
                        #print(CAMs[j].shape,CAMs[j].max(),CAMs[j].min(),CAMs[j].sum(),feature.shape[0])
                    #exit(0)
                    masks=CAMs
                    for num_mask in range(len(masks_cpu)):
                        print(len(masks_cpu))
                        #exit(0)
                        cv2.imwrite(Paths[num_mask]+'/'+file, masks_cpu[num_mask])
                    mask_fuse=masks_cpu[0]/255+masks_cpu[1]/255+masks_cpu[2]/255+masks_cpu[3]/255
                    mask_fuse=np.uint8((mask_fuse)>0)*255                        
                    cv2.imwrite(Paths[4]+'/'+file, mask_fuse)
                    #exit(0)
                        
        #print(len(masks),masks[0].shape)
               # masks_tensor = torch.stack(masks, dim=0) 
                masks_tensor=Funljj.to_tensor(Image.fromarray(mask_fuse)).float().reshape(1,1,84,84)
                #print(mask_tensor.shape)
                #exit(0)
                Xt_masks = masks_tensor.reshape(1,1,1,1,84,84)#[:,:,0]
                Xt_img_ori_repeat=Xt_img_ori.reshape(1,1,1,3,84,84)

                Xt_img_ori_repeat = Xt_img_ori_repeat.repeat(1,1,1,1,1,1)    
                Xt_img_gray_repeat=imgs_gray.reshape(1,1,1,1,84,84)

                Xt_img_gray_repeat = Xt_img_gray_repeat.repeat(1,1,1,1,1,1)          
        #print(Xt_img_ori.shape,Xt_masks.shape)
        #exit(0)
                mask_numpy=np.uint8(Xt_masks.numpy()*255)
                print(mask_numpy.shape)
                #exit(0)
                Xt_img_gray_numpy=np.uint8(imgs_gray.cpu().numpy()*255).reshape(1,1,1,84,84)
        #print(Xt_img_gray_numpy.shape)
                for i in range(1):
                    for j in range(1):
                        for k in range(1):
                            edge_PIL=Image.fromarray(load_edge(Xt_img_gray_numpy[i,j,0], mask_numpy[i,j,k,0]))
                            print(mask_numpy[i,j,k,0].sum()/255,'llll')
                            #exit(0)
                            edges.append(Funljj.to_tensor(edge_PIL).float())        
                edges = torch.stack(edges, dim=0) 
                edge_sh=edges#.reshape(4,5,1,84,84)
                #print(edge_sh.shape,Xt_img_gray_repeat.shape,masks_tensor.shape)
                #exit(0)
        #exit(0)        
        #model_edge.test(Xt_img_ori,edge_sh,Xt_img_gray,Xt_masks)
                with torch.no_grad():
                    inpaint_img=model_edge.test(Xt_img_ori_repeat.reshape(1,3,84,84),edge_sh,Xt_img_gray_repeat.reshape(1,1,84,84),masks_tensor)
                inpaint_img_np=inpaint_img.detach().cpu().numpy()
                Xt_img_ori_np=Xt_img_ori_repeat.detach().cpu().numpy()                
                #print(inpaint_img_np.shape)
                for id in range(1):
                    images_temp_train1=inpaint_img_np[id,:,:]
                    Xt_img_ori_repeat1=Xt_img_ori_np.reshape(-1,3,84,84)[id,:,:]
                    #print(Xt_img_ori_repeat1.shape)
            #images_temp_train=images_temp_train1*std+mean
                    images_ori_train=images_temp_train1.transpose((1,2,0))[:,:,::-1]
                    Xt_img_ori_repeat1=Xt_img_ori_repeat1.transpose((1,2,0))[:,:,::-1]
                    images_ori_train=np.uint8(images_ori_train*255)  
                    Xt_img_ori_repeat1=np.uint8(Xt_img_ori_repeat1*255)                    
                    cv2.imwrite(Paths[5]+'/'+file, images_ori_train)     
                    #cv2.imwrite('./result/inpaint_img/'+str(i)+'_'+str(id)+'_ori.jpg', Xt_img_ori_repeat1)                      
    exit(0)                    
                #exit(0)                
            #print(path)
            #print(path_1)
            #print(path_2)
            #print(path_3)
            #print(path_4)            
                #exit(0)            
    for batch_idx, (images_train, labels_train,tpids,Xt_img_ori,Xt_img_gray,images_test, labels_test, pids) in enumerate(trainloader):
    
    #for batch_idx, (images_train, labels_train, images_test, labels_test, pids) in enumerate(trainloader):    
        data_time.update(time.time() - end)
        #print(Xt_img_ori.shape,Xt_img_gray.shape,images_train.shape,'lll')
        edges=[]
        if use_gpu:
            images_train = images_train.cuda()

        batch_size, num_train_examples, channels, height, width = images_train.size()
        num_test_examples = images_test.size(1)
        
        labels_train_1hot = one_hot(labels_train).cuda()
        labels_train_1hot_4 = one_hot(labels_train_4).cuda()        
        #labels_train = labels_train.view(batch_size * num_train_examples)   
        #print( labels_train)
        #exit(0)        
        labels_test_1hot = one_hot(labels_test).cuda()
        labels_test_1hot_4 = one_hot(labels_test_4).cuda()
        #print(labels_test_1hot_4.shape,labels_test_1hot.shape)        
        #labels_test_1hot_4 = torch.cat((labels_test_1hot , labels_test_1hot_4), 1)
        #print(labels_test_1hot.shape,labels_test_1hot_4.shape)
        #exit(0)
        with torch.no_grad():
            ytest,feature= model_tradclass(images_train, images_train, labels_train_1hot, labels_test_1hot)
        #print(ytest.shape)
        #exit(0)
        images_train=images_train.reshape(4,5,1,3,84,84)
        #images_test=images_test.reshape(4,30,1,3,84,84)        
        feature_cpu=feature.detach().cpu().numpy()
        probs, idx = ytest.detach().sort(1, True)
        probs = probs.cpu().numpy()
        idx = idx.cpu().numpy() 
        #print(pids)
        #print(idx[:,0,0,0])
        #print(idx.shape)
        #exit(0)
        #print(feature.shape)
        #exit(0)
        masks=[]
        #output_cam=[]
        for i in range(feature.shape[0]):
            CAMs=returnCAM(feature_cpu[i], weight_softmax, [idx[i,:4,0,0]],masks)
            masks=CAMs
        #print(len(masks),masks[0].shape)
        masks_tensor = torch.stack(masks, dim=0)
        Xt_masks = masks_tensor.reshape(1,1,4,1,84,84)#[:,:,0]
        Xt_img_ori_repeat=Xt_img_ori.reshape(1,1,1,3,84,84)

        Xt_img_ori_repeat = Xt_img_ori_repeat.repeat(1,1,4,1,1,1)    
        Xt_img_gray_repeat=Xt_img_gray.reshape(1,1,1,1,84,84)

        Xt_img_gray_repeat = Xt_img_gray_repeat.repeat(1,1,4,1,1,1)          
        #print(Xt_img_ori.shape,Xt_masks.shape)
        #exit(0)
        mask_numpy=np.uint8(Xt_masks.numpy()*255)
        #print(mask_numpy.shape,Xt_img_gray_numpy.shape)
        Xt_img_gray_numpy=np.uint8(Xt_img_gray.numpy()*255)
        #print(Xt_img_gray_numpy.shape)
        for i in range(1):
            for j in range(1):
                for k in range(4):
                    edge_PIL=Image.fromarray(load_edge(Xt_img_gray_numpy[i,j,0], mask_numpy[i,j,k,0]))
                    edges.append(Funljj.to_tensor(edge_PIL).float())        
        edges = torch.stack(edges, dim=0) 
        edge_sh=edges#.reshape(4,5,1,84,84)
        #exit(0)        
        #model_edge.test(Xt_img_ori,edge_sh,Xt_img_gray,Xt_masks)
        with torch.no_grad():
            inpaint_img=model_edge.test(Xt_img_ori_repeat.reshape(4,3,84,84),edge_sh,Xt_img_gray_repeat.reshape(4,1,84,84),masks_tensor)
        inpaint_img_np=inpaint_img.detach().cpu().numpy()
        for i in range(4):
            images_temp_train1=inpaint_img_np[i,:,:].cpu().numpy()
            #images_temp_train=images_temp_train1*std+mean
            images_ori_train=images_temp_train1.transpose((1,2,0))[:,:,::-1]
            images_ori_train=np.uint8(images_ori_train*255)                 
            cv2.imwrite('./result/inpaint_img/'+str(i)+'_'+str(j)+'_'+str(labels_train_ex[i,j])+'.jpg', images_ori_train)            
        exit(0)
        inpaint_img_np=(inpaint_img_np-mean)/std
        #support set
        inpaint_tensor=torch.from_numpy(inpaint_img_np).cuda().reshape(4,5,4,3,84,84).float()        
        #images_train=torch.cat((images_train, inpaint_tensor), 2).reshape(4,25,3,84,84)#images_train
def test(model_edge, model, model_tradclass,weight_softmax, testloader, use_gpu):
    accs = AverageMeter()
    test_accuracies = []
    std=np.expand_dims(np.array([0.229, 0.224, 0.225]),axis=1)
    std=np.expand_dims(std,axis=2) 
    mean=np.expand_dims(np.array([0.485, 0.456, 0.406]),axis=1)             
    mean=np.expand_dims(mean,axis=2)     
    model.eval()
    model_tradclass.eval()
    with torch.no_grad():
        for batch_idx , (images_train, labels_train,Xt_img_ori,Xt_img_gray, images_test, labels_test) in enumerate(testloader):
            if use_gpu:
                images_train = images_train.cuda()
                images_test = images_test.cuda()

            end = time.time()
            #print(images_train.shape,images_test.shape)
            #exit(0)
            batch_size, num_train_examples, channels, height, width = images_train.size()
            num_test_examples = images_test.size(1)
            labels_train_4 = labels_train.reshape(4,5,1)#[:,:,0]

            labels_train_4 = labels_train_4.repeat(1,1,5).reshape(4,-1) 
            labels_train_4=labels_train_4.cuda()           
            labels_train_1hot = one_hot(labels_train).cuda()
            labels_test_1hot = one_hot(labels_test).cuda()
            labels_train_1hot_4 = one_hot(labels_train_4).cuda()            
            ytest,feature= model_tradclass(images_train, images_train, labels_train_1hot, labels_test_1hot)
        #print(ytest.shape)
        #exit(0)
            images_train=images_train.reshape(4,5,1,3,84,84)
            feature_cpu=feature.detach().cpu().numpy()
            probs, idx = ytest.detach().sort(1, True)
            probs = probs.cpu().numpy()
            idx = idx.cpu().numpy() 
        #print(pids)
        #print(idx[:,0,0,0])
        #print(idx.shape)
        #exit(0)
        #print(feature.shape)
        #exit(0)
            masks=[]
        #output_cam=[]
            for i in range(feature.shape[0]):
                CAMs=returnCAM(feature_cpu[i], weight_softmax, [idx[i,:4,0,0]],masks)
                masks=CAMs
        #print(len(masks),masks[0].shape)
            masks_tensor = torch.stack(masks, dim=0)
            Xt_masks = masks_tensor.reshape(4,5,4,1,84,84)#[:,:,0]
            Xt_img_ori_repeat=Xt_img_ori.reshape(4,5,1,3,84,84)

            Xt_img_ori_repeat = Xt_img_ori_repeat.repeat(1,1,4,1,1,1)    
            Xt_img_gray_repeat=Xt_img_gray.reshape(4,5,1,1,84,84)

            Xt_img_gray_repeat = Xt_img_gray_repeat.repeat(1,1,4,1,1,1)          
            #print(Xt_img_ori.shape,Xt_masks.shape)
        #exit(0)
            edges=[]
            mask_numpy=np.uint8(Xt_masks.numpy()*255)
        #print(mask_numpy.shape,Xt_img_gray_numpy.shape)
            Xt_img_gray_numpy=np.uint8(Xt_img_gray.numpy()*255)
        #print(Xt_img_gray_numpy.shape)
            for i in range(4):
                for j in range(5):
                    for k in range(4):
                        edge_PIL=Image.fromarray(load_edge(Xt_img_gray_numpy[i,j,0], mask_numpy[i,j,k,0]))
                        edges.append(Funljj.to_tensor(edge_PIL).float())        
            edges = torch.stack(edges, dim=0) 
            edge_sh=edges#.reshape(4,5,1,84,84)
        #exit(0)        
        #model_edge.test(Xt_img_ori,edge_sh,Xt_img_gray,Xt_masks)
            inpaint_img=model_edge.test(Xt_img_ori_repeat.reshape(80,3,84,84),edge_sh,Xt_img_gray_repeat.reshape(80,1,84,84),masks_tensor)
            inpaint_img_np=inpaint_img.detach().cpu().numpy()
            inpaint_img_np=(inpaint_img_np-mean)/std
            inpaint_tensor=torch.from_numpy(inpaint_img_np).cuda().reshape(4,5,4,3,84,84).float()
            images_train=torch.cat((images_train, inpaint_tensor), 2).reshape(4,25,3,84,84)
            cls_scores = model(images_train, images_test, labels_train_1hot_4, labels_test_1hot)
            cls_scores = cls_scores.view(batch_size * num_test_examples, -1)
            labels_test = labels_test.view(batch_size * num_test_examples)

            _, preds = torch.max(cls_scores.detach().cpu(), 1)
            acc = (torch.sum(preds == labels_test.detach().cpu()).float()) / labels_test.size(0)
            accs.update(acc.item(), labels_test.size(0))

            gt = (preds == labels_test.detach().cpu()).float()
            gt = gt.view(batch_size, num_test_examples).numpy() #[b, n]
            acc = np.sum(gt, 1) / num_test_examples
            acc = np.reshape(acc, (batch_size))
            test_accuracies.append(acc)

    accuracy = accs.avg
    test_accuracies = np.array(test_accuracies)
    test_accuracies = np.reshape(test_accuracies, -1)
    stds = np.std(test_accuracies, 0)
    ci95 = 1.96 * stds / np.sqrt(args.epoch_size)
    print('Accuracy: {:.2%}, std: :{:.2%}'.format(accuracy, ci95))

    return accuracy


if __name__ == '__main__':
    main()
