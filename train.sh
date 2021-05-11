#python train.py --nExemplars 1 --nTestNovel 1 --test-batch 1 --gpu-devices 2 --train-batch 2 --nKnovel 1
#python train.py --use_similarity True --save-dir ./result/miniImageNet/CAM/5-shot-seed112_similarity/ --gpu-devices 1
#CUDA_VISIBLE_DEVICES=1 python train.py --nExemplars 1 --save-dir ./result/miniImageNet/CAM/1-shot-seed112_notusesimilarity/
#CUDA_VISIBLE_DEVICES=1 python train_with_inpaint_final.py --nExemplars 1 --train_nTestNovel 30 --train-batch 4 --nKnovel 5 --Classic 0 --use_similarity 0 \
#--save-dir ./result/miniImageNet/CAM/1-shot-seed112_inpaint_use_CAM_argumenttest/
#CUDA_VISIBLE_DEVICES=2 python test_debug.py --nExemplars 1 --train_nTestNovel 30 --train-batch 4 --nKnovel 5 --Classic 0 --use_similarity 0 \
#--save-dir ./result/miniImageNet/CAM/1-shot-seed112_inpaint_use_CAM_test_ori/
#CUDA_VISIBLE_DEVICES=3 python train_multi_zhixin.py --nExemplars 1 --train_nTestNovel 30 --train-batch 4 --nKnovel 5 --Classic 0 --use_similarity 0 \
--save-dir ./result/miniImageNet/CAM/1-shot-seed112_inpaint_use_CAM_multizhixin_cls_1/
#CUDA_VISIBLE_DEVICES=3 python train_with_inpaint_read_from_data.py --nExemplars 5 --train_nTestNovel 30 --train-batch 4 --nKnovel 5 --Classic 0 --use_similarity 0 \
--save-dir ./result/miniImageNet/CAM/5-shot-seed112_inpaint_use_regular_retrain/
#CUDA_VISIBLE_DEVICES=1 python train_inpainting_trainargument_64.99.py --nExemplars 5 --train_nTestNovel 30 --train-batch 4 --nKnovel 5 --Classic 0 --use_similarity 0 \
--save-dir ./result/miniImageNet/CAM/5-shot-seed112_inpaint_use_gmcnn/
#CUDA_VISIBLE_DEVICES=0 python ./train_with_inpaint_read_from_data.py --nExemplars 1 --epoch_size 600 --train_nTestNovel 30 --train-batch 4 --nKnovel 5 --Classic 0 --use_similarity 0 \
--save-dir ./result/miniImageNet/CAM/1-shot-seed112_inpaint_support_fuse_Cam_surport_from_64.99_test_random/ --seed 4

#CUDA_VISIBLE_DEVICES=3 python ./train_tradi_net.py --nExemplars 1 --train_nTestNovel 30 --train-batch 4 --nKnovel 5 --Classic 0 --use_similarity 0 \
--save-dir ./result/miniImageNet/CAM/tired_images_ave_class/
#CUDA_VISIBLE_DEVICES=2 python ./tired_with_inpaint.py --nExemplars 1 --train_nTestNovel 30 --train-batch 4 --nKnovel 5 --Classic 0 --use_similarity 0 \
--save-dir ./result/tiredimg/CAM/1-shot-seed112_inpaint_8argument/
#CUDA_VISIBLE_DEVICES=2 python ./train_with_inpaint_read_from_data_fixed.py --nExemplars 1 --epoch_size 600 --train_nTestNovel 30 --train-batch 4 --nKnovel 5 --Classic 0 --use_similarity 0 \
--save-dir ./result/miniImageNet/CAM/1-shot-seed112_inpaint_support_fuse_Cam_surport_from_64.99_test_fixed_GPU0/  --seed 2111
#CUDA_VISIBLE_DEVICES=2 python ./train_with_inpaint_read_from_data_fixed.py --nExemplars 1 --epoch_size 600 --train_nTestNovel 30 --train-batch 4 --nKnovel 5 --Classic 0 --use_similarity 0 \
--save-dir ./result/miniImageNet/CAM/1-shot-seed112_inpaint_support_fuse_Cam_surport_from_64.99_test_fixed_GPU0/  --seed 22222
CUDA_VISIBLE_DEVICES=0 python ./train_with_inpaint_read_from_data_fixed.py --nExemplars 1 --epoch_size 600 --train_nTestNovel 30 --train-batch 4 --nKnovel 5 --Classic 0 --use_similarity 0 \
--save-dir ./result/miniImageNet/CAM/1-shot-seed112_inpaint_support_fuse_Cam_surport_from_65.96_test_fixed_GPU0_2333/  --seed 2333
#CUDA_VISIBLE_DEVICES=2 python ./train_with_inpaint_read_from_data_fixed.py --nExemplars 1 --epoch_size 600 --train_nTestNovel 30 --train-batch 4 --nKnovel 5 --Classic 0 --use_similarity 0 \
--save-dir ./result/miniImageNet/CAM/1-shot-seed112_inpaint_support_fuse_Cam_surport_from_64.99_test_fixed_GPU0/ --seed 2444
#CUDA_VISIBLE_DEVICES=2 python ./train_with_inpaint_read_from_data_fixed.py --nExemplars 1 --epoch_size 600 --train_nTestNovel 30 --train-batch 4 --nKnovel 5 --Classic 0 --use_similarity 0 \
--save-dir ./result/miniImageNet/CAM/1-shot-seed112_inpaint_support_fuse_Cam_surport_from_64.99_test_fixed_GPU0/ --seed 2555
#CUDA_VISIBLE_DEVICES=2 python ./train_with_inpaint_read_from_data_fixed.py --nExemplars 1 --epoch_size 600 --train_nTestNovel 30 --train-batch 4 --nKnovel 5 --Classic 0 --use_similarity 0 \
--save-dir ./result/miniImageNet/CAM/1-shot-seed112_inpaint_support_fuse_Cam_surport_from_64.99_test_fixed_GPU0/ --seed 2666
#CUDA_VISIBLE_DEVICES=2 python ./train_with_inpaint_read_from_data_fixed.py --nExemplars 1 --epoch_size 600 --train_nTestNovel 30 --train-batch 4 --nKnovel 5 --Classic 0 --use_similarity 0 \
--save-dir ./result/miniImageNet/CAM/1-shot-seed112_inpaint_support_fuse_Cam_surport_from_64.99_test_fixed_GPU0/ --seed 2777