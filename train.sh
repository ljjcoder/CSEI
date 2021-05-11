
CUDA_VISIBLE_DEVICES=0 python ./train_with_inpaint_read_from_data_fixed.py --nExemplars 1 --epoch_size 600 --train_nTestNovel 30 --train-batch 4 --nKnovel 5 --Classic 0 --use_similarity 0 \
--save-dir ./result/miniImageNet/CAM/1-shot-seed112_inpaint_support_fuse_Cam_surport_from_65.96_test_fixed/
