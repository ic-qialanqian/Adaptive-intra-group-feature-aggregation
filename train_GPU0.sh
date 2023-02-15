CUDA_VISIBLE_DEVICES=0 python train.py --model AIGANet --loss DSLoss_IoU_noCAM --trainset DUTS_class --size 224 --tmp tmp/sizes_statistics --lr 1e-4 --bs 16 --epochs 1
