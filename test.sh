CUDA_VISIBLE_DEVICES=0 python test.py --model AIGANet --param_root tmp/AIGANet --save_root ./data/AIGANet
CUDA_VISIBLE_DEVICES=0 python eval-co-sod/main.py --pred_dir ./data/AIGANet --gt_dir ./data/gts
