CUDA_VISIBLE_DEVICES=0 python test.py --model GCoNet --param_root tmp/dot_product --save_root ./data/dot_product
CUDA_VISIBLE_DEVICES=0 python eval-co-sod/main.py --pred_dir ./data/dot_product --gt_dir ./data/gts
