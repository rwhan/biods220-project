CUDA_VISIBLE_DEVICES=0 python train.py --forward 2 --trend 1
CUDA_VISIBLE_DEVICES=1 python train.py --forward 2 --dropout 0.3
CUDA_VISIBLE_DEVICES=2 python train.py --trend 1 --dropout 0.3
CUDA_VISIBLE_DEVICES=3 python train.py --forward 2 --trend 1 --dropout 0.3
