CUDA_VISIBLE_DEVICES=$1 python main.py --task 1 --mode train --seed $2
CUDA_VISIBLE_DEVICES=$1 python main.py --task 1 --mode test --seed $2