cd ../src

nohup python train.py --device cuda --num_workers 2 --visualize_attention --hidden_size 256 --S 128 --lr 3e-4 --weight_decay 1e-4 --batch_size 32 --epochs 600 --dataset coco --dropout .2 &