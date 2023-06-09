cd ../src

nohup python train.py --device cuda --num_workers 2 --visualize_attention --hidden_size 256 --S 16 --lr 3e-5 --weight_decay 0 --batch_size 16 --no_eval &
nohup python train.py --device cuda --num_workers 2 --visualize_attention --hidden_size 256 --S 32 --lr 3e-5 --weight_decay 0 --batch_size 16 --no_eval &
nohup python train.py --device cuda --num_workers 2 --visualize_attention --hidden_size 256 --S 64 --lr 3e-5 --weight_decay 0 --batch_size 16 --no_eval &
nohup python train.py --device cuda --num_workers 2 --visualize_attention --hidden_size 256 --S 128 --lr 3e-5 --weight_decay 0 --batch_size 16 --no_eval &

nohup python train.py --device cuda --num_workers 2 --visualize_attention --hidden_size 512 --S 16 --lr 3e-5 --weight_decay 0 --batch_size 16 --no_eval &
nohup python train.py --device cuda --num_workers 2 --visualize_attention --hidden_size 512 --S 32 --lr 3e-5 --weight_decay 0 --batch_size 16 --no_eval &
nohup python train.py --device cuda --num_workers 2 --visualize_attention --hidden_size 512 --S 64 --lr 3e-5 --weight_decay 0 --batch_size 16 --no_eval &
nohup python train.py --device cuda --num_workers 2 --visualize_attention --hidden_size 512 --S 128 --lr 3e-5 --weight_decay 0 --batch_size 16 --no_eval &

nohup python train.py --device cuda --num_workers 2 --visualize_attention --hidden_size 1024 --S 16 --lr 3e-5 --weight_decay 0 --batch_size 16 --no_eval &
nohup python train.py --device cuda --num_workers 2 --visualize_attention --hidden_size 1024 --S 32 --lr 3e-5 --weight_decay 0 --batch_size 16 --no_eval &
nohup python train.py --device cuda --num_workers 2 --visualize_attention --hidden_size 1024 --S 64 --lr 3e-5 --weight_decay 0 --batch_size 16 --no_eval &
nohup python train.py --device cuda --num_workers 2 --visualize_attention --hidden_size 1024 --S 128 --lr 3e-5 --weight_decay 0 --batch_size 16 --no_eval &