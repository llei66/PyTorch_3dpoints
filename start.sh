data=${1:-./data/train_test_whole_class_1km}
python  train.py  --model pointnet2msg  --log-dir pointnet2msg --batch-size 16  --data-path $data --blocks-per-epoch 2048
