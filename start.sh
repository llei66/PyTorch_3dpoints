data=${1:-./data/train_test_whole_class_1km}
python  train.py  --model pointnet  --log-dir pointnet --batch-size 16  --data-path $data --blocks-per-epoch 2048 --no-rgb
