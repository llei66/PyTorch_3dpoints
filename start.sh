data=${1:-./data/train_test_whole_class_1km}

# train model
python train.py  --model pointnet  --log-dir ./log --batch-size 8  --data-path $data --blocks-per-epoch 5000 --no-rgb --points-per-sample 10000 --epoch 100

# test model again and save predictions
python test.py  --model pointnet  --log-dir ./log --batch-size 1  --data-path $data --no-rgb
