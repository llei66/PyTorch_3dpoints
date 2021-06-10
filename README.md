# Pytorch Implementation of PointNet and PointNet++

### Introduction
```
This branch is to simplify the whole pipeline. 
We add the Geodata and dublindata training, especially the Geodata.

git branch -m test_ll pointnet_pointnet2
git fetch origin
git branch -u origin/pointnet_pointnet2 pointnet_pointnet2
git remote set-head origin -a
```
###  Environment

Pytorch, python3

### Way of training and testing

- sh start.sh
- Details
```buildoutcfg
1. Give the training, testing, validating data path

Please go to ERDA: DeepCrop/Datasets/GeoData/202102_train_test_val_set_whole_class/

2. python train.py  --model pointnet  --log-dir ./log --batch-size 12  --data-path $data --blocks-per-epoch 1200 --no-rgb --points-per-sample 10000 --epoch 100 --lr 3e-4

3. Please check the scripys to get the detail of parameters 

```

### Way of testing and Visualization
```buildoutcfg
1. python test.py  --model pointnet  --log-dir ./log --batch-size 1  --data-path $data --no-rgb
```

### Other
##### 1. we add the Dynamic Graph CNN(dgcnn) [Classfication] to the models, we will continue to add more architectures
##### 2. we add the the Dublin datasets benchmark to the datasets.
```buildoutcfg
Please go to ERDA: DeepCrop/Datasets/DublinCityData/
```
