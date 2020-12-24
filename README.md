# Pytorch Implementation of PointNet and PointNet++ 

This repo is implementation for [PointNet](http://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf) and [PointNet++](http://papers.nips.cc/paper/7095-pointnet-deep-hierarchical-feature-learning-on-point-sets-in-a-metric-space.pdf) in pytorch.


## Environments
Ubuntu 16.04 <br>
Python 3.6.7 <br>
Pytorch 1.1.0

## Predata with Geodata
### convert to TXT data
```buildoutcfg
cd LAStools
sh las2txt.sh .laz .txt
```
### clear and crop Geodata

#### clear data with 1km *1km
```buildoutcfg
cd data_utils/pre_utils_data
python clear_geodata_1.py

```

#### clear and crop Geodata
```buildoutcfg
cd data_utils/pre_utils_data
python crop_geodata_1.py

```
## Semantic Segmentation (with Geodata)
### Run

#### run with 250m * 250m subares
```
sh train_geodata_crop_1.sh
```

#### run with the whole areas
```buildoutcfg
sh train_geodata_1.sh
```
### Test and Visulize
```buildoutcfg
python test_vis_Geodata_crop_1.py
```

## Reference By
[halimacc/pointnet3](https://github.com/halimacc/pointnet3)<br>
[fxia22/pointnet.pytorch](https://github.com/fxia22/pointnet.pytorch)<br>
[charlesq34/PointNet](https://github.com/charlesq34/pointnet) <br>
[charlesq34/PointNet++](https://github.com/charlesq34/pointnet2) <br>
[yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)

## Environments
Ubuntu 16.04 <br>
Python 3.6.7 <br>
Pytorch 1.1.0

## based 
./scripts_config/README.md.config
