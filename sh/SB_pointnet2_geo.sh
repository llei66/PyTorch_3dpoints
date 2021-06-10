#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=10 --mem=32GB#SBATCH --job-name="Mosaic"
# we run on the gpu partition and we allocate 1 titanx gpu
#SBATCH -p gpu --gres=gpu:titanrtx:1#SBATCH --time=10:00:00
#SBATCH --mail-type=END,FAIL#SBATCH --mail-user=stefan.oehmcke@di.ku.dk
#SBATCH --exclude a00610,a00757
#your script, in this case: write the hostname and the ids of the chosen gpus.
hostname
echo $CUDA_VISIBLE_DEVICES
poetry run python train.py task=segmentation model_type=pointnet2 model_name=pointnet2_charlesss dataset=denmark
