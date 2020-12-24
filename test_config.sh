#!/bin/bash
#python test_semseg_de_2.py --log_dir pointnet2_sem_seg_Den_3 --test_area 5 --visual --root $1
python test_semseg_de_config.py --log_dir pointnet2_sem_seg_Den_3 --test_area 5 --visual --root $1
