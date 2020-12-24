import os
import sys
#from indoor3d_util import DATA_PATH, collect_point_label
from denmark3d_util import DATA_PATH, collect_point_label
import numpy as np
#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#ROOT_DIR = os.path.dirname(BASE_DIR)
#sys.path.append(BASE_DIR)
#
#anno_paths = [line.rstrip() for line in open(os.path.join(BASE_DIR, 'meta/anno_paths.txt'))]
#anno_paths = [os.path.join(DATA_PATH, p) for p in anno_paths]
#
#output_folder = os.path.join(ROOT_DIR, 'data/stanford_indoor3d')
#if not os.path.exists(output_folder):
#    os.mkdir(output_folder)
#
## Note: there is an extra character in the v1.2 data in Area_5/hallway_6. It's fixed manually.
#for anno_path in anno_paths:
#    print(anno_path)
#    try:
#        elements = anno_path.split('/')
#        out_filename = elements[-3]+'_'+elements[-2]+'.npy' # Area_1_hallway_1.npy
#        collect_point_label(anno_path, os.path.join(output_folder, out_filename), 'numpy')
#    except:
#        print(anno_path, 'ERROR!!')

#test_path = '/data/REASEARCH/DEEPCROP/code/PYcode/point++_test/Pointnet_Pointnet2_pytorch/data/Detest/200w_PUNKTSKY_1km_6210_510.txt' 
#
#out_filename = '/data/REASEARCH/DEEPCROP/code/PYcode/point++_test/Pointnet_Pointnet2_pytorch/data/Detest/200w_PUNKTSKY_1km_6210_510.npy'

#test_path = '/data/REASEARCH/DEEPCROP/code/PYcode/point++_test/Pointnet_Pointnet2_pytorch/data/De_1/100w_PUNKTSKY_1km_6210_520_rgb.txt' 
#out_filename = '/data/REASEARCH/DEEPCROP/code/PYcode/point++_test/Pointnet_Pointnet2_pytorch/data/De_1/100w_PUNKTSKY_1km_6210_520_rgb.npy'
#test_path = '/data/REASEARCH/DEEPCROP/code/PYcode/point++_test/Pointnet_Pointnet2_pytorch/data/Detest/100w_PUNKTSKY_1km_6210_520_rgb.txt' 
#out_filename = '/data/REASEARCH/DEEPCROP/code/PYcode/point++_test/Pointnet_Pointnet2_pytorch/data/De_1/100w_PUNKTSKY_1km_6210_520_rgb.npy'

#test = np.loadtxt(test_path)
#np.save(out_filename, test)
file_list = '/data/REASEARCH/DEEPCROP/PointCloudData/20201127/true_label_txt_list_1.txt'
output_root = '/data/REASEARCH/DEEPCROP/PointCloudData/20201127/PUNKTSKY_621_51_TIF_UTM32-ETRS89_clear_npy/'

with open (file_list) as f:
    lines = f.readlines()
for line in lines:

    test_path = line.strip("\n")
    filename = test_path.split("/")[-1]
    out_filename = output_root + filename
    test = np.loadtxt(test_path)
    np.save(out_filename, test)


#open(test_path) as f:
#    lines = f.readlines()
#fout = open(out_filename, 'w')
#
#for line in lines:
        
