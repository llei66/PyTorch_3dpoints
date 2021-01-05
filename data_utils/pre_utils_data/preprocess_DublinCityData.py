import os
import sys
import pdb
import numpy as np
#classes = ['ground','low_vegetatation','medium_vegetation','high_vegetataion','building','water']

file_list = [
    'T_315500_234500_NW.bin',
    'T_315500_233500_NE_T_315500_234000_SE.bin',
    'T_315500_234500_NE.bin'
]

# list_path = '/home/SENSETIME/lilei/DEEPCROP/DublinCity/sample/T_315500_233500_NE_T_315500_234000_SE.bin/Vegetation'

for j in file_list:

    # list_path = '/home/SENSETIME/lilei/DEEPCROP/DublinCity/sample/' + str(j) +'/vegetation'
    # out_path = list_path.replace("vegetation", "vegetation_label")
    list_path = '/home/SENSETIME/lilei/DEEPCROP/DublinCity/sample/' + str(j) +'/ground'
    out_path = list_path.replace("ground", "ground_label")
    # list_path = '/home/SENSETIME/lilei/DEEPCROP/DublinCity/sample/' + str(j) + '/building'
    # out_path = list_path.replace("building", "building_label")
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    label_lists = os.listdir(list_path)
    for i in label_lists:
        print(i)
        test_txt = os.path.join(list_path , i)
        # out_path = '/data/REASEARCH/DEEPCROP/PointCloudData/20201127/true_label_txt_ground_vegetataion_building/'
        # out_path = '/data/REASEARCH/DEEPCROP/PointCloudData/20201218_train_test_set/train_test_whole_class_1km/train/'
        out_put_name = str(j) + "_" + i+"_label.txt"
        # out_txt = out_path +  test_txt.split("/")[-1].replace(".txt","true_label.txt")
        out_txt = out_path + "/" + out_put_name

        f_out = open(out_txt, 'w')

        # t_x = np.loadtxt(test_txt)
        # # pdb.set_trace()
        # x_max = t_x[:,0].max()
        # x_min = t_x[:,0].min()
        # y_max = t_x[:,1].max()
        # y_min = t_x[:,1].min()
        #
        # count=0

        with open(test_txt, 'r') as f:
            lines = f.readlines()

        for line in lines:
            # print(line)
            items = line.strip("\n").split(" ")
            # pdb.set_trace()
            label = ' 0'
            str_tmep = " "
            new_line = str_tmep.join(items[0:6])
            new_line = new_line + label
            f_out.write(new_line + "\n")



        print(out_txt)

