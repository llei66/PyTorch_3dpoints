import os
import sys
# import numpy as np
import random

#
# label_lists = ['PUNKTSKY_1km_6210_510_RGB.txt', 'PUNKTSKY_1km_6210_511_RGB.txt', \
#                'PUNKTSKY_1km_6210_512_RGB.txt', 'PUNKTSKY_1km_6210_513_RGB.txt', \
#                'PUNKTSKY_1km_6210_514_RGB.txt', 'PUNKTSKY_1km_6210_515_RGB.txt']

# label_lists = [
#     'PUNKTSKY_1km_6210_516_RGB.txt',
# 'PUNKTSKY_1km_6210_517_RGB.txt',
# 'PUNKTSKY_1km_6210_518_RGB.txt',
# 'PUNKTSKY_1km_6210_519_RGB.txt',
#                'PUNKTSKY_1km_6211_510_RGB.txt',
#                'PUNKTSKY_1km_6211_511_RGB.txt']
# classes = ['ground','low_vegetatation','medium_vegetation','high_vegetataion','building','water']


label_lists = ['PUNKTSKY_1km_6353_546.txt']

for i in label_lists:
    print(i)
    test_txt = '/home/SENSETIME/lilei/DEEPCROP/6350/' + i
    # test_txt = '/data/REASEARCH/DEEPCROP/PointCloudData/20201127/building/' + i

    # out_path = '/data/REASEARCH/DEEPCROP/PointCloudData/20201127/true_label_txt_ground_vegetataion_building/'
    out_path = '/home/SENSETIME/lilei/DEEPCROP/6350/'

    out_txt = out_path +  test_txt.split("/")[-1].replace(".txt","true_label.txt")
    # out_txt = out_path +  test_txt.split("/")[-1].replace(".txt","RGB.txt")

    #with open(out_txt, 'w') as f_out
    f_out = open(out_txt, 'w')

    with open(test_txt, 'r') as f:
        lines = f.readlines()

    for line in lines:
        # print(line)
        items = line.strip("\n").split(" ")
        label = int(items[-1])
        # if (label >= 1) and (label <=7):
        #     f_out.write(line)
        # else:
        #     print(label)

        if (label >= 2 and label <= 6):
            # print(label)

            label_index_0 = label - 2
            if label ==6:
                items[-1] = str(label_index_0)
                str_temp = " "
                new_line = str_temp.join(items)
                f_out.write(new_line + "\n")
            else:
                items[-1] = str(random.randint(0,5))
                str_temp = " "
                new_line = str_temp.join(items)
                f_out.write(new_line + "\n")
        if label == 9:
            label_index_0 = label - 4
            items[-1] = str(label_index_0)
            str_temp = " "
            new_line = str_temp.join(items)
            f_out.write(new_line + "\n")
        else:
            print(label)


    print(out_txt)

