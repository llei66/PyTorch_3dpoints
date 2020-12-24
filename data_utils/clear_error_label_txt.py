import os
import sys

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

# label_lists = ['PUNKTSKY_1km_6368_561.txt','PUNKTSKY_1km_6368_567.txt']
label_lists = ['PUNKTSKY_1km_6368_561.txt']


for i in label_lists:
    print(i)
    test_txt = '/data/REASEARCH/DEEPCROP/PointCloudData/20201127/PUNKTSKY_621_51_TIF_UTM32-ETRS89/' + i
    # test_txt = '/data/REASEARCH/DEEPCROP/PointCloudData/20201127/building/' + i

    # out_path = '/data/REASEARCH/DEEPCROP/PointCloudData/20201127/true_label_txt_ground_vegetataion_building/'
    out_path = '/data/REASEARCH/DEEPCROP/PointCloudData/20201218_train_test_set/train_whole_class_200m/origin_ground_vegetataion_building_water/'

    out_txt = out_path +  test_txt.split("/")[-1].replace(".txt","true_label.txt")
    # out_txt = out_path +  test_txt.split("/")[-1].replace(".txt","RGB.txt")

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
        #
        # if (label >= 2 and label <= 6):
        #     # print(label)
        #     label_index_0 = label - 2
        #     items[-1] = str(label_index_0)
        #     str_temp = " "
        #     new_line = str_temp.join(items)
        #     f_out.write(new_line + "\n")
        #
        # if label == 9:
        #     label_index_0 = label - 4
        #     items[-1] = str(label_index_0)
        #     str_temp = " "
        #     new_line = str_temp.join(items)
        #     f_out.write(new_line + "\n")
        # else:
        #     print(label)

        if (label == 2 ):
            label_index_0 = 0
            items[-1] = str(label_index_0)
            str_tmep = " "
            new_line = str_tmep.join(items)
            f_out.write(new_line + "\n")
        if (label == 3 or label == 4 or label ==5 ):
            label_index_0 = 1
            items[-1] = str(label_index_0)
            str_tmep = " "
            new_line = str_tmep.join(items)
            f_out.write(new_line + "\n")
        if (label == 6 ):
            label_index_0 = 2
            items[-1] = str(label_index_0)
            str_tmep = " "
            new_line = str_tmep.join(items)
            f_out.write(new_line + "\n")
        if (label == 9 ):
            label_index_0 = 3
            items[-1] = str(label_index_0)
            str_tmep = " "
            new_line = str_tmep.join(items)
            f_out.write(new_line + "\n")
        else:
            continue
            print(label)

    print(out_txt)

