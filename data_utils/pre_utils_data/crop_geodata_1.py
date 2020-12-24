import os
import sys
import pdb
import numpy as np
#classes = ['ground','low_vegetatation','medium_vegetation','high_vegetataion','building','water']

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

label_lists = [
#     'PUNKTSKY_1km_6210_516_RGB.txt',
# 'PUNKTSKY_1km_6210_517_RGB.txt',
# 'PUNKTSKY_1km_6210_518_RGB.txt',
# 'PUNKTSKY_1km_6210_519_RGB.txt',
#                'PUNKTSKY_1km_6211_510_RGB.txt',
#                'PUNKTSKY_1km_6211_511_RGB.txt',
#     'PUNKTSKY_1km_6368_561.txt',
#     'PUNKTSKY_1km_6368_567.txt',
# 'PUNKTSKY_1km_6210_510_RGB.txt',
#     'PUNKTSKY_1km_6210_511_RGB.txt',
               'PUNKTSKY_1km_6210_512_RGB.txt',
    'PUNKTSKY_1km_6210_513_RGB.txt',
               'PUNKTSKY_1km_6210_514_RGB.txt',
    'PUNKTSKY_1km_6210_515_RGB.txt'
]

# label_lists = ['PUNKTSKY_1km_6368_561true_label.txt']
# label_lists = ['PUNKTSKY_1km_6368_561.txt']
# label_lists = ['PUNKTSKY_1km_6220_720.txt']

for j in label_lists:
    print(j)
    test_txt = '/data/REASEARCH/DEEPCROP/PointCloudData/20201127/PUNKTSKY_621_51_TIF_UTM32-ETRS89/' + j
    # test_txt = '/home/SENSETIME/lilei/DEEPCROP/Class_PUNKTSKY_6220_720/' + i
    # test_txt = '/data/REASEARCH/DEEPCROP/PointCloudData/20201218_train_test_set/train_whole_class_200m/origin_ground_vegetataion_building_water/' + i
    # test_txt = '/data/REASEARCH/DEEPCROP/PointCloudData/20201127/PUNKTSKY_621_51_TIF_UTM32-ETRS89/' + i
    # out_path = '/data/REASEARCH/DEEPCROP/PointCloudData/20201127/true_label_txt_ground_vegetataion_building/'
    # out_path = '/data/REASEARCH/DEEPCROP/PointCloudData/20201218_train_test_set/train_test_whole_class_200m/test_one_area/'

    out_path = '/home/SENSETIME/lilei/DEEPCROP/origin_ground_vegetataion_building_water/sub_areas/'

    #with open(out_txt, 'w') as f_out

    t_x = np.loadtxt(test_txt)
    # pdb.set_trace()
    x_max = t_x[:,0].max()
    x_min = t_x[:,0].min()
    y_max = t_x[:,1].max()
    y_min = t_x[:,1].min()

    count=0

    with open(test_txt, 'r') as f:
        lines = f.readlines()

    # x_internal = (x_max -x_min) / 250
    # y_internal = (y_max -y_min) / 250
    x_internal =  250
    y_internal = 250

    NUM = int(x_max/x_internal)

    X_local_1 = x_min
    Y_local_1 = y_min

    X_local = x_min

    ### the tranigle
    for i in range(NUM):

        X_local = X_local_1 + x_internal
        Y_local = Y_local_1 + y_internal
        if count >= 5:
            continue
        count += 1
        out_str = "_" + str(count) + "_crop.txt"
        out_txt = out_path + test_txt.split("/")[-1].replace(".txt", out_str)
        f_out = open(out_txt, 'w')

        for line in lines:
            if line.strip() == '':
                break
            # print(line)
            items = line.strip("\n").split(" ")
            label = int(items[-1])
            # pdb.set_trace()
            if X_local_1 <= float(items[0]) <= float(X_local) and Y_local_1 <= float(items[1]) <= Y_local:

                if (label == 2):
                    label_index_0 = 0
                    items[-1] = str(label_index_0)
                    str_tmep = " "
                    new_line = str_tmep.join(items)
                    f_out.write(new_line + "\n")
                if (label == 3 or label == 4 or label == 5):
                    label_index_0 = 1
                    items[-1] = str(label_index_0)
                    str_tmep = " "
                    new_line = str_tmep.join(items)
                    f_out.write(new_line + "\n")
                if (label == 6):
                    label_index_0 = 2
                    items[-1] = str(label_index_0)
                    str_tmep = " "
                    new_line = str_tmep.join(items)
                    f_out.write(new_line + "\n")
                if (label == 9):
                    label_index_0 = 3
                    items[-1] = str(label_index_0)
                    str_tmep = " "
                    new_line = str_tmep.join(items)
                    f_out.write(new_line + "\n")
                else:
                    continue
                    print(label)

        X_local_1 = X_local_1 + x_internal
        Y_local_1 = Y_local_1 + y_internal

        print(out_txt)
    break
    # for i in range(NUM):
    #     X_local = X_local + x_internal
    #     # pdb.set_trace()
    #
    #     for j in range(NUM):
    #         # X_local = X_local_1 + x_internal
    #         Y_local = Y_local_1 + y_internal
    #         count += 1
    #         out_str = "_" + str(count) + "_crop.txt"
    #         out_txt = out_path + test_txt.split("/")[-1].replace(".txt", out_str)
    #         f_out = open(out_txt, 'w')
    #
    #         for line in lines:
    #             # print(line)
    #             items = line.strip("\n").split(" ")
    #             label = int(items[-1])
    #             # pdb.set_trace()
    #             if Y_local_1 >= y_max:
    #                 break
    #
    #             if X_local_1 <= float(items[0]) <= float(X_local) and Y_local_1 <= float(items[1]) <= Y_local:
    #
    #                 if (label >= 2 and label <= 6):
    #                     # print(label)
    #                     label_index_0 = label - 2
    #                     items[-1] = str(label_index_0)
    #                     str_temp = " "
    #                     new_line = str_temp.join(items)
    #                     f_out.write(new_line + "\n")
    #                 if label == 9:
    #                     label_index_0 = label - 4
    #                     items[-1] = str(label_index_0)
    #                     str_temp = " "
    #                     new_line = str_temp.join(items)
    #                     f_out.write(new_line + "\n")
    #                 else:
    #                     print(label)
    #                 #
    #                 # print(X_local_1, X_local, Y_local_1,Y_local )
    #             if Y_local >= y_max:
    #                 continue
    #
    #         Y_local_1 = Y_local_1 + y_internal
    #         print(out_txt)
    #
    #         if X_local >= x_max:
    #             continue
    #
    #     # X_local = X_local + x_internal
    #     X_local_1 = X_local_1 + x_internal
    #



### the tranigle
    # for i in range(19):
    #
    #     X_local = X_local_1 + x_internal
    #     Y_local = Y_local_1 + y_internal
    #
    #
    #     count += 1
    #     out_str = "_" + str(count) + "_crop.txt"
    #     out_txt = out_path + test_txt.split("/")[-1].replace(".txt", out_str)
    #     f_out = open(out_txt, 'w')
    #
    #     for line in lines:
    #         # print(line)
    #         items = line.strip("\n").split(" ")
    #         label = int(items[-1])
    #         # pdb.set_trace()
    #         if X_local_1 <= float(items[0]) <= float(X_local) and Y_local_1<= float(items[1]) <= Y_local:
    #
    #             if (label == 2):
    #                 label_index_0 = 0
    #                 items[-1] = str(label_index_0)
    #                 str_tmep = " "
    #                 new_line = str_tmep.join(items)
    #                 f_out.write(new_line + "\n")
    #             if (label == 3 or label == 4 or label == 5):
    #                 label_index_0 = 1
    #                 items[-1] = str(label_index_0)
    #                 str_tmep = " "
    #                 new_line = str_tmep.join(items)
    #                 f_out.write(new_line + "\n")
    #             if (label == 6):
    #                 label_index_0 = 2
    #                 items[-1] = str(label_index_0)
    #                 str_tmep = " "
    #                 new_line = str_tmep.join(items)
    #                 f_out.write(new_line + "\n")
    #             if (label == 9):
    #                 label_index_0 = 3
    #                 items[-1] = str(label_index_0)
    #                 str_tmep = " "
    #                 new_line = str_tmep.join(items)
    #                 f_out.write(new_line + "\n")
    #             else:
    #                 continue
    #                 print(label)
    #     X_local_1 = X_local_1 + x_internal
    #     Y_local_1 = Y_local_1 + y_internal
    #
    #     print(out_txt)

