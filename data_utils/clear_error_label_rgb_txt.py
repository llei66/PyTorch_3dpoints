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
label_lists = ['PUNKTSKY_1km_6368_561.txt','PUNKTSKY_1km_6368_567.txt']

g_class2color = {'unclassified':	[0,255,0],
                 'ground':	[0,0,255],
                 'low_vegetatation':	[0,255,255],
                 'medium_vegetation':        [255,255,0],
                 'high_vegetataion':      [255,0,255],
                 'building':      [100,100,255],
                 'noise':        [200,0,0]}
for i in label_lists:
    print(i)
    # test_txt = '/data/REASEARCH/DEEPCROP/PointCloudData/20201127/PUNKTSKY_621_51_TIF_UTM32-ETRS89/' + i
    test_txt = '/data/REASEARCH/DEEPCROP/PointCloudData/20201127/building/' + i

    # out_path = '/data/REASEARCH/DEEPCROP/PointCloudData/20201127/true_label_txt_ground_vegetataion_building/'
    out_path = '/data/REASEARCH/DEEPCROP/PointCloudData/20201127/building/clear/'

    # out_txt = out_path +  test_txt.split("/")[-1].replace(".txt","true_label.txt")
    out_txt = out_path +  test_txt.split("/")[-1].replace(".txt","RGB_1.obj")

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
        color = ''
        if (label >= 1 and label <= 7):
            # print(label)
            label_index_0 = label - 1
            if label_index_0 ==0:
                color = '0 255 0'
            if label_index_0 == 1:
                color = '0 0 255'
            if label_index_0 == 2:
                color = '0 255 255'
            if label_index_0 == 3:
                color = '255 255 0'
            if label_index_0 == 4:
                color = '255 0 255'
            if label_index_0 == 5:
                color = '100 100 255'
            if label_index_0 == 6:
                color = '200 0 0'

            items[-1] = str(label_index_0)
            str_temp = "v " + items[0] + " " + items[1] + " " +items[2] + " " + color
            new_line = str(str_temp)
            f_out.write(new_line + "\n")
        else:
            print(label)


    print(out_txt)

