#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""Randomised segmented positive images with histogram equalization."""


# In[2]:


import sys, glob, pickle, math, argparse
import numpy as np
from skimage import transform


# In[3]:


parser = argparse.ArgumentParser(usage='PosNegSegMaker.py -P <Positive Image Directory for input> -N <Negative Image Directory for input> -M <Pickled File of Python3 dictionary object> -PS <Positive Segments Directory for output> -NS <Negative Segments Directory for output>')
parser.add_argument('-P', type=str,  help='Specify <Positive Image Directory>')
parser.add_argument('-N', type=str,  help='Specify <Negative Image Directory>')
parser.add_argument('-M', type=str,  help='Specify <Pickled File of Python3 dictionary object>')
parser.add_argument('-PS', type=str, help='Specify <Positive Segments Directory>')
parser.add_argument('-NS', type=str, help='Specify <Negative Segments Directory>')

args = parser.parse_args()


# In[4]:


PosHistEqDirPath = args.P
NegHistEqDirPath = args.N
DictPathForCenterOfMulberryCells = args.M
PosSegmentDir = args.PS
NegSegmentDir = args.NS

positive_img_list = sorted(glob.glob(PosHistEqDirPath + "/*.npy"))
negative_img_list = sorted(glob.glob(NegHistEqDirPath + "/*.npy"))


# In[ ]:


with open(DictPathForCenterOfMulberryCells, mode='rb') as f:
     body_cent_Y_X = pickle.load(f)
        
# The example of body_cent_Y_X; the values represent the center of mulberry cells (Y, X)  
#body_cent_Y_X = {1: (405, 470),
#                 2: (405, 513),
#                 3: (380, 480),
#                 4: (370, 460),
#                 5: (390, 490),
#                 6: (350, 500),
#                 7: (193, 190),
#                 8: (475, 475),
#                 9: (375, 435),
#                 10: (425, 515),
#                 11: (320, 560),
#                 12: (320, 560),
#                 13: (320, 565),
#                 14: (390, 540),
#                 15: (350, 525),
#                 16: (355, 590),
#                 17: (360, 560),
#                 18: (374, 545),
#                 19: (385, 518),
#                 20: (355, 532),
#                 21: (400, 540),
#                 22: (362, 520),
#                 23: (356, 465),
#                 24: (360, 506),
#                 25: (360, 520)}


# In[ ]:


pos_imgarr_dict = {}
neg_imgarr_dict = {}

for i, pos_img in enumerate(positive_img_list):
    pos_imgarr_dict[i] = np.load(pos_img)
    
for i, neg_img in enumerate(negative_img_list):
    neg_imgarr_dict[i] = np.load(neg_img)


# In[ ]:


shape_vert = pos_imgarr_dict[0].shape[0]
shape_hori = pos_imgarr_dict[0].shape[1]
vert_half, hori_half = math.ceil(shape_vert / 8), math.ceil(shape_hori / 8)


# In[ ]:


# make the number of [(len(positive_img_list)) * 5000] X images of malberry negative segments

np_count = 0

while np_count < len(positive_img_list) * 5000:
    i = np.random.randint(0, 25)

    Mul_centY, Mul_centX = body_cent_Y_X[i+1]
    rand_angle = np.random.randint(0, 360)
    vert_move = np.random.randint(vert_half * (-1), vert_half)
    hori_move = np.random.randint(hori_half * (-1), hori_half)

    upper_line = Mul_centY - vert_half + vert_move
    lower_line = Mul_centY + vert_half + vert_move
    left_line  = Mul_centX - hori_half + hori_move
    right_line = Mul_centX + hori_half + hori_move

    if (upper_line >= 0) and (lower_line <= shape_vert) and (left_line >= 0) and (right_line <= shape_hori):
        rotate_img = transform.rotate(pos_imgarr_dict[i], angle= rand_angle, resize=False, center=(Mul_centX, Mul_centY))
        f_img =rotate_img[upper_line: lower_line, left_line: right_line]
    
        color0_set = set([(x, y) for x, y in zip (np.where(f_img[:, :, 0] == 0)[0], np.where(f_img[:, :, 0] == 0)[1])])
        color1_set = set([(x, y) for x, y in zip (np.where(f_img[:, :, 1] == 0)[0], np.where(f_img[:, :, 1] == 0)[1])])
        color2_set = set([(x, y) for x, y in zip (np.where(f_img[:, :, 2] == 0)[0], np.where(f_img[:, :, 2] == 0)[1])])
        color_3chn_black = (color0_set & color1_set) & color2_set

        if len(color_3chn_black) == 0:
            np.save(PosSegmentDir + "/f_imgP_" + str(i) + "_" + str(np_count), f_img)
            np_count += 1


# In[ ]:


# make the number of [(len(negative_img_list)) * 100] X images of malberry negative segments

np_count = 0
while np_count < (len(negative_img_list)) * 100:

    i = np.random.randint(len(negative_img_list))
    body_centY, body_centX = np.random.randint(shape_vert), np.random.randint(shape_hori)
    rand_angle = np.random.randint(0, 360)
    vert_move = np.random.randint(vert_half * (-1), vert_half)
    hori_move = np.random.randint(hori_half * (-1), vert_half)

    upper_line = body_centY - vert_half + vert_move
    lower_line = body_centY + vert_half + vert_move
    left_line  = body_centX - hori_half + hori_move
    right_line = body_centX + hori_half + hori_move

    if (upper_line >= 0) and (lower_line <= shape_vert) and (left_line >= 0) and (right_line <= shape_hori):
        rotate_img = transform.rotate(neg_imgarr_dict[i], angle= rand_angle, resize=False, center=(body_centX, body_centY))
        f_img =rotate_img[upper_line: lower_line, left_line: right_line]
    
        color0_set = set([(x, y) for x, y in zip (np.where(f_img[:, :, 0] == 0)[0], np.where(f_img[:, :, 0] == 0)[1])])
        color1_set = set([(x, y) for x, y in zip (np.where(f_img[:, :, 1] == 0)[0], np.where(f_img[:, :, 1] == 0)[1])])
        color2_set = set([(x, y) for x, y in zip (np.where(f_img[:, :, 2] == 0)[0], np.where(f_img[:, :, 2] == 0)[1])])
        color_3ch_black = (color0_set & color1_set) & color2_set

        if len(color_3ch_black) == 0:
            np.save(NegSegmentDir + "/f_imgN_" + str(i) + "_" + str(np_count), f_img)
            np_count += 1

