#!/usr/bin/env python
# coding: utf-8

import os, sys, glob, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from skimage import transform, io

positiveJpgDir = sys.argv[1]
negativeJpgDir = sys.argv[2]

positive_img_list = sorted(glob.glob("IMG_FOLDER/positive_*.jpg"))
negative_img_list = sorted(glob.glob("IMG_FOLDER/negative_*.jpg"))

# Convert jpeg file to numpy array
pos_imgarr_dict = {}
neg_imgarr_dict = {}

i = 0
for pos_img in positive_img_list:
    pos_imgarr_dict[i] = io.imread(pos_img)
    i += 1
    
i = 0
for neg_img in negative_img_list:
    neg_imgarr_dict[i] = io.imread(neg_img)
    i += 1

shape_vert = pos_imgarr_dict[0].shape[0]
shape_hori = pos_imgarr_dict[0].shape[1]

vert_half, hori_half = int(shape_vert / 8), int(shape_hori / 8)
vert_half, hori_half

# center of malberry are decide with direct view.
#shape_vert = 767, shape_hori = 1024

body_cent_Y_X = {}
body_cent_Y_X[0] = (350, 500)
body_cent_Y_X[1] = (370, 470)
body_cent_Y_X[2] = (190, 180)
body_cent_Y_X[3] = (370, 730)
body_cent_Y_X[4] = (405, 470)
body_cent_Y_X[5] = (180, 660)
body_cent_Y_X[6] = (390, 490)


nearest_point = {i :[(0, body_cent_Y_X[i][1]), (shape_vert, body_cent_Y_X[i][1]), (body_cent_Y_X[i][0], 0), (body_cent_Y_X[i][0], shape_hori) ] for i in range(7)}

with open('Image_amplifier_body_cent_point_dict.pkl', mode='wb') as f:
    pickle.dump(body_cent_Y_X, f)

i = 0

body_centY ,body_centX = body_cent_Y_X[i]
rand_angle = np.random.randint(0, 360)
vert_move = np.random.randint(vert_half * (-1), vert_half)
hori_move = np.random.randint(hori_half * (-1), vert_half)

#X for X_axis on math, Y for Y_axis on math
vecX_Y = [(point[1] - body_centX, point[0] - body_centY)  for point in nearest_point[i]]
vecX_Y

#rand_angle = 90, angle move of image and vector_move are reversedirection, so (-1) are multpilied.
rand_pie = (rand_angle / 360) * (np.pi * 2) * (-1)
rotate_matrix = np.array([[np.cos(rand_pie), np.sin(rand_pie) * (-1)], [np.sin(rand_pie), np.cos(rand_pie)]])

# rotate vec on X_Y axis of Math-graph
rot_vecX_vecY = [np.dot (rotate_matrix , np.array(list(vec)))for vec in vecX_Y]

#Convert the point with rotation
rotate_point = [(body_centY + rot_vecX_vecY[i][1], body_centX + rot_vecX_vecY[i][0]) for i in range(4)]
rotate_point

upper_line = body_centY - vert_half + vert_move
lower_line = body_centY + vert_half + vert_move
left_line  = body_centX - hori_half + hori_move
right_line = body_centX + hori_half + hori_move

inner_rotate_point = [point for point in rotate_point if (upper_line <= point[0] <= lower_line) and (left_line <= point[1] <= right_line)]

########################
# United codes revised #
########################

i = np.random.randint(0, 7)

#nearest_point = {i :[(0, body_cent_Y_X[i][1]), (shape_vert, body_cent_Y_X[i][1]), (body_cent_Y_X[i][0], 0), (body_cent_Y_X[i][0], shape_hori) ] for i in range(7)}
#near_dist = min(body_cent_Y_X[i][0], shape_vert - body_cent_Y_X[i][0], body_cent_Y_X[i][1], shape_hori - body_cent_Y_X[i][1])

body_centY, body_centX = body_cent_Y_X[i]
rand_angle = np.random.randint(0, 360)
vert_move = np.random.randint(vert_half * (-1), vert_half)
hori_move = np.random.randint(hori_half * (-1), vert_half)

#画像上での座標の変換
#rotate_point = [(body_centY + rot_vecX_vecY[i][1], body_centX + rot_vecX_vecY[i][0]) for i in range(4)]

upper_line = body_centY - vert_half + vert_move
lower_line = body_centY + vert_half + vert_move
left_line  = body_centX - hori_half + hori_move
right_line = body_centX + hori_half + hori_move

#max_rect_dist = (max(abs(vert_move+vert_half), abs(vert_move-vert_half)) ** 2 + 
#                 max(abs(hori_move+hori_half), abs(hori_move-hori_half)) ** 2) ** (1/2)

#if len(inner_rotate_point) == 0:
#print(near_dist, max_rect_dist)
#if max_rect_dist <= near_dist:


if (upper_line >= 0) and (lower_line <= shape_vert) and (left_line >= 0) and (right_line <= shape_hori):
    rotate_img = transform.rotate(pos_imgarr_dict[i], angle= rand_angle, resize=False, center=(body_centX, body_centY))
    f_img =rotate_img[upper_line: lower_line, left_line: right_line]
    
    color0_set = set([(x, y) for x, y in zip (np.where(f_img[:, :, 0] == 0)[0], np.where(f_img[:, :, 0] == 0)[1])])
    color1_set = set([(x, y) for x, y in zip (np.where(f_img[:, :, 1] == 0)[0], np.where(f_img[:, :, 1] == 0)[1])])
    color2_set = set([(x, y) for x, y in zip (np.where(f_img[:, :, 2] == 0)[0], np.where(f_img[:, :, 2] == 0)[1])])
    color_3chn_black = (color0_set & color1_set) & color2_set
    #len(color_3chn_black)
    #black_point_num = len(np.where(f_img == [0, 0, 0])[0]) + len(np.where(f_img == [0, 0, 0])[1]) + len(np.where(f_img == [0, 0, 0])[2])
    #if black_point_num == 0:
    if len(color_3chn_black) == 0:
        plt.imshow(rotate_img[upper_line: lower_line, left_line: right_line])


# In[846]:


plt.imshow(rotate_img[upper_line: lower_line, left_line: right_line])


# In[744]:


np.where(f_img[:, :, 0] == 0), np.where(f_img[:, :, 1] == 0), np.where(f_img[:, :, 2] == 0)


# In[754]:


color0_set = set([(x, y) for x, y in zip (np.where(f_img[:, :, 0] == 0)[0], np.where(f_img[:, :, 0] == 0)[1])])
color1_set = set([(x, y) for x, y in zip (np.where(f_img[:, :, 1] == 0)[0], np.where(f_img[:, :, 1] == 0)[1])])
color2_set = set([(x, y) for x, y in zip (np.where(f_img[:, :, 2] == 0)[0], np.where(f_img[:, :, 2] == 0)[1])])
color_3chn_black = (color0_set & color1_set) & color2_set

f_img =rotate_img[upper_line: lower_line, left_line: right_line]
black_point_num = len(np.where(f_img == [0, 0, 0])[0]) + len(np.where(f_img == [0, 0, 0])[1]) + len(np.where(f_img == [0, 0, 0])[2])



########################
# United codes revised #
########################
#nearest_point = {i :[(0, body_cent_Y_X[i][1]), (shape_vert, body_cent_Y_X[i][1]), (body_cent_Y_X[i][0], 0), (body_cent_Y_X[i][0], shape_hori) ] for i in range(7)}


#i = np.random.randint(0, 7)
i = 2
print(i)
#i = 0

body_centY, body_centX = body_cent_Y_X[i]
rand_angle = np.random.randint(0, 360)
vert_move = np.random.randint(vert_half * (-1), vert_half)
hori_move = np.random.randint(hori_half * (-1), vert_half)

#X for X_axis on math, Y for Y_axis on math
#vecX_Y = [(point[1] - body_centX, (point[0] - body_centY ) * (-1)) for point in nearest_point[i]]
vecX_Y = [(point[1] - body_centX, point[0] - body_centY)  for point in nearest_point[i]]

#rand_angle = 90, angle move of image and vector_move are reversedirection, so (-1) are multpilied.
rand_pie = (rand_angle / 360) * (np.pi * 2) * (-1)
rotate_matrix = np.array([[np.cos(rand_pie), np.sin(rand_pie) * (-1)], [np.sin(rand_pie), np.cos(rand_pie)]])
# rotate vec on X_Y axis of Math-graph
rot_vecX_vecY = [np.dot (rotate_matrix , np.array(list(vec)))for vec in vecX_Y]

#画像上での座標の変換
rotate_point = [(body_centY + rot_vecX_vecY[i][1], body_centX + rot_vecX_vecY[i][0]) for i in range(4)]

upper_line = body_centY - vert_half + vert_move
lower_line = body_centY + vert_half + vert_move
left_line  = body_centX - hori_half + hori_move
right_line = body_centX + hori_half + hori_move

inner_rotate_point = [point for point in rotate_point if (upper_line <= point[0] <= lower_line) and (left_line <= point[1] <= right_line)]
print(len(inner_rotate_point))

if len(inner_rotate_point) == 0:
    rotate_img = transform.rotate(pos_imgarr_dict[i], angle= rand_angle, resize=False, center=(body_centX, body_centY))
    plt.imshow(rotate_img[upper_line: lower_line, left_line: right_line])



rand_angle = np.random.rand() * 360

img_1 = transform.rotate(pos_imgarr_dict[0], angle= rand_angle, resize=False, center=None)


pos_imgarr_all = np.ones((len(positive_img_list), 767, 1024, 3))
neg_imgarr_all = np.ones((len(negative_img_list), 767, 1024, 3))

for i in range(0, len(positive_img_list)):
    pos_imgarr_all[i] = pos_imgarr_dict[i]

for i in range(0, len(negative_img_list)):
    neg_imgarr_all[i] = neg_imgarr_dict[i]

all_img_arr = np.concatenate((pos_imgarr_all, neg_imgarr_all), axis = 0)
answer_arr = np.concatenate((np.ones(len(positive_img_list)), np.zeros(len(negative_img_list))), axis = 0)

np.save("all_img_arr_urine", all_img_arr)
np.save("answer_arr_urine", answer_arr)





