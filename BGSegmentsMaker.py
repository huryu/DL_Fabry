#!/usr/bin/env python
# coding: utf-8

import os, sys, glob, pickle, math, argparse
import numpy as np
from skimage import transform, io

parser = argparse.ArgumentParser(usage='BGSegmentsMaker.py -B <Image Directory for input> -BS <Segments Directory for output>')
parser.add_argument('-B',  type=str,  help='Specify <Image Directory for input>')
parser.add_argument('-BS', type=str,  help='Specify <Segments Directory for output>')

args = parser.parse_args()

BGHistEqDirPath = args.B
BGSegmentDir = args.BS

BG_img_list = sorted(glob.glob(BGHistEqDirPath + "/*.npy"))


BG_imgarr_dict = {}

for i, BG_img in enumerate(BG_img_list):
    BG_imgarr_dict[i] = np.load(BG_img)

shape_vert = BG_imgarr_dict[0].shape[0]
shape_hori = BG_imgarr_dict[0].shape[1]
vert_half, hori_half = math.ceil(shape_vert / 8), math.ceil(shape_hori / 8)

# make the number of [(len(BG_img_list)) * 2500] X images of background in the urine sample images of Fabry patient.

while np_count < len(BG_img_list) * 2500:
    i = np.random.randint(0, 25)

    body_centY, body_centX = np.random.randint(shape_vert), np.random.randint(shape_hori)
    rand_angle = np.random.randint(0, 360)
    vert_move = np.random.randint(vert_half * (-1), vert_half)
    hori_move = np.random.randint(hori_half * (-1), vert_half)

    upper_line = body_centY - vert_half + vert_move
    lower_line = body_centY + vert_half + vert_move
    left_line  = body_centX - hori_half + hori_move
    right_line = body_centX + hori_half + hori_move

    if (upper_line >= 0) and (lower_line <= shape_vert) and (left_line >= 0) and (right_line <= shape_hori):
        rotate_img = transform.rotate(BG_imgarr_dict[i], angle= rand_angle, resize=False, center=(body_centX, body_centY))
        f_img =rotate_img[upper_line: lower_line, left_line: right_line]

        color0_set = set([(x, y) for x, y in zip (np.where(f_img[:, :, 0] == 0)[0], np.where(f_img[:, :, 0] == 0)[1])])
        color1_set = set([(x, y) for x, y in zip (np.where(f_img[:, :, 1] == 0)[0], np.where(f_img[:, :, 1] == 0)[1])])
        color2_set = set([(x, y) for x, y in zip (np.where(f_img[:, :, 2] == 0)[0], np.where(f_img[:, :, 2] == 0)[1])])
        color_3chn_black = (color0_set & color1_set) & color2_set

        if len(color_3chn_black) == 0:
            np.save(BGSegmentDir + "/f_imgBG_" + str(i) + "_" + str(np_count) + ".npy", f_img)
            np_count += 1

