#!/usr/bin/env python
# coding: utf-8

import sys, glob, pickle, math, argparse
import numpy as np
from skimage import exposure

parser = argparse.ArgumentParser(usage='HistEqualizer.py -P <Positive Image Directory for input> -N <Negative Image Directory for input> -PH <Histogram Equalized Positive Image Directory for output> -NH <Histogram Equalized Negative Image Directory for output>')
parser.add_argument('-P', type=str,  help='Specify <Positive Image Directory for input>')
parser.add_argument('-N', type=str,  help='Specify <Negative Image Directory for input>')
parser.add_argument('-PH', type=str, help='Specify <Histogram Equalized Positive Image Directory for output>')
parser.add_argument('-NH', type=str, help='Specify <Histogram Equalized Negative Image Directory for output>')

args = parser.parse_args()

PathForPositiveImageFolder = args.P
PathForNegativeImageFolder = args.N
PathFor_HE_Positive_ImageFolder = args.PH
PathFor_HE_Negative_ImageFolder = args.NH
positive_img_list = sorted(glob.glob(PathForPositiveImageFolder + "/*.jpg")
negative_img_list = sorted(glob.glob(PathForNegativeImageFolder + "/*.jpg")

pos_imgarr_dict = {}
neg_imgarr_dict = {}

for i, pos_img in enumerate(positive_img_list):
    pos_imgarr_dict[i] = io.imread(pos_img)
    
for i, neg_img in enumerate(negative_img_list):
    neg_imgarr_dict[i] = io.imread(neg_img)

shape_vert = neg_imgarr_dict[0].shape[0]
shape_hori = neg_imgarr_dict[0].shape[1]
vert_half, hori_half = math.ceil(shape_vert / 8), math.ceil(shape_hori / 8)

# For positive image histgram equalization

for i in pos_imgarr_dict.keys():
    original_file_name = positive_img_list[i].split("/")[-1]
    base_file_name = original_file_name.split(".")[0]
    
    img_0 = pos_imgarr_dict[i][:, :, 0]
    img_1 = pos_imgarr_dict[i][:, :, 1]
    img_2 = pos_imgarr_dict[i][:, :, 2]

    # Adaptive Equalization
    img_adapteq_0 = exposure.equalize_adapthist(img_0, clip_limit=0.03)
    img_adapteq_1 = exposure.equalize_adapthist(img_1, clip_limit=0.03)
    img_adapteq_2 = exposure.equalize_adapthist(img_2, clip_limit=0.03)

    img_unite = pos_imgarr_dict[i].copy().astype(float)
    img_unite[:, :, 0] = img_adapteq_0
    img_unite[:, :, 1] = img_adapteq_1
    img_unite[:, :, 2] = img_adapteq_2

    np.save(PathFor_HE_Positive_ImageFolder + "/" + base_file_name + ".npy", img_unite)

# For negative image histgram equalization

for i in neg_imgarr_dict.keys():
    original_file_name = negative_img_list[i].split("/")[-1]
    base_file_name = original_file_name.split(".")[0]
    
    img_0 = neg_imgarr_dict[i][:, :, 0]
    img_1 = neg_imgarr_dict[i][:, :, 1]
    img_2 = neg_imgarr_dict[i][:, :, 2]

    # Adaptive Equalization
    img_adapteq_0 = exposure.equalize_adapthist(img_0, clip_limit=0.03)
    img_adapteq_1 = exposure.equalize_adapthist(img_1, clip_limit=0.03)
    img_adapteq_2 = exposure.equalize_adapthist(img_2, clip_limit=0.03)

    img_unite = neg_imgarr_dict[i].copy().astype(float)
    img_unite[:, :, 0] = img_adapteq_0
    img_unite[:, :, 1] = img_adapteq_1
    img_unite[:, :, 2] = img_adapteq_2

    np.save(PathFor_HE_Negative_ImageFolder + "/" + base_file_name + ".npy", img_unite)
