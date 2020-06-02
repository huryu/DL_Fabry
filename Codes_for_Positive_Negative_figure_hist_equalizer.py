#!/usr/bin/env python
# coding: utf-8

# The code for making images with histogram equalized

import os, sys, glob, pickle, math, shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from skimage import exposure
from skimage import transform, io

positiveJpgDir = sys.argv[1]
negativeJpgDir = sys.argv[3]

# The path for the folder to save Histequalized vectorized image
posiHistEqDir = sys.argv[2]
negaHistEqDir = sys.argv[4]

positive_img_list = sorted(glob.glob(positiveJpgDir + "/*.jpg"))
negative_img_list = sorted(glob.glob(negativeJpgDir + "/*.jpg"))

pos_imgarr_dict = {}
neg_imgarr_dict = {}

for i, pos_img in enumerate(positive_img_list):
    pos_imgarr_dict[i] = io.imread(pos_img)

for i, neg_img in enumerate(negative_img_list):
    neg_imgarr_dict[i] = io.imread(neg_img)

shape_vert = pos_imgarr_dict[0].shape[0]
shape_hori = pos_imgarr_dict[0].shape[1]


vert_half, hori_half = math.ceil(shape_vert / 8), math.ceil(shape_hori / 8)


# For positive image histgram equalization
for i in pos_imgarr_dict.keys():
    original_file_name = positive_img_list[i].split("/")[-1]
    base_file_name = original_file_name.split(".")[0][5:]
    
    img_0 = pos_imgarr_dict[i][:, :, 0]
    img_1 = pos_imgarr_dict[i][:, :, 1]
    img_2 = pos_imgarr_dict[i][:, :, 2]


    # Contrast stretching
    # p2_0, p98_0 = np.percentile(img_0, (2, 98))
    # p2_1, p98_1 = np.percentile(img_1, (2, 98))
    # p2_2, p98_2 = np.percentile(img_2, (2, 98))
    # img_rescale_0 = exposure.rescale_intensity(img_0, in_range=(p2_0, p98_0))
    # img_rescale_1 = exposure.rescale_intensity(img_1, in_range=(p2_1, p98_1))
    # img_rescale_2 = exposure.rescale_intensity(img_2, in_range=(p2_2, p98_2))

    # Equalization
    # img_eq_0 = exposure.equalize_hist(img_0)
    # img_eq_1 = exposure.equalize_hist(img_1)
    # img_eq_2 = exposure.equalize_hist(img_2)

    # Adaptive Equalization
    img_adapteq_0 = exposure.equalize_adapthist(img_0, clip_limit=0.03)
    img_adapteq_1 = exposure.equalize_adapthist(img_1, clip_limit=0.03)
    img_adapteq_2 = exposure.equalize_adapthist(img_2, clip_limit=0.03)

    img_unite = pos_imgarr_dict[i].copy().astype(float)
    img_unite[:, :, 0] = img_adapteq_0
    img_unite[:, :, 1] = img_adapteq_1
    img_unite[:, :, 2] = img_adapteq_2

    np.save(posiHistEqDir + "/" + base_file_name + ".npy", img_unite)

# For negative image histgram equalization

for i in neg_imgarr_dict.keys():
    original_file_name = negative_img_list[i].split("/")[-1]
    base_file_name = original_file_name.split(".")[0]
    
    img_0 = neg_imgarr_dict[i][:, :, 0]
    img_1 = neg_imgarr_dict[i][:, :, 1]
    img_2 = neg_imgarr_dict[i][:, :, 2]


    # Contrast stretching
    # p2_0, p98_0 = np.percentile(img_0, (2, 98))
    # p2_1, p98_1 = np.percentile(img_1, (2, 98))
    # p2_2, p98_2 = np.percentile(img_2, (2, 98))
    # img_rescale_0 = exposure.rescale_intensity(img_0, in_range=(p2_0, p98_0))
    # img_rescale_1 = exposure.rescale_intensity(img_1, in_range=(p2_1, p98_1))
    # img_rescale_2 = exposure.rescale_intensity(img_2, in_range=(p2_2, p98_2))

    # Equalization
    # img_eq_0 = exposure.equalize_hist(img_0)
    # img_eq_1 = exposure.equalize_hist(img_1)
    # img_eq_2 = exposure.equalize_hist(img_2)

    # Adaptive Equalization
    img_adapteq_0 = exposure.equalize_adapthist(img_0, clip_limit=0.03)
    img_adapteq_1 = exposure.equalize_adapthist(img_1, clip_limit=0.03)
    img_adapteq_2 = exposure.equalize_adapthist(img_2, clip_limit=0.03)

    img_unite = neg_imgarr_dict[i].copy().astype(float)
    img_unite[:, :, 0] = img_adapteq_0
    img_unite[:, :, 1] = img_adapteq_1
    img_unite[:, :, 2] = img_adapteq_2

    np.save(negaHistEqDir + "/" + base_file_name + ".npy", img_unite)
