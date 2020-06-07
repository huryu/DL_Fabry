#!/usr/bin/env python
# coding: utf-8

# This code randomly divides negative image segments into training or test data.
# The data in training is used to train models.
# The data in test dataset is used for hyper-parameter tuning and confirmation of learning progression.

# How to use:
# RandomDivider.py <Directory Path for adopted histogram equalized non-segmented image files as Input>

import os, sys, glob, shutil
import numpy as np

# The list is for the adopted histogram equalized non-segmented images.
negativeHEimgDir = sys.argv[1]
negative_HE_img_list = sorted(glob.glob(negativeHEimgDir + "/*.npy"))

select_index_test = np.random.choice(len(negative_HE_img_list), 100, replace=False)
select_index_train = np.array([ind for ind in range(len(negative_HE_img_list)) if ind not in select_index_test])

Train_List = sorted([negative_HE_img_list[i].split("/")[-1].split(".")[0] for i in select_index_train])
Test_List  = sorted([negative_HE_img_list[i].split("/")[-1].split(".")[0] for i in select_index_test])

os.mkdir(negativeHEimgDir + "/Train_image/")
os.mkdir(negativeHEimgDir + "/Test_image/")

for ind in select_index_train:
    npy_file = negative_HE_img_list[ind].split("/")[-1]
    shutil.copy(negative_HE_img_list[ind], negativeHEimgDir + "/Train_image/" + npy_file)
    
for ind in select_index_test:
    npy_file = negative_HE_img_list[ind].split("/")[-1]
    shutil.copy(negative_HE_img_list[ind], negativeHEimgDir + "/Test_image/" + npy_file)

