#!/usr/bin/env python
# coding: utf-8

# In[1]:


# The code for random choice of negative_image_into_training_test_data


# In[2]:


from IPython.core.display import display, HTML 
display(HTML("<style>.container { width:100% !important; }</style>")) 


# In[3]:


#General module
import os, sys, glob, h5py, math, pickle
import numpy as np
import pandas as pd
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage import transform, io
from skimage import exposure
get_ipython().run_line_magic('matplotlib', 'inline')

import tensorflow as tf
import keras
from keras import backend as K
from keras.applications.xception import Xception
from keras.models import Sequential
from keras.layers import Input, Dense, BatchNormalization, Dropout, Activation, Flatten
from keras.optimizers import RMSprop
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint


# In[8]:


negative_img_2018_list = glob.glob("../IMAGE_DIR/Imgr_negative_set/Test_image_negative_2018NovDec/HistEqual_IMG/*.npy")


# In[29]:


select_index_train = np.random.choice(len(negative_img_2018_list), 100, replace=False) # np.random.choice でreplace = Falseにしないと重複したindexもつくられる。 


# select_index_train
# array([ 51, 106, 145, 132, 120, 111, 101,  40,  62,  52,   6,  10, 119,
#         90,  83,  70,  71,   9,  20, 117,  97,  68,  29,  96,  42,   7,
#         22,  39, 127, 123, 140,  35,  59, 141,  49,  16,  53, 118,  80,
#        148,  81, 130, 129, 116,   5, 105,  27, 139, 149,  25,  15, 107,
#        152, 157,  84,  11,  48,  92, 151, 147,  64,  99,  24, 122, 143,
#         50,  57,  56, 103,  73,  41,  72, 135,  89,  85, 137,  77, 134,
#         63,  26,  65, 113, 104, 124, 109, 102,  47, 133, 144,  19,  38,
#         45,   1,   4, 128,  46, 112,  28, 136, 110])

# select_index_test = np.array([ind for ind in range(len(negative_img_2018_list)) if ind not in select_index_train])
# select_index_test
# array([  0,   2,   3,   8,  12,  13,  14,  17,  18,  21,  23,  30,  31,
#         32,  33,  34,  36,  37,  43,  44,  54,  55,  58,  60,  61,  66,
#         67,  69,  74,  75,  76,  78,  79,  82,  86,  87,  88,  91,  93,
#         94,  95,  98, 100, 108, 114, 115, 121, 125, 126, 131, 138, 142,
#        146, 150, 153, 154, 155, 156])

# In[16]:


import shutil


# In[17]:


os.mkdir("../IMAGE_DIR/Imgr_negative_set/Test_image_negative_2018NovDec/HistEqual_IMG/Train_plus/")
os.mkdir("../IMAGE_DIR/Imgr_negative_set/Test_image_negative_2018NovDec/HistEqual_IMG/Test_image/")


# In[35]:


for ind in select_index_train:
    npy_file = negative_img_2018_list[ind].split("/")[-1]
    shutil.copy(negative_img_2018_list[ind], "../IMAGE_DIR/Imgr_negative_set/Test_image_negative_2018NovDec/HistEqual_IMG/Train_plus/" + npy_file)


# In[36]:


for ind in select_index_test:
    npy_file = negative_img_2018_list[ind].split("/")[-1]
    shutil.copy(negative_img_2018_list[ind], "../IMAGE_DIR/Imgr_negative_set/Test_image_negative_2018NovDec/HistEqual_IMG/Test_image/" + npy_file)


# In[33]:


len(select_index_train), len(select_index_test) 


# In[34]:


len(negative_img_2018_list)


# In[ ]:




