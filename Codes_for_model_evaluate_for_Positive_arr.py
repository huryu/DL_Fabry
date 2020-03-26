#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python3

#################
# Module import #
#################

import os, sys, glob, pickle, h5py
import numpy as np
import pandas as pd
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
get_ipython().run_line_magic('matplotlib', 'inline')

#self-made module
#from my_classes_genarator import DataGenerator

##################################
# keras-tensorflow module import #
##################################

import tensorflow as tf
import keras
from keras import backend as K
from keras.applications.xception import Xception
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, BatchNormalization, Activation, Flatten, GlobalAveragePooling2D
from keras.optimizers import RMSprop
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint


# In[2]:


get_ipython().run_line_magic('env', 'CUDA_VISIBLE_DEVICES=1')


# In[3]:


pos_list1 = glob.glob("../IMAGE_DIR/Original_images/2018_11_22_Urinary_Images_from_StMarianna/image/Positive_NEW_HistEq/Segmented_images/*.npy")
pos_list2 = glob.glob("../IMAGE_DIR/Original_images/2018_11_22_Urinary_Images_from_StMarianna/image/Positive_NEW_HistEq//*/*/*.jpg")
pos_list = pos_list1 + pos_list2


# In[4]:


#neg_list


# In[5]:


#io.(neg_list[0])


# for elm in neg_list:
#     arr = io.imread(elm)
#     plt.imshow(arr)
#     plt.show()

# In[6]:


#pos_arr_list = glob.glob("../IMAGE_DIR/Original_images/2018_11_22_Urinary_Images_from_StMarianna/image/Positive_NEW_HistEq/Segmented_images/*.npy")
pos_arr_list = ["../IMAGE_DIR/Original_images/2018_11_22_Urinary_Images_from_StMarianna/"
                + "image/Positive_NEW_HistEq/Segmented_images/" 
                + "positive_new_histEq_" +str(l) + "_" + str(m) + ".npy" for l in range(14) for m in range(16)]


# In[7]:


#val_id_arr = np.load("s_test7_DL_Imgr_on_5th_HE_Triple_VGG19/save_val_id.npy")
#lavel_arr = np.load("s_test7_DL_Imgr_on_5th_HE_Triple_VGG19/val_label_arr.npy")
test_pos_arr = [np.load(arr) for arr in pos_arr_list]
#lavel_pos_arr = np.ones(len(test_pos_arr))


# In[ ]:


best_model = keras.models.load_model("../DL_Imgr_on_5th_Model/L_test2_DL_Imgr_on_5th_HE_Triple_VGG19/model_check_point_vgg19.h5")


# In[7]:


test_pos_arr[0].shape


# In[22]:


predict_arr = [best_model.predict(arr.reshape(1, 192, 256, 3)) for arr in test_pos_arr]


# In[23]:


#predict_arr_num = np.array([round(float(ans[0]), 3) for ans in predict_arr])
predict_arr_num = np.array([float(ans[0]) for ans in predict_arr])


# In[ ]:


print(predict_arr_num)


# In[25]:


#np.array(test_pos_arr[:16])


# In[26]:


for n in range(0, len(pos_arr_list), 16):
    whole_arr = test_pos_arr[n:(n+16)]
    plt.figure(figsize = (10, 8))
    for p in range(16):
        plt.subplot(4, 4, p+1)
        plt.imshow(test_pos_arr[n + p])
        plt.xticks([])
        plt.yticks([])
    plt.show()
    
    answer_arr = np.array(predict_arr_num[n:n+16]).reshape(4, 4)
    print(answer_arr)


# In[ ]:





# In[20]:


cutoff_p = 0.9999999
np.where(predict_arr_num >= cutoff_p)[0], len(np.where(predict_arr_num >= cutoff_p)[0])


# In[12]:


len(test_pos_arr) / 16


# In[27]:


false_postive_index = np.where(predict_arr_num >= 0.9)[0]


# In[ ]:


for ind in false_postive_index:
    print(ind)
    plt.imshow(test_neg_arr[ind])
    plt.show()


# In[ ]:


predict_arr = [best_model.predict(np.load(val_image_list[i]).reshape(1, 192, 256, 3)) for i in range(len(val_image_list))]

