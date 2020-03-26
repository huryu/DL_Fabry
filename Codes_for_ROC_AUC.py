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


get_ipython().run_line_magic('env', 'CUDA_VISIBLE_DEVICES=3')


# In[3]:


#pos_list1 = glob.glob("../IMAGE_DIR/Original_images/2018_11_22_Urinary_Images_from_StMarianna/image/Positive_NEW_HistEq/Segmented_images/*.npy")
#pos_list2 = glob.glob("../IMAGE_DIR/Original_images/2018_11_22_Urinary_Images_from_StMarianna/image/Positive_NEW_HistEq//*/*/*.jpg")
#pos_list = pos_list1 + pos_list2


# In[4]:


#neg_list


# In[5]:


#io.(neg_list[0])


# for elm in neg_list:
#     arr = io.imread(elm)
#     plt.imshow(arr)
#     plt.show()

# In[6]:


w_posIMG, w_negIMG = (14, 217)
#w_posIMG, w_negIMG


# In[7]:


#pos_arr_list = glob.glob("../IMAGE_DIR/Original_images/2018_11_22_Urinary_Images_from_StMarianna/image/Positive_NEW_HistEq/Segmented_images/*.npy")
pos_arr_list = ["../IMAGE_DIR/Original_images/2018_11_22_Urinary_Images_from_StMarianna/"
                + "image/Positive_NEW_HistEq/Segmented_images/" 
                + "positive_new_histEq_" +str(l) + "_" + str(m) + ".npy" for l in range(w_posIMG) for m in range(16)]


# In[78]:


# These index are confirmed by direct check.

true_pos_ind = [6, 22, 37, 58, 69, 86, 106, 117, 138, 149, 166, 198, 213, 214, 209]
interM_ind = [10, 26, 38, 54, 70, 73, 74, 102, 113, 114, 118, 137, 150, 200, 212, 217, 218, 222]
BG_pos_ind = [ ind for ind in np.arange(w_posIMG * 16) if (ind not in true_pos_ind) and (ind not in interM_ind)]


# In[8]:


#neg_arr_list = glob.glob("../IMAGE_DIR/Original_images/2018_11_22_Urinary_Images_from_StMarianna/image/Negative_NEW_HistEq/Segmented_images/*.npy")
neg_arr_list = ["../IMAGE_DIR/Original_images/2018_11_22_Urinary_Images_from_StMarianna/"
                + "image/Negative_NEW_HistEq/Segmented_images/" 
                + "negative_new_histEq_" +str(l) + "_" + str(m) + ".npy" for l in range(w_negIMG) for m in range(16)]


# In[ ]:





# In[181]:


#val_id_arr = np.load("s_test7_DL_Imgr_on_5th_HE_Triple_VGG19/save_val_id.npy")
#lavel_arr = np.load("s_test7_DL_Imgr_on_5th_HE_Triple_VGG19/val_label_arr.npy")
test_pos_arr = np.array([np.load(arr) for arr in pos_arr_list])
test_neg_arr = np.array([np.load(arr) for arr in neg_arr_list])
#lavel_pos_arr = np.ones(len(test_pos_arr))
#test_arr = test_pos_arr + test_neg_arr


# In[256]:


xcep_model = keras.models.load_model("../DL_Imgr_on_5th_Model/L_test8_DL_Imgr_on_5th_HE_Triple_Xcep/model_check_point_xception.h5")
v16_model = keras.models.load_model("../DL_Imgr_on_5th_Model/L_test8_DL_Imgr_on_5th_HE_Triple_VGG16/model_check_point_vgg16.h5")
v19_model = keras.models.load_model("../DL_Imgr_on_5th_Model/L_test8_DL_Imgr_on_5th_HE_Triple_VGG19/model_check_point_vgg19.h5")


# In[257]:


model_dict = {"Xcep": xcep_model, "VGG16":v16_model, "VGG19": v19_model}


# In[258]:


#test_pos_arr[:1].shape


# In[259]:


#xcep_predict_pos_arr = xcep_model.predict(test_pos_arr)
#xcep_predict_neg_arr = xcep_model.predict(test_neg_arr)
#v16_predict_pos_arr = v16_model.predict(test_pos_arr)
#v16_predict_neg_arr = v16_model.predict(test_neg_arr)
#v19_predict_pos_arr = v19_model.predict(test_pos_arr)
#v19_predict_neg_arr = v19_model.predict(test_neg_arr)


# In[260]:


#predict_arr = [best_model.predict(arr.reshape(1, 192, 256, 3)) for arr in test_arr]
#predict_pos_arr = [best_model.predict(arr.reshape(1, 192, 256, 3)) for arr in test_pos_arr]
#predict_neg_arr = [best_model.predict(arr.reshape(1, 192, 256, 3)) for arr in test_neg_arr]


# In[261]:


#predict_arr_num = np.array([round(float(ans[0]), 3) for ans in predict_arr])
#predict_pos_arr_num = np.array([float(ans[0]) for ans in predict_pos_arr])
#predict_neg_arr_num = np.array([float(ans[0]) for ans in predict_neg_arr])


# In[262]:


#predict_arr_num


# In[263]:


#np.array(test_pos_arr[:16])


# In[264]:


model_df = {}
for name in ["Xcep", "VGG16", "VGG19"]:
    model = model_dict[name]
    predict_pos_arr = model.predict(test_pos_arr)
    predict_neg_arr = model.predict(test_neg_arr)
    
    model_sen_spe_dict = {}
    cutoff_list = [0, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 
                   0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 
                   0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999, 1.0, 1.000001]
    #cutoff_p = 0.5
    for ind, cutoff_p in enumerate(cutoff_list):
        #pred_pos_ind = np.where(predict_pos_arr_num >= cutoff_p)[0]
        pred_pos_ind = np.where(predict_pos_arr >= cutoff_p)[0]
        #pred_pos_ind, len(pred_pos_ind)

        pos_hit_ind = list(set(true_pos_ind) & set(pred_pos_ind))
        sens_segIMG = len(pos_hit_ind) / len(true_pos_ind)

        #pred_falPos_ind = np.where(predict_neg_arr_num >= cutoff_p)[0]
        pred_falPos_ind = np.where(predict_neg_arr >= cutoff_p)[0]
        #pred_falPos_ind, len(pred_falPos_ind)

        #spec_segIMG = 1- len(pred_falPos_ind) / len(predict_neg_arr_num)
        spec_segIMG = 1- len(pred_falPos_ind) / len(predict_neg_arr)
        #spec_segIMG

        # IMG_sen and IMG_spe
        #w_posIMG_sens_num = len(set([(ind // 16) for ind in np.where(predict_pos_arr_num >= cutoff_p)[0]]))
        w_posIMG_sens_num = len(set([(ind // 16) for ind in np.where(predict_pos_arr >= cutoff_p)[0]]))
        sens_wIMG = w_posIMG_sens_num / w_posIMG
        #w_posIMG_sens_num, w_posIMG, sen_wIMG

        #w_negIMG_false_pos = len(set([(ind // 16) for ind in np.where(predict_neg_arr_num >= cutoff_p)[0]]))
        w_negIMG_false_pos = len(set([(ind // 16) for ind in np.where(predict_neg_arr >= cutoff_p)[0]]))
        spec_wIMG = 1- (w_negIMG_false_pos / w_negIMG)
        #w_negIMG_false_pos, w_negIMG, spec_wIMG
        model_sen_spe_dict[ind] = [cutoff_p, sens_segIMG, spec_segIMG, sens_wIMG, spec_wIMG]
        #print(cutoff_p, sens_segIMG, spec_segIMG, sens_wIMG, spec_wIMG)
    model_SenSpe_df = pd.DataFrame(model_sen_spe_dict, index=["cutoff_p", "sens_segIMG", "spec_segIMG", "sens_wIMG", "spec_wIMG"])
    model_SenSpe_df = model_SenSpe_df.T
    model_df[name] = model_SenSpe_df


# In[265]:


model_df["VGG16"]


# In[266]:


model_df["VGG19"]


# In[267]:


model_df["Xcep"]


# In[268]:


plt.figure(figsize=(7, 7))
plt.plot([0.0, 1.0], [0.0, 1.0], "--", linewidth = 3)
plt.scatter( 1 - model_df["VGG16"]["spec_segIMG"], model_df["VGG16"]["sens_segIMG"], c = "b", linewidth=2)
plt.plot( 1 - model_df["VGG16"]["spec_segIMG"], model_df["VGG16"]["sens_segIMG"], "b", linewidth = 3)#, c = "r", linewidths=2)
plt.scatter( 1 - model_df["VGG19"]["spec_segIMG"], model_df["VGG19"]["sens_segIMG"], c = "r", linewidth=2)
plt.plot( 1 - model_df["VGG19"]["spec_segIMG"], model_df["VGG19"]["sens_segIMG"], "r", linewidth = 3, )#, c = "r", linewidths=2)
plt.scatter( 1 - model_df["Xcep"]["spec_segIMG"], model_df["Xcep"]["sens_segIMG"], c = "g", linewidth=2)
plt.plot( 1 - model_df["Xcep"]["spec_segIMG"], model_df["Xcep"]["sens_segIMG"], "g", linewidth = 3)#, c = "r", linewidths=2)
#plt.xlim(-0.01, 0.3)
#plt.ylim(0.70, 1.01)


# In[269]:


plt.figure(figsize=(7, 7))
#plt.plot([0.0, 1.0], [0.0, 1.0], "--", linewidth = 3)
plt.scatter( 1 - model_df["VGG16"]["spec_segIMG"], model_df["VGG16"]["sens_segIMG"], c = "b",s=50, marker="x")#, linewidth=4)
plt.plot( 1 - model_df["VGG16"]["spec_segIMG"], model_df["VGG16"]["sens_segIMG"], "b", linewidth = 2)#, c = "r", linewidths=2)
plt.scatter( 1 - model_df["VGG19"]["spec_segIMG"], model_df["VGG19"]["sens_segIMG"], c = "r", s=50, marker="o")#, linewidth=4)
plt.plot( 1 - model_df["VGG19"]["spec_segIMG"], model_df["VGG19"]["sens_segIMG"], "r", linewidth = 2)#, c = "r", linewidths=2)
plt.scatter( 1 - model_df["Xcep"]["spec_segIMG"], model_df["Xcep"]["sens_segIMG"], c = "g", s =50, marker="8") #, linewidth=4)
plt.plot( 1 - model_df["Xcep"]["spec_segIMG"], model_df["Xcep"]["sens_segIMG"], "g", linewidth = 2)#, c = "r", linewidths=2)

plt.xlim(-0.01, 0.3)
plt.ylim(0.70, 1.01)


# In[270]:


#ROC_for_whole_IMG

plt.figure(figsize=(5, 5))
plt.plot([0.0, 1.0], [0.0, 1.0], "--", linewidth = 3)
plt.scatter( 1 - model_df["VGG16"]["spec_wIMG"], model_df["VGG16"]["sens_wIMG"], c = "b", linewidth=2)
plt.plot( 1 - model_df["VGG16"]["spec_wIMG"], model_df["VGG16"]["sens_wIMG"], "b", linewidth = 3)#, c = "r", linewidths=2)
plt.scatter( 1 - model_df["VGG19"]["spec_wIMG"], model_df["VGG19"]["sens_wIMG"], c = "r", linewidth=2)
plt.plot( 1 - model_df["VGG19"]["spec_wIMG"], model_df["VGG19"]["sens_wIMG"], "r", linewidth = 3)#, c = "r", linewidths=2)
plt.scatter( 1 - model_df["Xcep"]["spec_wIMG"], model_df["Xcep"]["sens_wIMG"], c = "g", linewidth=2)
plt.plot( 1 - model_df["Xcep"]["spec_wIMG"], model_df["Xcep"]["sens_wIMG"], "g", linewidth = 3)#, c = "r", linewidths=2)
#plt.xlim(-0.01, 0.3)
#plt.ylim(0.70, 1.01)


# In[271]:


#ROC_AUC_for_seg_IMG
for key in model_df.keys():
    model = model_df[key]
    target_falPos_arr = np.array(1 - model["spec_segIMG"])#.sort()
    target_sen_arr = np.array(model["sens_segIMG"])#.sort()
    target_falPos_arr.sort()
    target_sen_arr.sort()
    AUC_segIMG = 0
    for ind in range(len(target_falPos_arr[:-1])):
        X_p = target_falPos_arr[ind]
        nex_X = target_falPos_arr[ind+1]
        Y_p = target_sen_arr[ind]
        nex_Y = target_sen_arr[ind+1]
        area = (nex_X - X_p) * Y_p + (nex_X - X_p) * (nex_Y - Y_p) / 2
        AUC_segIMG += area

    print(key, AUC_segIMG)


# In[272]:


#ROC_AUC_for_whole_IMG
for key in model_df.keys():
    model = model_df[key]
    target_falPos_arr = np.array(1 - model["spec_wIMG"])#.sort()
    target_sen_arr = np.array(model["sens_wIMG"])#.sort()
    target_falPos_arr.sort()
    target_sen_arr.sort()
    AUC_wIMG = 0
    for ind in range(len(target_falPos_arr[:-1])):
        X_p = target_falPos_arr[ind]
        nex_X = target_falPos_arr[ind+1]
        Y_p = target_sen_arr[ind]
        nex_Y = target_sen_arr[ind+1]
        area = (nex_X - X_p) * Y_p + (nex_X - X_p) * (nex_Y - Y_p) / 2
        AUC_wIMG += area

    print(key, AUC_wIMG)


# In[118]:


#print(true_pos_ind)


# In[117]:


#print(interM_ind)


# In[116]:


#true_pos_ind, pos_hit_ind


# In[119]:


#pred_falen(np.where(predict_neg_arr_num >= cutoff_p)[0]), len(np.where(predict_neg_arr_num >= cutoff_p)[0])/len(predict_neg_arr_num)


# In[77]:


# For imaging and predict result...

for n in range(0, len(pos_arr_list), 16):
    whole_arr = test_pos_arr[n:(n+16)]
    plt.figure(figsize = (10, 8))
    for p in range(16):
        plt.subplot(4, 4, p+1)
        plt.imshow(test_pos_arr[n + p])
        plt.xticks([])
        plt.yticks([])
    plt.show()
    
    answer_arr = np.array(predict_pos_arr_num[n:n+16]).reshape(4, 4)
    print(np.arange(n, n + 16).reshape(4, 4))
    print(answer_arr)


# In[93]:


#true_pos_ind = [6, 22, 37, 58, 69, 86, 106, 117, 138, 149, 166, 186, 198, 209, 213, 214]
#interM_ind = [10, 26, 38, 54, 70, 73, 74, 102, 113, 114, 118, 137, 150, 182, 200, 212, 217, 218, 222]
#BG_pos_ind = [ ind for ind in np.arange(w_posIMG * 16) if (ind not in true_pos_ind) and (ind not in interM_ind)]


# In[83]:


len(true_pos_ind), len(interM_ind), len(BG_pos_ind)


# In[21]:


false_postive_index = np.where(predict_neg_arr_num >= cutoff_p)[0]


# In[22]:


for ind in false_postive_index:
    print(ind)
    plt.imshow(test_neg_arr[ind])
    plt.show()


# In[ ]:


predict_arr = [best_model.predict(np.load(val_image_list[i]).reshape(1, 192, 256, 3)) for i in range(len(val_image_list))]

