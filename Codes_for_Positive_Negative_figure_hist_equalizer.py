#!/usr/bin/env python
# coding: utf-8

# In[1]:


# The code for making images with histogram equalized


# In[2]:


from IPython.core.display import display, HTML 
display(HTML("<style>.container { width:100% !important; }</style>")) 


# In[3]:


import os, sys, glob, pickle, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


import shutil


# In[5]:


from skimage import exposure
from skimage import transform, io


# In[6]:


#positive_img_list = glob.glob("../IMAGE_DIR/Imgr_positive_set/Positive_training_image_set/positive_*.jpg")
#negative_new_img_list_1 = glob.glob("../IMAGE_DIR/Original_images/2018_11_22_Urinary_Images_from_StMarianna/image/Negative_NEW/*/*.jpg")
#negative_new_img_list_2 = glob.glob("../IMAGE_DIR/Original_images/2018_11_22_Urinary_Images_from_StMarianna/image/Negative_NEW/*/*/*.jpg")
#negative_new_img_list = negative_new_img_list_1 + negative_new_img_list_2

#positive_new_img_list_1 = glob.glob("../IMAGE_DIR/Original_images/2018_11_22_Urinary_Images_from_StMarianna/image/Imgr_candidate/マルベリー/test_*.jpg")
#positive_new_img_list_2 = glob.glob("../IMAGE_DIR/Original_images/2018_11_22_Urinary_Images_from_StMarianna/image/Imgr_candidate/マルベリー様未確定/test_*.jpg")
#positive_new_img_list = positive_new_img_list_1 + positive_new_img_list_2
negative_img_list = glob.glob("../IMAGE_DIR/Imgr_negative_set/Test_image_negative_2018NovDec/181*.jpg")
#negative_new_img_list_2 = glob.glob("../IMAGE_DIR/Original_images/2018_11_22_Urinary_Images_from_StMarianna/image/Negative_NEW/*/*/*.jpg")
#negative_new_img_list = negative_new_img_list_1 + negative_new_img_list_2

#20181208 以前のファイルに追加で作成
#positive_img_list = glob.glob("../IMAGE_DIR/Imgr_positive_set/Imgr_candidate/右田先生用/180718-*.jpg")

#negative_img_list = glob.glob("../IMAGE_DIR/Original_images/2018_11_22_Urinary_Images_from_StMarianna/image/Neg_OLD_numbered/negative_*.jpg")
#negative_img_list = glob.glob("../IMAGE_DIR/Imgr_negative_set/Neg_OLD_numbered/negative_*.jpg")


# In[7]:


#positive_img_list
#negative_img_list


# In[8]:


#positive_new_img_list.sort()
#positive_img_list.sort()
negative_img_list.sort()


# In[9]:


#positive_new_img_list
#positive_img_list


# In[10]:


# Carefully use when appending positive image list
#for i in range(len(positive_img_list)):
#    shutil.copy2(positive_img_list[i], "../IMAGE_DIR/IMG_FOLDER/positive_" + str(i + 8) + ".jpg")


# In[11]:


#pos_imgarr_dict = {}
#pos_new_imgarr_dict = {}


# In[12]:


neg_imgarr_dict = {}
#neg_new_imgarr_dict = {}


# In[13]:


#i = 0
#for pos_img in positive_img_list:
#    pos_imgarr_dict[i] = io.imread(pos_img)
#    i += 1


# In[16]:


#negative_img_list


# In[17]:


i = 0
for neg_img in negative_img_list:
    neg_imgarr_dict[i] = io.imread(neg_img)
    i += 1


# In[18]:


#i = 0
#for neg_img in negative_new_img_list:
#    neg_new_imgarr_dict[i] = io.imread(neg_img)
#    i += 1


# In[19]:


#for i, pos_img in enumerate(positive_new_img_list):
#    pos_new_imgarr_dict[i] = io.imread(pos_img)


# In[20]:


#shape_vert = pos_imgarr_dict[0].shape[0]
#shape_hori = pos_imgarr_dict[0].shape[1]
#shape_vert = pos_new_imgarr_dict[0].shape[0]
#shape_hori = pos_new_imgarr_dict[0].shape[1]
shape_vert = neg_imgarr_dict[0].shape[0]
shape_hori = neg_imgarr_dict[0].shape[1]
#shape_vert = pos_new_imgarr_dict[0].shape[0]
#shape_hori = pos_new_imgarr_dict[0].shape[1]

shape_vert, shape_hori


# In[21]:


vert_half, hori_half = math.ceil(shape_vert / 8), math.ceil(shape_hori / 8)
vert_half, hori_half
#body_centY ,body_centX = body_cent_Y_X[1]


# In[22]:


#pos_imgarr_dict.keys()
neg_imgarr_dict.keys()
#pos_new_imgarr_dict.keys()


# In[23]:


# center of malberry are decide with direct view.
#body_cent_Y_X = {}
#body_cent_Y_X[0] = (350, 500)
#body_cent_Y_X[1] = (370, 470)
#body_cent_Y_X[2] = (190, 180)
#body_cent_Y_X[3] = (370, 730)
#body_cent_Y_X[4] = (405, 470)
#body_cent_Y_X[5] = (180, 660)
#body_cent_Y_X[6] = (390, 490)


# In[ ]:





# In[24]:


#plt.hist(img_unite[:, :, 2].reshape(1, -1)[0], bins=255)


# In[25]:


#plt.imshow(neg_imgarr_dict[2])


# In[26]:


#positive_img_list[0]


# In[27]:


#negative_img_list[1200]


# In[28]:


#pos_imgarr_dict[0]


# In[29]:


#for i in pos_imgarr_dict.keys():
#    original_file_name = positive_img_list[i].split("/")[-1]
#    base_file_name = original_file_name.split(".")[0]
#
#    print(base_file_name)


# neg_imgarr_dict.keys()

# In[31]:


# For negative image histgram equalization

for i in neg_imgarr_dict.keys():
    original_file_name = negative_img_list[i].split("/")[-1]
    base_file_name = original_file_name.split(".")[0]

    #print(original_file_name, base_file_name)
    #break
    
    img_0 = neg_imgarr_dict[i][:, :, 0]
    img_1 = neg_imgarr_dict[i][:, :, 1]
    img_2 = neg_imgarr_dict[i][:, :, 2]


    # Contrast stretching
    p2_0, p98_0 = np.percentile(img_0, (2, 98))
    p2_1, p98_1 = np.percentile(img_1, (2, 98))
    p2_2, p98_2 = np.percentile(img_2, (2, 98))

    img_rescale_0 = exposure.rescale_intensity(img_0, in_range=(p2_0, p98_0))
    img_rescale_1 = exposure.rescale_intensity(img_1, in_range=(p2_1, p98_1))
    img_rescale_2 = exposure.rescale_intensity(img_2, in_range=(p2_2, p98_2))

    # Equalization
    img_eq_0 = exposure.equalize_hist(img_0)
    img_eq_1 = exposure.equalize_hist(img_1)
    img_eq_2 = exposure.equalize_hist(img_2)

    # Adaptive Equalization
    img_adapteq_0 = exposure.equalize_adapthist(img_0, clip_limit=0.03)
    img_adapteq_1 = exposure.equalize_adapthist(img_1, clip_limit=0.03)
    img_adapteq_2 = exposure.equalize_adapthist(img_2, clip_limit=0.03)

    img_unite = neg_imgarr_dict[i].copy().astype(float)
    img_unite[:, :, 0] = img_adapteq_0
    img_unite[:, :, 1] = img_adapteq_1
    img_unite[:, :, 2] = img_adapteq_2
    
    #print(base_file_name + ".npy")
    #plt.imshow(img_unite)
    #plt.show()
    
    #break

    np.save("../IMAGE_DIR/Imgr_negative_set/Test_image_negative_2018NovDec/HistEqual_IMG/" + base_file_name + ".npy", img_unite)
    #np.save("IMG_FOLDER_HIST_EQUAL/" + positive_" + str(i) + ".npy", img_unite)


# In[21]:


# For positive new image histgram equalization

for i in pos_new_imgarr_dict.keys():
    original_file_name = positive_new_img_list[i].split("/")[-1]
    base_file_name = original_file_name.split(".")[0][5:]

    #print(original_file_name, base_file_name)
    #break
    #continue
    
    img_0 = pos_new_imgarr_dict[i][:, :, 0]
    img_1 = pos_new_imgarr_dict[i][:, :, 1]
    img_2 = pos_new_imgarr_dict[i][:, :, 2]


    # Contrast stretching
    p2_0, p98_0 = np.percentile(img_0, (2, 98))
    p2_1, p98_1 = np.percentile(img_1, (2, 98))
    p2_2, p98_2 = np.percentile(img_2, (2, 98))

    img_rescale_0 = exposure.rescale_intensity(img_0, in_range=(p2_0, p98_0))
    img_rescale_1 = exposure.rescale_intensity(img_1, in_range=(p2_1, p98_1))
    img_rescale_2 = exposure.rescale_intensity(img_2, in_range=(p2_2, p98_2))

    # Equalization
    img_eq_0 = exposure.equalize_hist(img_0)
    img_eq_1 = exposure.equalize_hist(img_1)
    img_eq_2 = exposure.equalize_hist(img_2)

    # Adaptive Equalization
    img_adapteq_0 = exposure.equalize_adapthist(img_0, clip_limit=0.03)
    img_adapteq_1 = exposure.equalize_adapthist(img_1, clip_limit=0.03)
    img_adapteq_2 = exposure.equalize_adapthist(img_2, clip_limit=0.03)

    img_unite = pos_new_imgarr_dict[i].copy().astype(float)
    img_unite[:, :, 0] = img_adapteq_0
    img_unite[:, :, 1] = img_adapteq_1
    img_unite[:, :, 2] = img_adapteq_2
    
    #print(base_file_name + ".npy")
    #plt.imshow(img_unite)
    #plt.show()

    np.save("../IMAGE_DIR/Original_images/2018_11_22_Urinary_Images_from_StMarianna/image/Positive_NEW_HistEq/" + base_file_name + ".npy", img_unite)
    #np.save("IMG_FOLDER_HIST_EQUAL/" + positive_" + str(i) + ".npy", img_unite)


# In[26]:


img_unite =  np.load("../IMAGE_DIR/Original_images/2018_11_22_Urinary_Images_from_StMarianna/image/Positive_NEW_HistEq/181011-141741.npy")
plt.imshow(img_unite, vmax=1.0, vmin=0)


# In[36]:


# Confirm original images!!
plt.imshow(io.imread(glob.glob("../IMAGE_DIR/Original_images/2018_11_22_Urinary_Images_from_StMarianna/image/Negative_NEW/*/180202-153854.jpg")[0]))


# In[ ]:





# In[ ]:


#############################################################################
#### Core cord are above. The following codes are just for confirmation. ####
#############################################################################


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[97]:


plt.imshow(img_unite[250:500, 400:800])


# In[95]:


plt.imshow(pos_imgarr_dict[4][350:500, 400:600])


# In[59]:


img_unite


# In[62]:


np.where(img_unite == 1)


# In[37]:


plt.imshow(img_adapteq_0)


# In[ ]:





# In[25]:


plt.imshow(pos_imgarr_dict[1][:, :, 0])


# In[20]:


plt.hist(pos_imgarr_dict[0][:, :, 0].reshape(1,-1)[0], bins=255)


# In[21]:


plt.hist(pos_imgarr_dict[1][:, :, 0].reshape(1,-1)[0], bins=255)


# In[22]:


plt.hist(pos_imgarr_dict[2][:, :, 0].reshape(1,-1)[0], bins=255)


# In[23]:


plt.hist(pos_imgarr_dict[3][:, :, 0].reshape(1,-1)[0], bins=255)


# In[ ]:





# In[ ]:





# In[ ]:





# In[68]:


np.arange(0, shape_vert, 100), np.arange(0, shape_hori, 100)


# In[69]:


check_point = [(i, j) for i in np.arange(0, shape_vert, 100) for j in np.arange(0, shape_hori, 100)]


# In[70]:


len(check_point)


# In[73]:


i = 1 # image_number
k = 17 # check_point number
win_size = 200

plt.figure(figsize=(20, 40))

plt.subplot(2, 4, 1)
plt.imshow(pos_imgarr_dict[i][check_point[k][0]:check_point[k][0]+win_size, check_point[k][1]:check_point[k][1]+win_size])

plt.subplot(2, 4, 2)
plt.imshow(pos_imgarr_dict[i][check_point[k + 1][0]:check_point[k + 1][0]+win_size, check_point[k + 1][1]:check_point[k + 1][1]+win_size])

plt.subplot(2, 4, 3)
plt.imshow(pos_imgarr_dict[i][check_point[k + 2][0]:check_point[k + 2][0]+win_size, check_point[k + 2][1]:check_point[k + 2][1]+win_size])

plt.subplot(2, 4, 4)
plt.imshow(pos_imgarr_dict[i][check_point[k + 3][0]:check_point[k + 3][0]+win_size, check_point[k + 3][1]:check_point[k + 3][1]+win_size])

plt.subplot(2, 4, 5)
plt.imshow(pos_imgarr_dict[i][check_point[k + 4][0]:check_point[k + 4][0]+win_size, check_point[k + 4][1]:check_point[k + 4][1]+win_size])

plt.subplot(2, 4, 6)
plt.imshow(pos_imgarr_dict[i][check_point[k + 5][0]:check_point[k + 5][0]+win_size, check_point[k + 5][1]:check_point[k + 5][1]+win_size])

plt.subplot(2, 4, 7)
plt.imshow(pos_imgarr_dict[i][check_point[k + 6][0]:check_point[k + 6][0]+win_size, check_point[k + 6][1]:check_point[k + 6][1]+win_size])

plt.subplot(2, 4, 8)
plt.imshow(pos_imgarr_dict[i][check_point[k + 7][0]:check_point[k + 7][0]+win_size, check_point[k + 7][1]:check_point[k + 7][1]+win_size])


# In[182]:


i_list = [0, 0, 0, 0, 0, 0,
          1, 1, 1, 1, 1, 1, 1, 1, 1,
          2, 2, 2, 2, 
          3, 3, 3, 3, 3, 
          4, 4, 4, 4]
k_list = [12, 26, 46, 46, 61, 70,
          3, 9, 17, 18, 37, 38, 44, 77, 84,
          12, 47, 62, 69, 
          7, 20, 26, 39, 72, 
          22, 37, 38, 38]
target_point = [(180, 140), (150, 110), (100, 50), (90, 160), (50, 100), (130, 60),
                (80, 30), (35, 85), (130, 60), (160, 75), (75, 75), (30, 60), (125, 35), (25, 100), (50, 60),
                (80, 85), (70, 110), (80, 105), (45, 70), 
                (60, 60), (90, 100), (170, 55), (70, 130), (95, 45), 
                (80, 20), (105, 75), (50, 125), (120, 105)] 


# In[183]:


target_list_dict = {"i_list": i_list, "k_list": k_list, "target_point_list": target_point}


# In[184]:


with open('target_list_dict.pkl', mode='wb') as f:
    pickle.dump(target_list_dict, f)


# In[77]:


#with open('Image_amplifier_body_cent_point_dict.pkl', mode='wb') as f:
#    pickle.dump(body_cent_Y_X, f)


# In[78]:


########################
# United codes revised #
########################

### make 70000 image of malberry negative figure ###
np_count = 0 

while np_count < 150000:
    if np_count % 1000 == 0:
        print(np_count)
        
    #i = np_count // 10000

#body_centY, body_centX = body_cent_Y_X[i]
    i = np.random.randint(27)
#print(i)
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
        #plt.imshow(f_img)
            np.save("../../../../../mnt/usb34/Imgr_negative/" + "f_imgN_" + str(i) + "_" + str(np_count), f_img)
            np_count += 1


# In[ ]:





# In[ ]:


#### Temporary complete of code for image_amplifier of malberry_cells positive figures. #####


# In[ ]:





# In[ ]:


### Window for confirmation ###


# In[ ]:





# In[ ]:


import glob
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[79]:


#np_count = np.random.randint(0, 150000)
# basic negative figure are numbered as 0, 1, 2, 3, ....., 27
i = np.random.randint(27)

a_list = glob.glob("../../../../../mnt/usb34/Imgr_negative/" + "f_imgN_" + str(i) + "_" + "*" + ".npy")
rand_id = np.random.randint(len(a_list))
a = np.load(a_list[rand_id])


# In[ ]:


plt.imshow(a)


# In[ ]:





# In[ ]:





# In[25]:


# Trial code
-
    
    np.save("IMG_FOLDER_HIST_EQUAL/negative_" + str(i) + ".npy", img_unite)


# In[29]:


img_unite = np.load("IMG_FOLDER_HIST_EQUAL/negative_15.npy")
plt.imshow(img_unite, vmax=1.0, vmin=0)

