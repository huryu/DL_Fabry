#!/usr/bin/env python3

#################
# Module import #
#################

import os, sys, glob, pickle, h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
#%matplotlib inline

#self-made module
from my_classes_genarator import DataGenerator

##################################
# keras-tensorflow module import #
##################################

import tensorflow as tf
import keras
from keras import backend as K
from keras.applications.vgg19 import VGG19
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, BatchNormalization, Activation, Flatten, GlobalAveragePooling2D
from keras.optimizers import RMSprop
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint

##################
# img_shape vars #
##################

## this line is just for the setting of the width and height of each image.
scale_file = glob.glob("/mnt/ssd500/uryu/model_5th/POS_HistEq/f_img_*_0.npy")

img_shape = np.load(scale_file[0]).shape
img_rows, img_cols, color_chn = img_shape[0], img_shape[1], img_shape[2]


######################
# Model Construction #
######################

# load the filter from VGG19
# include_top=False to remove fully-connected layers.

input_tensor = Input(shape=(img_rows, img_cols, 3))
vgg19_model = VGG19(include_top=False, weights='imagenet', input_tensor=input_tensor)

x = vgg19_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(2048, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
x = Dense(1024, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=vgg19_model.input, outputs=predictions)

# VGGからのlayerのうち，15層目までのモデル重みを固定（VGG19のモデル重みを用いる）
for layer in model.layers[:17]:
     layer.trainable = False

# compile the model
#model.compile(loss='binary_crossentropy',
#              optimizer=SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True),
#              metrics=['accuracy'])

#binary_crossentropy >> ref to DL_textbook(François Chollet) P135

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=1e-4),
              metrics=['acc'])

model.summary()

##############################
# Prepare for Data Generator #
##############################

# img_list作成 glob.glob(*)は，listの再現性はないが，以後のdictの利用により自動的に再現性を担保した

pos_img_list = glob.glob("/mnt/ssd500/uryu/Imgr_5th/POS_HistEq/*")
neg_img_list = glob.glob("/mnt/ssd500/uryu/Imgr_5th/NEG_HistEq/*")
BG_pos_list  = glob.glob("/mnt/ssd500/uryu/Imgr_5th/BGfromPOS_HistEq/*")

# pos_img_list[0] は '../../../../../../mnt/ssd500/uryu/Imgr_2nd/Imgr_positive/f_img_1_14659.npy'のようになるので，
# .npyの直前 14659を keyに，listに含まれる要素をvalueに設定して，辞書を作成する。書き方は下記
# pos [0,...., 69999], neg [70000, ....., 139999, ], BG_pos [140000, .... ,, 161000] みたいになる。
# posやneg, BG_posのふくまれる要素が変更されると，これらは変更される。
# pos >>> neg >>> BG_pos の順番は確定しているので，下記を基本にする。

pos_arr_dict = {(name.split("_")[-1]).split(".")[0] : name for name in pos_img_list}
neg_arr_dict = {(name.split("_")[-1]).split(".")[0] : name for name in neg_img_list}
BG_pos_arr_dict  = {(name.split("_")[-1]).split(".")[0] : name for name in BG_pos_list}

# 空のdictを準備
pos_neg_BG_dict = {}

pos_neg_BG_dict.update(pos_arr_dict)
pos_neg_BG_dict.update(neg_arr_dict)
pos_neg_BG_dict.update(BG_pos_arr_dict)

# confirmation of contents of databases.

print("Number of Imgr images in databases: ", len(pos_arr_dict))
print("Number of control images in databases: ", len(neg_arr_dict))
print("Number of background images from positives in databases: ", len(BG_pos_arr_dict))
print("Number of total images in databases: ", len(pos_neg_BG_dict))

#ラベルをつくって，辞書化する。
labels_list = np.concatenate((np.ones(len(pos_arr_dict)), np.zeros(len(neg_arr_dict) + len(BG_pos_arr_dict))), axis = 0)
labels = {str(ind): elm for ind, elm in enumerate(labels_list)}

from sklearn.model_selection import train_test_split

pos_choice = np.arange(0, 25)
train_pos_choice, val_pos_choice = train_test_split(pos_choice, test_size = 0.20)
train_pos_index = [(name.split("_")[-1]).split(".")[0] for name in pos_img_list if int(name.split("_")[-2]) in train_pos_choice]
train_BGP_index = [(name.split("_")[-1]).split(".")[0] for name in BG_pos_list if int(name.split("_")[-2]) in train_pos_choice]
val_pos_index = [(name.split("_")[-1]).split(".")[0] for name in pos_img_list if int(name.split("_")[-2]) in val_pos_choice]
val_BGP_index = [(name.split("_")[-1]).split(".")[0] for name in BG_pos_list if int(name.split("_")[-2]) in val_pos_choice]
#print("train_pos_choice: ", train_pos_choice, "val_pos_choice: ", val_pos_choice)

neg_choice = np.arange(0, 1848)
train_neg_choice, val_test_neg_choice = train_test_split(neg_choice, test_size = 0.20)
val_neg_choice, test_neg_choice = train_test_split(val_test_neg_choice, test_size = 0.50)
train_neg_index = [(name.split("_")[-1]).split(".")[0] for name in neg_img_list if int(name.split("_")[-2]) in train_neg_choice]
val_neg_index = [(name.split("_")[-1]).split(".")[0] for name in neg_img_list if int(name.split("_")[-2]) in val_neg_choice]
test_neg_index = [(name.split("_")[-1]).split(".")[0] for name in neg_img_list if int(name.split("_")[-2]) in test_neg_choice]

train_id = train_pos_index + train_neg_index + train_BGP_index
#val_id = val_pos_index + val_neg_index + val_BG_p_index
val_id = val_pos_index + val_neg_index + val_BGP_index
test_neg_id = test_neg_index

save_val_id = np.array([int(elm) for elm in val_id])
save_test_neg_id = np.array([int(elm) for elm in test_neg_id])
val_label_arr = np.array([labels[elm] for elm in val_id])
np.save("save_val_id.npy", save_val_id)
np.save("save_test_neg_id.npy", save_test_neg_id)
np.save("val_label_arr.npy", val_label_arr)

if len(set(train_id) & set(val_id)) != 0:
    print("Alert!! duplication on data set!!")
    sys.exit()

Data_partition = {"train": train_id, "val": val_id}

batch_size = 128

params = {"vert_size": img_rows, 
          "hori_size": img_cols, 
          "color_chn": color_chn, 
          "batch_size": batch_size, 
          "shuffle": True}

training_generator = DataGenerator(**params).generate(labels, pos_neg_BG_dict, Data_partition["train"])
validation_generator = DataGenerator(**params).generate(labels, pos_neg_BG_dict, Data_partition["val"])

####################
##  model run !!! ##
####################

# model vars
model_epochs = 300
model_patience = 5

callbacks_list = [EarlyStopping(monitor='val_loss', patience=model_patience), 
                  ModelCheckpoint(filepath='model_check_point_vgg19.h5', monitor='val_loss', save_best_only=True)]

history = model.fit_generator(generator = training_generator,
                    steps_per_epoch = len(Data_partition['train'])//batch_size,
                    validation_data = validation_generator,
                    validation_steps = len(Data_partition['val'])//batch_size,
                    epochs = model_epochs,
                    callbacks = callbacks_list)

######################
##  history save !! ##
######################

history_arr = np.array([history.epoch, history.history["loss"], history.history["val_loss"], history.history["acc"], history.history["val_acc"]])
history_df = pd.DataFrame(history_arr.T, columns=["epoch", "loss", "val_loss", "acc", "val_acc"])
history_df.to_csv("vgg19_history.tsv", sep="\t")

####################
##  model save !! ##
####################

time_point = str(datetime.today().year) + str(datetime.today().month) +  str(datetime.today().day)

model.save("DL_Imgr_vgg19_" + time_point + "_"+ str(model_epochs) + "epoch_" + str(model_patience) + "patience.hdf5")

plt.plot(history.epoch, history.history["loss"])
plt.plot(history.epoch, history.history["val_loss"])
plt.savefig("DL_Imgr_vgg19_" + time_point + "_" + "Loss_value.png")

plt.clf()

plt.plot(history.epoch, history.history["acc"])
plt.plot(history.epoch, history.history["val_acc"])
plt.savefig("DL_Imgr_vgg19_" + time_point + "_" + "Accuracy.png")

plt.clf()
