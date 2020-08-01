# DL_Fabry

"DL_Fabry" is a deep neural network model to detect Fabry-specific signs in samples with Python.
 
# Features
 
DL_Fabry used jpeg-formatted files as training and validation datasets.

These codes are written to detect the Fabry disease with the disease-specific findings from the urine samples.

The sample figures are pre-processed with Codes_for_Positive_Negative_figure_hist_equalizer.py for adaptive histogram equalization and vectorization. Then, these processed image vectors are randomly separated with segments with PosNegSegMaker.py or BGSegmentsMaker.py for image augmentation.

These segmented images are divided into Training_data or Test_data with RandomDivider.py.
After that, the training data set are used to train the model of neural network model, which is written in DNN_Model_Train.py.

Trained model is evaluated with the sample images in a validation dataset, and these codes are written in Codes_for_model_evaluate_for_Positive_arr.py, and Codes_for_model_evaluate_for_Negative_arr.py.

Codes_for_ROC_AUC.py are written to make ROC curve and to estimate the AUC-ROC of each models.
 
# Requirement 
* Python 3.6.5
* keras
 
Environments under [Anaconda for Linux](https://www.anaconda.com/distribution/) is tested.

 
# Note
* The codes are written and performed in the setting of Linux server (Ubuntu 18.04) with NVIDIA GPU chips (NVIDIA Tesla P100 / 16GB).
* In our case, it will take 5 to 6 hours for the training of a model with GPU chips. The training dataset consists of 125000 positive segmented images, 62500 background segmented images, and 184800 negative segmented images, and each segments are vectorized into a (192, 256, 3) shaped vector. 
* With keras application, you can also perform model training without GPU chips, but it will take substantially longer time than machine with GPU.
* We have not tried our codes under Windows or Mac operation system, but we assume these python codes can usually be executed under both of these environment if you have updated python module in your system.
 
# Author
 
* Hidetaka Uryu
* National Center for Child Health and Development
 
# License
 
"DL_Fabry" is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).
 
Thank you!
