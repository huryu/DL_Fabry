# DL_Fabry

"DL_Fabry" is a deep neural network model to detect Fabry-specific signs in samples with Python.
 
# Features
 
DL_Fabry used jpeg-formatted files as training and validation datasets.

These codes are written to detect the Fabry disease with the disease-specific findings from the urine samples.

The sample figures are pre-processed with Codes_for_Positive_Negative_figure_hist_equalizer.py for adaptive histogram equalization and vectorization. Then, these processed image vectors are randomly separated with segments with PosNegSegMaker.py or BGSegmentsMaker.py for image augmentation.

These segmented images are divided into Training_data or Test_data with Codes_for_Random_Choice.py.
After that, the training data set are used to train the model of neural network model, which is written in Codes_for_NeuralNetworkTraining.py.

Trained model is evaluated with the test sample images, and these codes are written in Codes_for_model_evaluate_for_Positive_arr.py, and Codes_for_model_evaluate_for_Negative_arr.py.

Codes_for_ROC_AUC.py are written to make ROC curve and to estimate the AUC-ROC of each models.
 
# Requirement
 
* Python 3.6.5
* keras
 
Environments under [Anaconda for Linux](https://www.anaconda.com/distribution/) is tested.

 
# Note
This codes is supposed to use under Linux. 
I don't test environments under Windows and Mac.
 
# Author
 
* Hidetaka Uryu
* National Center for Child Health and Development
 
# License
 
"DL_Fabry" is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).
 
Thank you!
