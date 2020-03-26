These codes are written to diagnose the disease with the clue from the urine samples.

The sample figures are pre-processed with Codes_for_Positive_Negative_figure_hist_equalizer.py followed by Codes_for_POSITIVE_SEG_with_HistEqual.py, and Codes_for_NEGATIVE_SEG_with_HistEqual.py.

Then, these processed figures are separated into small pieces with Codes_for_Image_amplification.py, and divided into Training_data or Test_data with Codes_for_Random_Choice.py.
After that, the training data set are used to train the model of neural network model, which is written in Codes_for_NeuralNetworkTraining_on5th.py.

Trained model is evaluated with the test sample images, and these codes are written in Codes_for_model_evaluate_for_Positive_arr.py, and Codes_for_model_evaluate_for_Negative_arr.py.
Codes_for_ROC_AUC.py are written to make ROC curve and to estimate the AUC-ROC of each models.
