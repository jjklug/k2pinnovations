# k2pinnovations
machine learning final project - farzan, david, jack

Breakdown of Main files
* Ensemble.py This is the file that trains the Neural Network Ensemble (Ensemble.ipybn should be the same but foratted as a notebook so singular sections can be run one at a time)
* KFoldsCrossValidationForEnsemble.py This file runs the K-folds validation on the NN Ensemble
* logreg.py - This file runs the logistic regression on the original dataset used for comparison to NN\


Files that are not a main part of the project.
* scikitlearnMLP_NN.py was used to train a neural network on a smaller set of data just to show that it could be done
* Large_model_scikitlearnMLP_NN.py was used to train a neural network on a larger data set but we ended up using an enemble of neural networks for higher accuracy
* NN_Demo_For_Presentation is a shortened version to fit in the time limit of the presentation
* linreg.py - linear regression model for initial testing on original dataset
* k2p_nn.py - failed attempt at nn on big dataset using pytorch and tf

Various pngs that are pots produced from the code:
* NN_Ensemble_visualization.png
* log_reg_image.png
* log_reg_conf_matrix.png
* lin_reg_ex.png
* logregblockdiag.png
* ConfusionmatrixforNNensemble.png


To achieve test results presented:
* Logistic regression - run logreg.py using mas_data_ex_3.5.csv