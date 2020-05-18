import os
os.environ['KERAS_BACKEND'] = 'theano'

import argparse
from analysisFuncs import *
import numpy as np
from joblib import load, dump
import copy
from keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold

# Args received from maestro
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--configFile', action='store', dest='configFile', required=True,
                    help="The job config file that will be used to configure the job (sort and init).")
parser.add_argument('-o', '--outputFile', action='store', dest='outputFile', required=True, default=None,
                    help="The output tuning name.")
parser.add_argument('-d', '--dataFile', action='store', dest='dataFile', required=True, default=None,
                    help="The data/target file used to train the model.")
args = parser.parse_args()

# Loading data from .mat file
exames, target = load_data(args.dataFile)

# analysis parameters
fs = 601.5  # sampling rate
f1 = 31.13  # modulating frequency 1
f2 = 39.36  # modulating frequency 2

# filter parameters
band = 0.4
order = 4

# parameters for the feature extraction
win_size = 64
nwin = int(exames.shape[1]/win_size)
max_pcs = 18

num_inits = 10  # number of initializations inside each iteration of k-fold
num_folds = 5   # number of folds for k-fold cross-validation


win_results = {}

for win in range(1, nwin+1):
    # filtering the data for the frequencies of interest (f1 and f2)
    # data is cut to have the number of points desired (number of windows * window size) before filtering
    dataf1, dataf2 = filt_data(exames[:, 0:win * win_size, :], fs, band / 2, order, f1, f2)

    # load models and other stuff from configFile
    analysis_steps = load(args.configFile)

    transformed_dataf1 = []
    transformed_dataf2 = []

    # Access each exam and applies PCA to it according the n_pcs defined by the configFile loaded
    for exam_it in range(exames.shape[0]):

        standard = copy.deepcopy(analysis_steps['standardization'])
        pca = copy.deepcopy(analysis_steps['PCA'])
        transformed_exam = pca.fit_transform(standard.fit_transform(dataf1[exam_it, :, :]))
        transformed_dataf1.append(transformed_exam)

        standard = copy.deepcopy(analysis_steps['standardization'])
        pca = copy.deepcopy(analysis_steps['PCA'])
        transformed_exam = pca.fit_transform(standard.fit_transform(dataf2[exam_it, :, :]))
        transformed_dataf2.append(transformed_exam)

    # Calculates the energy of the resultant signals obtained by the filtering process
    energyf1 = get_energy(np.asarray(transformed_dataf1))
    energyf2 = get_energy(np.asarray(transformed_dataf2))

    # Concatenates the energy values obtained
    features = np.concatenate((energyf1, energyf2), axis=1)

    # CV K-Fold
    kf = KFold(n_splits=num_folds)

    results = []
    models = []
    histories = []

    for train_index, test_index in kf.split(features):

        # loads normalization method
        normalize = copy.deepcopy(analysis_steps['normalization'])

        # divides the data and normalizes it
        X_train, X_test = normalize.fit_transform(features[train_index]), normalize.transform(features[test_index])
        y_train, y_test = target[train_index], target[test_index]

        best_result_fold = 0
        best_model = None
        best_history = None

        # Open model, compile and set early stopping for each init
        for init in range(num_inits):
            model = copy.deepcopy(analysis_steps['model'])

            es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=20, restore_best_weights=True)
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            history = model.fit(X_train, y_train, epochs=500, batch_size=50, verbose=0, callbacks=[es],
                                validation_data=(X_test, y_test))

            result_fold = model.evaluate(X_test, y_test, verbose=0)

            if result_fold[1] > best_result_fold:
                best_result_fold = result_fold[1]
                best_model = model
                best_history = history

        results.append(np.array(best_result_fold))
        models.append(best_model)
        histories.append(best_history)

    win_results[win] = {'results': results, 'histories': histories, 'models': models}
    dump(win_results, args.outputFile)