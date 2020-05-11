import os
os.environ['KERAS_BACKEND'] = 'theano'

from analysisFuncs import *
from sklearn.model_selection import KFold
import numpy as np
from keras.callbacks import EarlyStopping
from joblib import load, dump
import argparse

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

num_inits = 10  # number of initializations inside each iteration of k-fold
num_folds = 5   # number of folds for k-fold cross-validation

win_results = {}    # storage for results


# Starting analysis
for win in range(1, nwin+1):

    # filtering the data for the frequencies of interest (f1 and f2)
    # data is cut to have the number of points desired (number of windows * window size) before filtering
    dataf1, dataf2 = filt_data(exames[:, 0:win * win_size, :], fs, band/2, order, f1, f2)

    # calculates the energy of the resultant signals obtained by the filtering process
    energyf1 = get_energy(dataf1)
    energyf2 = get_energy(dataf2)

    # concatenates the energy values obtained
    features = np.concatenate((energyf1, energyf2), axis=1)

    # CV K-Fold
    kf = KFold(n_splits=num_folds)

    results = []
    pipes = []
    histories = []

    for train_index, test_index in kf.split(features):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = target[train_index], target[test_index]

        best_result_fold = 0
        best_pipe = None
        best_history = None

        # Open model, compile and set early stopping for each init
        for init in range(num_inits):
            pipe = load(args.configFile)

            X_train_transformed = \
                pipe['standard'].fit_transform(
                    pipe['reducer'].fit_transform(
                        pipe['scale'].fit_transform(X_train)))

            X_test_transformed = \
                pipe['standard'].transform(
                    pipe['reducer'].transform(
                        pipe['scale'].transform(X_test)))


            es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=20, restore_best_weights=True)
            pipe['model'].compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            history = pipe['model'].fit(X_train_transformed, y_train, epochs=500, batch_size=50, verbose=0,
                                        callbacks=[es], validation_data=(X_test_transformed, y_test))

            result_fold = pipe['model'].evaluate(X_test_transformed, y_test, verbose=0)

            if result_fold[1] > best_result_fold:
                best_result_fold = result_fold[1]
                best_pipe = pipe
                best_history = history


        results.append(np.array(best_result_fold))
        pipes.append(best_pipe)
        histories.append(best_history)

    win_results[win] = {'results': results, 'histories': histories, 'pipes': pipes}
dump(win_results, args.outputFile)