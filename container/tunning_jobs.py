# pode-se passar os callbacks do keras para o fit do pipeline usando o formato modelo__callbacks=es
# epochs=500, batch_size=50, callbacks=[es], verbose=0
# es = EarlyStopping(monitor='loss', mode='min', verbose=0, patience=20, restore_best_weights=True)
# pipe['model'].compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
import os
os.environ['KERAS_BACKEND'] = 'theano'

from analysisFuncs import *
from modelFuncs import *
import keras.backend as K
from sklearn.model_selection import KFold
import numpy as np
from keras.callbacks import EarlyStopping
from joblib import load, dump
import argparse

# Arg parser
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--configFile', action='store', dest='configFile', required=True,
                    help="The job config file that will be used to configure the job (sort and init).")
parser.add_argument('-o', '--outputFile', action='store', dest='outputFile', required=True, default=None,
                    help="The output tuning name.")
parser.add_argument('-d', '--dataFile', action='store', dest='dataFile', required=True, default=None,
                    help="The data/target file used to train the model.")
args = parser.parse_args()


# Load data from pickle file
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
num_inits = 10

win_results = {}

for win in range(1, nwin+1):

    # filtering the data for the frequencies of interest (f1 and f2)
    # data is cut to have the number of points desired (number of windows * window size) before filtering
    dataf1, dataf2 = filt_data(exames[:, 0:win * win_size, :], fs, 0.2, 4, f1, f2)

    # calculates the energy of the resultant signals obtained by the filering process
    energyf1 = get_energy(dataf1)
    energyf2 = get_energy(dataf2)

    # concatenates the energy values obtained
    features = np.concatenate((energyf1, energyf2), axis=1)

    # CV K-Fold (k = 5)
    kf = KFold(n_splits=5)

    results = []
    models = []

    for train_index, test_index in kf.split(features):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = target[train_index], target[test_index]

        best_result_fold = 0
        best_pipe = None

        # Open model, compile and set early stopping
        for init in range(num_inits):
            pipe = load(args.configFile)
            pipe['model'].compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=20, restore_best_weights=True)
            pipe.fit(X_train, y_train, model__epochs=500, model__batch_size=50, model__verbose=0,
                    model__callbacks=[es], validation_data=(X_test, y_test))

            result_fold = pipe.score(X_test, y_test)

            if result_fold > best_result_fold:
                best_results_fold = result_fold
                best_pipe = pipe

            K.clear_session()

        results.append(np.array(best_result_fold))
        models.append(best_pipe)

    win_results[win] = (results, models)

dump(win_results, args.outputFile)