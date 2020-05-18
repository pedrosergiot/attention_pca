import os
os.environ['KERAS_BACKEND'] = 'theano'

from analysisFuncs import *
import numpy as np
from joblib import load, dump
import argparse

from keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.decomposition import PCA

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
electrodes_used = (1, 4, 9, 12, 13)     # Conjunto melhor resultado da Ana

num_inits = 10  # number of initializations inside each iteration of k-fold
num_folds = 5   # number of folds for k-fold cross-validation


win_results = {}    # storage for results for each number of windows of signal used

# Starting analysis
for win in range(1, nwin+1):
    # Cut data to window size and filter it
    cut_data = exames[:, 0:win * win_size, :]

    pcs_results = {}    # storage for results for each number of pcs used

    # Loop for determining the number of PCs to use in PCA
    for pcs in range(1, max_pcs+1):
        print('pcs')
        # Transforming and filtering data
        transformed_data = []

        # Loop for applying PCA individually to each exam
        for exam_index in range(exames.shape[0]):
            exam = cut_data[exam_index, :, :]

            # Exam data transformed to zero mean and unit variance
            standard = StandardScaler()
            transformed_exam = standard.fit_transform(exam)

            # Applies PCA and reconstructs exam with some PCs
            pca = PCA(n_components=pcs)
            transformed_exam = pca.fit_transform(transformed_exam)
            transformed_exam = pca.inverse_transform(transformed_exam)
            transformed_exam = standard.inverse_transform(transformed_exam)

            transformed_data.append(transformed_exam)

        # Transformed data of all exams is filtered and energy is calculated
        transformed_data = np.asarray(transformed_data, dtype=np.float32)

        data32, data38 = filt_data(transformed_data[:, :, electrodes_used], fs, band / 2, order, f1, f2)
        # OBS: Applying filter before or after PCA still has to be discussed

        energy32 = get_energy(data32)
        energy38 = get_energy(data38)

        features = np.concatenate((energy32, energy38), axis=1)


        # Initializes K-Fold object for cross validation
        kf = KFold(n_splits=num_folds)

        results = []
        models = []
        histories = []

        for train_index, test_index in kf.split(features):

            abs_scaler = MaxAbsScaler()
            X_train = abs_scaler.fit_transform(features[train_index])
            y_train = target[train_index]

            X_test = abs_scaler.transform(features[test_index])
            y_test = target[test_index]

            best_result_fold = 0
            best_model = None
            best_history = None

            for init in range(num_inits):

                model = load(args.configFile)

                es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=20, restore_best_weights=True)
                model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                history = model.fit(X_train, y_train, epochs=500, batch_size=50, verbose=0,
                                    callbacks=[es], validation_data=(X_test, y_test))

                result_fold = model.evaluate(X_test, y_test, verbose=0)

                if result_fold[1] > best_result_fold:
                    best_result_fold = result_fold[1]
                    best_model = model
                    best_history = history

            results.append(np.array(best_result_fold))
            models.append(best_model)
            histories.append(best_history)

        pcs_results[pcs] = {'results': results, 'histories': histories, 'models': models}
    win_results[win] = pcs_results
    dump(win_results, args.outputFile)