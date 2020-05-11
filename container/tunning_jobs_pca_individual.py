import os
os.environ['KERAS_BACKEND'] = 'theano'

from analysisFuncs import *
from sklearn.model_selection import KFold
import numpy as np
from keras.callbacks import EarlyStopping
from joblib import load, dump
import argparse

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

num_inits = 10  # number of initializations inside each iteration of k-fold
num_folds = 5   # number of folds for k-fold cross-validation

win_results = {}    # storage for results


# Starting analysis
for win in range(1, nwin+1):
    # Transforming and filtering data
    data_transformed = []

    cut_data = exames[:, 0:win * win_size, :]

    for exam_index in len(exames.shape[0]):

        exam = cut_data[exam_index, :, :]

        standard = StandardScaler()
        transformed_exam = standard.fit_transform(exam)

        for pcs in range(max_pcs):

            pca = PCA(n_components=pcs)
            pca.fit_transform(transformed_exam)




