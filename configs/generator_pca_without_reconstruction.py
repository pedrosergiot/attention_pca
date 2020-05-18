import os
os.environ['KERAS_BACKEND'] = 'theano'

from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from joblib import dump


# function which creates the model to be used
def create_model(neurons1=1, neurons2=0, n_features=None):
    # create model
    model = Sequential()
    model.add(Dense(neurons1, input_dim=n_features, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(0.2))
    if neurons2 != 0:
        model.add(Dense(neurons2, kernel_initializer='uniform', activation='relu'))
        model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    # Model is not compiled to not save weights, allowing weight initialization later
    return model


pcs = range(1,20)           # 19 PCs for PCA
n_neurons1 = range(1, 11)   # 1-10 neurons on the first layer
n_neurons2 = range(0, 11)   # 0-10 neurons on the second layer

for npcs in pcs:
    for neuron1 in n_neurons1:
        for neuron2 in n_neurons2:
            analysis_steps = {'standardization': StandardScaler(),
                              'PCA': PCA(n_components=npcs),
                              'normalization': MaxAbsScaler(),
                              'model': create_model(n_features=2*npcs, neurons1=neuron1, neurons2=neuron2)}
            # n_features is equal to 2 times the npcs because there will be two sets of exames: one for 32 Hz and
            # onde for 38 Hz

            file_name = "user.pedrosergiot.pca_without_reconstruction/" \
                        "user.pedrosergiot.pca_without_reconstruction." \
                        "npcs{}.neurons1_{}.neurons2_{}.pkl".format(npcs, neuron1, neuron2)


            dirname = os.path.dirname(file_name)
            if not os.path.exists(dirname):
                os.makedirs(dirname)

            dump(analysis_steps, file_name)