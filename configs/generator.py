import os
os.environ['KERAS_BACKEND'] = 'theano'

from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.pipeline import Pipeline
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
    # Compile model
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


pcs = range(1,39)
n_neurons1 = range(1, 11)
n_neurons2 = range(0, 11)

for npcs in pcs:
    for neuron1 in n_neurons1:
        for neuron2 in n_neurons2:
            pipe = Pipeline([('scale', StandardScaler()),
                             ('reducer', PCA(n_components=npcs)),
                             ('standard', MaxAbsScaler()),
                             ('model', create_model(n_features=npcs, neurons1=neuron1, neurons2=neuron2))])

            file_name = "user.pedrosergiot.attention_pca_filter_first/" \
                        "user.pedrosergiot.attention_pca_filter_first." \
                        "npcs{}.neurons1{}.neurons2{}.pkl".format(npcs, neuron1, neuron2)


            dirname = os.path.dirname(file_name)
            if not os.path.exists(dirname):
                os.makedirs(dirname)

            dump(pipe, file_name)