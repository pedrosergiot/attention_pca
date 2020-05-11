import os
os.environ['KERAS_BACKEND'] = 'theano'

from keras.models import Sequential
from keras.layers import Dense, Dropout
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


n_neurons1 = range(1, 11)
n_neurons2 = range(0, 11)
num_inputs = 10     # considering 5 electrodes

for neuron1 in n_neurons1:
    for neuron2 in n_neurons2:
        model = create_model(n_features=num_inputs, neurons1=neuron1, neurons2=neuron2)

        file_name = "user.pedrosergiot.attention_pca_individual/" \
                    "user.pedrosergiot.attention_pca_individual." \
                    "neurons1_{}.neurons2_{}.pkl".format(neuron1, neuron2)

        dirname = os.path.dirname(file_name)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        dump(model, file_name)

