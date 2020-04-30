__all__ = ["myBasicDense","create_model","CustomWrapper"]

#from tensorflow.python.keras.models import Sequential
#from tensorflow.python.keras.layers import Dense, Dropout
#from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
#from tensorflow.python.keras.callbacks import EarlyStopping

import os
os.environ['KERAS_BACKEND'] = 'theano'

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.base import BaseEstimator, ClassifierMixin


# Defines wrapper for the dense model created
def myBasicDense(n_features, neurons1, neurons2):
    # defining criteria for early stopping (based on the val_loss and with patience of 50 iterations)
    es = EarlyStopping(monitor='loss', mode='min', verbose=0, patience=20, restore_best_weights=True)

    return KerasClassifier(build_fn=create_model, n_features=n_features,
                           neurons1=neurons1, neurons2=neurons2,
                           epochs=500, batch_size=50, callbacks=[es], verbose=0)


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
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Custom wrapper used to bound n_features to n_components
class CustomWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, n_features=10, neurons1=1, neurons2=0):
        self.n_features = n_features
        self.neurons1 = neurons1
        self.neurons2 = neurons2

        # This n_features is passed to both your parts of the pipeline
        self.pipe = Pipeline([('scale', StandardScaler()),
                              ('reducer', PCA(n_components=n_features)),
                              ('standard', MaxAbsScaler()),
                              ('model', myBasicDense(n_features=n_features, neurons1=neurons1, neurons2=neurons2))])

    def fit(self, X, y):
        self.pipe.fit(X, y)
        return self

    def predict(self, X):
        return self.pipe.predict(X)

    def set_params(self, **params):
        super(CustomWrapper, self).set_params(**params)
        self.pipe = Pipeline([('scale', StandardScaler()),
                              ('reducer', PCA(n_components=self.n_features)),
                              ('standard', MaxAbsScaler()),
                              ('model', myBasicDense(n_features=self.n_features,
                                                     neurons1=self.neurons1,
                                                     neurons2=self.neurons2))])
        return self