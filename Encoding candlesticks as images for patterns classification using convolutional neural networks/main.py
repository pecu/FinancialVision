from sklearn.metrics import confusion_matrix
import numpy as np

from keras import backend as K
from keras import optimizers
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Activation

# customized utilities
from utils import util_processor as pro


def get_model(params):
    model = Sequential()

    # Conv1
    model.add(Conv2D(16, (2, 2), input_shape=(10, 10, 4), padding='same', strides=(1, 1)))
    model.add(Activation('sigmoid'))

    # Conv2
    model.add(Conv2D(16, (2, 2), padding='same', strides=(1, 1)))
    model.add(Activation('sigmoid'))

    # FC
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))

    model.add(Dense(params['classes']))
    model.add(Activation('softmax'))
    model.summary()

    return model


def train_model(params, data):
    model = get_model(params)
    model.compile(loss='categorical_crossentropy', optimizer=params['optimizer'], metrics=['accuracy'])
    hist = model.fit(x=data['train_gaf'], y=data['train_label_arr'],
                     validation_data=(data['val_gaf'], data['val_label_arr']),
                     batch_size=params['batch_size'], epochs=params['epochs'], verbose=2)
    
    return (model, hist)


def print_result(data, model):
    # get train & test pred-labels
    train_pred = model.predict_classes(data['train_gaf'])
    test_pred = model.predict_classes(data['test_gaf'])
    # get train & test true-labels
    train_label = data['train_label'][:, 0]
    test_label = data['test_label'][:, 0]
    # train & test confusion matrix
    train_result_cm = confusion_matrix(train_label, train_pred, labels=range(9))
    test_result_cm = confusion_matrix(test_label, test_pred, labels=range(9))

    print(train_result_cm, '\n', test_result_cm)


if __name__ == "__main__":
    PARAMS = {}
    PARAMS['pkl_name'] = './data/label8_eurusd_10bar_1500_500_val200_gaf_culr.pkl'
    PARAMS['model_name'] = './model/cnn_model_10bar.h5'
    PARAMS['classes'] = 9
    PARAMS['lr'] = 0.01
    PARAMS['epochs'] = 50
    PARAMS['batch_size'] = 64
    PARAMS['optimizer'] = optimizers.SGD(lr=PARAMS['lr'])

    # ---------------------------------------------------------
    # load data & keras model
    data = pro.load_pkl(PARAMS['pkl_name'])

    # train cnn model
    model, hist = train_model(PARAMS, data)
    model.save(PARAMS['model_name'])
    
    # train & test result
    print_result(data, model)
