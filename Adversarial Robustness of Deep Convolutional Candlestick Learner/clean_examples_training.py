from sklearn.metrics import confusion_matrix
import numpy as np
import pickle

from keras import backend as K
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Activation, Dropout
from keras import callbacks as cb

# customized utilities
from utils import util_process as pro


def get_model(params):
    model = Sequential()

    # Conv1
    model.add(Conv2D(20, (4, 4), input_shape=(10, 10, 4), padding='valid', strides=(1, 1)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    # Conv2
    model.add(Conv2D(20, (4, 4), padding='valid', strides=(1, 1)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
 
    # FC
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(params['classes']))
    model.add(Activation('softmax'))
    model.summary()

    return model


def train_model(params):
    model = get_model(params)
    model.compile(loss='categorical_crossentropy', optimizer=params['optimizer'], metrics=['accuracy'])
    # callback objects
    early_stopping = cb.EarlyStopping(monitor='val_loss', patience = params['patience'],
                     restore_best_weights=True, verbose=2)
    saveBestModel = cb.ModelCheckpoint(params['model_path'], monitor='val_loss',
                     save_weights_only=False , save_best_only=True, mode='auto', verbose=1)
    callback_ls = [early_stopping, saveBestModel]
    class_weight = {0: params['zero_weight'], 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1}
    hist = model.fit(x=params['data']['train_gaf'], y=params['data']['train_label_arr'],
                     validation_data = (params['data']['val_gaf'], params['data']['val_label_arr']),
                     batch_size = params['batch_size'], epochs = params['epochs'],
                     class_weight = class_weight,
                     callbacks = callback_ls, verbose=2)

    return (model, hist)


def collect_result(params, model, hist, result_dict):
    # get train & test pred-labels
    train_pred = model.predict_classes(params['data']['train_gaf'])
    test_pred = model.predict_classes(params['data']['test_gaf'])

    # get train & test true-labels
    train_label = params['data']['train_label'][:, 0]
    test_label = params['data']['test_label'][:, 0]

    # train & test confusion matrix
    train_result_cm = confusion_matrix(train_label, train_pred, labels=range(params['classes']))
    test_result_cm = confusion_matrix(test_label, test_pred, labels=range(params['classes']))
    print(train_result_cm, '\n')
    print(test_result_cm, '\n')

    total_acc = 0
    for i in range(params['classes']):
        total_acc += test_result_cm[i, i]
    total_acc /= 5000
    print('[ Info ] average testing accuracy : %s' % total_acc)
    
    result_dict['train_result_cm'].append(train_result_cm)
    result_dict['test_result_cm'].append(test_result_cm)
    result_dict['total_acc'].append(total_acc)


def main(params):
    # loop training
    result_dict = {'train_result_cm': [], 'test_result_cm': [], 'total_acc': []}
    for i in range(params['n_loop']):
        model, hist = train_model(params)
        # collect testing results
        collect_result(params, model, hist, result_dict)

    # save all results to pickle file
    filename = './results/clean_results_%s.pkl' % params['n_loop']
    with open(filename, 'wb') as f:
        pickle.dump(result_dict, f)


if __name__ == "__main__":
    PARAMS = {}
    PARAMS['data'] = pro.load_pkl('./data/label8_eurusd_10bar_1500_500_val200_gaf_culr.pkl') 
    PARAMS['classes'] = 9
    PARAMS['lr'] = 0.01
    PARAMS['epochs'] = 1000
    PARAMS['batch_size'] = 64
    PARAMS['patience'] = 80
    PARAMS['zero_weight'] = 2
    PARAMS['optimizer'] = optimizers.SGD(lr=PARAMS['lr'])
    PARAMS['n_loop'] = 100
    PARAMS['model_path'] = './models/orgdata_model.h5'
    
    main(PARAMS)