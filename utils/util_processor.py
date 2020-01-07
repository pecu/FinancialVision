import numpy as np
import pickle


def load_pkl(pkl_name):
    '''
    Args:
        pkl_name (string): path for pickle.
    
    Returns:
        (dict): including following structure
            `raw time-series data` (N, 32, 4):
                'train_data', 'val_data', 'test_data'
            `gasf data` (N, 32, 32, 4):
                'train_gaf', 'val_gaf', 'test_gaf'
            `label data` (N, 3):
                'train_label', 'val_label', 'test_label',
            `one-hot label data` (N, 9):
                'train_label_arr', 'val_label_arr', 'test_label_arr'
    '''
    # load data from data folder
    with open(pkl_name, 'rb') as f:
        data = pickle.load(f)
    return data


def gasf2ts(arr):
    '''
    Args:
        arr (numpy array):  (32, 32)
    Returns:
        numpy.series: (1d)
    '''
    # Get element from diagonal
    diag_v = np.zeros((arr.shape[0],))
    for i in range(arr.shape[0]):
        diag_v[i] = arr[i, i]
    # Inverse to Arc
    diag_v_arc = np.arccos(diag_v) / 2
    # Inverse to Normalized ts
    ts = np.cos(diag_v_arc)
    return ts


def ohlc2culr(ohlc):
    '''
    Args:
        ohlc (numpy): (N, ts_n, 4)

    Returns:
        culr (numpy): (N, ts_n, 4)
    '''
    culr = np.zeros((ohlc.shape[0], ohlc.shape[1], ohlc.shape[2]))
    culr[:, :, 0] =  ohlc[:, :, -1]
    culr[:, :, 1] = ohlc[:, :, 1] - np.maximum(ohlc[:, :, 0], ohlc[:, :, -1])
    culr[:, :, 2] = np.minimum(ohlc[:, :, 0], ohlc[:, :, -1]) - ohlc[:, :, 2]
    culr[:, :, 3] = ohlc[:, :, -1] - ohlc[:, :, 0]
    return culr


def culr2ohlc(culr_n, culr):
    '''
    Args:
        culr_n (numpy): (N, ts_n, 4)
        culr (numpy): (N, ts_n, 4)

    Returns:
        ohlc (numpy): (N, ts_n, 4)
    '''
    ohlc = np.zeros((*culr_n.shape, ))
    for i in range(culr_n.shape[0]):
        for c in range(culr_n.shape[-1]):
            # get min & max from data before normalized
            each_culr = culr[i, :, c]
            min_v = np.amin(each_culr)
            max_v = np.amax(each_culr)
            # inverse normalization
            each_culr_n = culr_n[i, :, c]
            culr_n[i, :, c] = (each_culr_n * (max_v - min_v)) + min_v
        # convert culr to ohlc
        ohlc[i, :, -1] = culr_n[i, :, 0]
        ohlc[i, :, 0] = ohlc[i, :, -1] - culr_n[i, :, -1]
        ohlc[i, :, 1] = culr_n[i, :, 1] + np.maximum(ohlc[i, :, 0], ohlc[i, :, -1])
        ohlc[i, :, 2] = np.minimum(ohlc[i, :, 0], ohlc[i, :, -1]) - culr_n[i, :, 2]
    return ohlc


def load_each_class(data, target):
	x_train, y_train, time_series_train = [None] * 3
	label = data["train_label"][:, 0]

	cond = (label == target)
	x_train = data['train_gaf'][cond, :, :]
	y_train = data['train_label_arr'][cond]
	time_series_train = data["train_data"][cond]

	return (x_train, y_train, time_series_train)