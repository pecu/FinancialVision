import pandas as pd
import numpy as np
import pickle

def load_pkl(pkl_name):
    '''
    Args:
        pkl_name (string): path of pickle name

    Returns:
        data (dict): with the following keys
            "train_data" (numpy): (train_N, 32, 4)
            "train_gaf" (numpy): (train_N, 32, 32, 4)
            "train_label" (numpy): (train_N, 3)
            "train_label_arr" (numpy): (train_N, 9)
            "val_data" (numpy): (val_N, 32, 4)
            "val_gaf" (numpy): (val_N, 32, 32, 4)
            "val_label" (numpy): (val_N, 3)
            "val_label_arr" (numpy): (val_N, 9)
            "test_data" (numpy): (test_N, 32, 4)
            "test_gaf" (numpy): (test_N, 32, 32, 4)
            "test_label" (numpy): (test_N, 3)
            "test_label_arr" (numpy): (test_N, 9)
    '''
    with open(pkl_name, "rb") as f:
        data = pickle.load(f)
    return data


def ohlc2culr(ohlc):
    '''
    Args:
        ohlc (numpy): (N, ts_n, 4)

    Returns:
        culr (numpy): (N, ts_n, 4)
    '''
    culr = np.zeros(ohlc.shape)
    culr[:, :, 0] =  ohlc[:, :, -1]
    culr[:, :, 1] = ohlc[:, :, 1] - np.maximum(ohlc[:, :, 0], ohlc[:, :, -1])
    culr[:, :, 2] = np.minimum(ohlc[:, :, 0], ohlc[:, :, -1]) - ohlc[:, :, 2]
    culr[:, :, 3] = ohlc[:, :, -1] - ohlc[:, :, 0]
    return culr


def ts2gasf(ts, max_v, min_v):
    '''
    Args:
        ts (numpy): (N, )
        max_v (int): max value for normalization
        min_v (int): min value for normalization

    Returns:
        gaf_m (numpy): (N, N)
    '''
    # Normalization : 0 ~ 1
    if max_v == min_v:
        gaf_m = np.zeros((len(ts), len(ts)))
    else:
        ts_nor = np.array((ts-min_v) / (max_v-min_v))
        # Arccos
        ts_nor_arc = np.arccos(ts_nor)
        # GAF
        gaf_m = np.zeros((len(ts_nor), len(ts_nor)))
        for r in range(len(ts_nor)):
            for c in range(len(ts_nor)):
                gaf_m[r, c] = np.cos(ts_nor_arc[r] + ts_nor_arc[c])
    return gaf_m


def gasf2ts(gasf_m):
    '''
    Args:
        gasf_m (numpy): (N, N)

    Returns:
        ts (numpy): (N, )
    '''
    # Get element from diagonal
    diag_v = np.zeros((gasf_m.shape[0],))
    for i in range(gasf_m.shape[0]):
        diag_v[i] = gasf_m[i, i]
    # Inverse to Arc
    diag_v_arc = np.arccos(diag_v) / 2
    # Inverse to Normalized ts
    ts = np.cos(diag_v_arc)
    return ts


def get_gasf(arr):
    '''Convert time-series to gasf    
    Args:
        arr (numpy): (N, ts_n, 4)

    Returns:
        gasf (numpy): (N, ts_n, ts_n, 4)

    Todos:
        add normalization together version
    '''
    arr = arr.copy()
    gasf = np.zeros((arr.shape[0], arr.shape[1], arr.shape[1], arr.shape[2]))
    for i in range(arr.shape[0]):
        for c in range(arr.shape[2]):
            each_channel = arr[i, :, c]
            c_max = np.amax(each_channel)
            c_min = np.amin(each_channel)
            each_gasf = ts2gasf(each_channel, max_v=c_max, min_v=c_min)
            gasf[i, :, :, c] = each_gasf
    return gasf


def get_label(label):
    '''
    Args:
        label (numpy): (N, 3)

    Returns:
        label (numpy): (N, )
    '''
    label = label[:, 0].astype('int32')
    return label


def onehot2cate(onehot):
    '''
    Args:
        onehot (numpy): (N, class_n)

    Returns:
        cate (numpy): (N, )
    '''
    cate = np.argmax(onehot, axis=1)
    return cate


def get_error_data(pred_label, true_label, test_ts, label=None):
    '''
    Args:
        pred_label (numpy): (N, )
        true_label (numpy): (N, )
        test_ts (numpy): (N, 32, 4)
        label (int): which true label to focus

    Returns:
        error_pred_label_l (numpy): (error_n, )
        error_test_ts_l (numpy): (error_n, 32, 4)
    '''
    label_cond = (true_label == label)
    pred_label_l = pred_label[label_cond]
    true_label_l = true_label[label_cond]
    test_ts_l = test_ts[label_cond, :, :]

    error_cond = (pred_label_l != true_label_l)
    error_pred_label_l = pred_label_l[error_cond]
    error_test_ts_l = test_ts_l[error_cond, :, :]
    return (error_pred_label_l, error_test_ts_l)


def get_target_error(error_pred, error_ts, label=None):
    '''
    Args:
        error_pred (numpy): (error_n, )
        error_ts (numpy): (error_n, )

    Returns:
        target_error_pred (numpy): (target_error_n, )
        target_error_ts (numpy): (target_error_n, 32, 4)   
    '''
    label_cond = (error_pred == label)
    target_error_pred = error_pred[label_cond]
    target_error_ts = error_ts[label_cond]
    return (target_error_pred, target_error_ts)


def zero_feature(gaf_arr, *feature):
    '''Turn the feature to zero    
    Args:
        gaf_arr (numpy array): (N, 32, 32, 4)
        feature (int): the feature want to be zero

    Returns:
        gaf_arr (numpy): (N, 32, 32, 4)
    '''
    for i in feature:
        gaf_arr[:,:,:,i] = 0
    return gaf_arr


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

def generate_time_interval_csv(file):
    '''use for cgan
    Args:
        raw data: csv

    Outputs:
        5M / 30M / 1D / 1W data: csv
    '''
    df = pd.read_csv(file, parse_dates=['Gmt time'], index_col='Gmt time')

    form = ['5min', '30min', '1H', '1D']
    filename = ['5M', '30M', '1H', '1D']
    for i in form:
        new_open = df.Open.resample(i).first()
        new_high = df.High.resample(i).max()
        new_low = df.Low.resample(i).min()
        new_close = df.Close.resample(i).last()
        new_vol = df.Volume.resample(i).sum()
        
        new_df = pd.DataFrame({'open': new_open, 'high': new_high, 'low': new_low, 'close': new_close, 'volume': new_vol, 'time': new_open.index})
        new_df = new_df.dropna().reset_index(drop = "TRUE")
        
        new_df.to_csv('./data/' + filename[form.index(i)] + "_data.csv", index = False)
