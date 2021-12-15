import pandas as pd
import numpy as np


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

def inverse_minmax(ts_n, min_v, max_v):
    ts = ts_n * (max_v - min_v) + min_v
    return ts
