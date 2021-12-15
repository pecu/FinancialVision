import numpy as np
import pickle

from six import viewvalues


def load_pkl(fn):
    with open(fn, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pkl(data, fn):
    with open(fn, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    return data

def load_data(fn, garbage=False):
    data = load_pkl(fn)
    print(data.keys())
    # remove garbage label
    if garbage:
        garbage_cond = (data['train_label'][:, 0] != 0)
        train_x = data['train_culr_gaf'][garbage_cond]
        train_label = data['train_onehot'][garbage_cond].argmax(axis=-1) - 1
        train_ohlc = data['train_ohlc'][garbage_cond]
    else:
        train_x = data['train_culr_gaf']
        train_label = data['train_onehot'].argmax(axis=-1)
        train_ohlc = data['train_ohlc']
    return (train_x, train_label, train_ohlc)

class Data_saver:
    def __init__(self, data_saver_ls):
        '''    
        Args:
            data_saver_ls (list): pure list
        '''
        for key in data_saver_ls:
            setattr(self, key, [])

    def set(self, target, values):
        '''Save data to saver
        Args:
            target (string): key of data
            values (list): pure list
        '''
        getattr(self, target).extend(values)
    
    def setAll(self, adv_res):
        '''Save all data to saver via dict
        Args:
            adv_res (dict): result of adv
        '''
        for k, v in adv_res.items():
            getattr(self, k).extend(v)
    
    def get(self, target):
        '''Extract data out of saver
        Args:
            target (string): key of data
        '''
        res = getattr(self, target)
        return res

def aggregate_multi_res(res_ls, save=True, fn=None):
    adv_res = {'suc_gaf': [], 'suc_label': [], 'suc_logit': [], 'org_ohlc': [],
               'org_label': [], 'atk_candle': [], 'atk_channel': []}
    for p in res_ls:
        n_sample = len(p['data']['suc_gaf'])
        adv_res['atk_candle'].extend([p['idx'][0]] * n_sample)
        adv_res['atk_channel'].extend([p['idx'][1]] * n_sample)
        for i in range(n_sample):
            adv_res['suc_gaf'].append(p['data']['suc_gaf'][i])
            adv_res['suc_label'].append(p['data']['suc_label'][i])
            adv_res['suc_logit'].append(p['data']['suc_logit'][i])
            adv_res['org_ohlc'].append(p['data']['org_ohlc'][i])
            adv_res['org_label'].append(p['data']['org_label'][i])
    # convert to np array format
    for k, v in adv_res.items():
        adv_res[k] = np.array(v)
    if save:
        save_pkl(adv_res, fn)
    return adv_res

def print_attack_results(paras, adv_res):
    res = dict()
    for l in paras['atk_labels']:
        tmp_dc = dict()
        for cdl in paras['atk_candles']:
            for chl in paras['atk_channels']:
                tmp_dc[str(cdl) + str(chl)] = 0
        res[l] = tmp_dc
    
    for i in range(len(adv_res['suc_gaf'])):
        l = adv_res['org_label'][i]
        cdl = adv_res['atk_candle'][i]
        chl = adv_res['atk_channel'][i]
        res[l][str(cdl) + str(chl)] += 1
    return res
    