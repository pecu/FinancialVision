import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import multiprocessing as mp
import mpl_finance as mpf
import numpy as np
import pandas as pd
import pickle
import time
import os

# customized utilities
from utils import util_processor as util_pro
from Attack import *

from keras.models import load_model
from keras import optimizers


def process_data(paras):
    # (N, 10, 10, 4) / ***(N,) / (N, 10, 4)
    paras['train_x'], paras['train_label'], paras['train_ohlc'] = util_pro.load_data(paras['data_fn'], False)
    return paras

def extract_atk_labels_data(paras):
    # join all conditions
    labels = paras['train_label']
    cond = np.zeros(len(labels), dtype=bool)
    for l in paras['atk_labels']:
        cond += (labels == l)
    # extract data
    paras['train_x'] = paras['train_x'][cond]
    paras['train_label'] = paras['train_label'][cond]
    paras['train_ohlc'] = paras['train_ohlc'][cond]
    return paras

def get_pairs(iter_pair1, iter_pair2):
    return [(p1, p2) for p1 in iter_pair1 for p2 in iter_pair2]

def get_cdl_chn_pairs(paras):
    if paras['mode'] == 0:
        cdl_chn_pairs = get_pairs(paras['atk_candles'], paras['atk_channels'])
    elif paras['mode'] == 1:
        cdl_chn_pairs = paras['atk_cus_pairs']
    else:
        raise ValueError('Invalid parameter mode. Integer 0 or 1 only.')
    return cdl_chn_pairs

def job(paras, candle, channel, idx=None, q=None):
    model = load_model(paras['model_fn'], compile=False)
    attacker = Attacker(inputs = (paras['train_x'], paras['train_label'], paras['train_ohlc']),
                        model = model, saver_ls = paras['saver_ls'],
                        atk_candle = candle, atk_channel = channel)
    adv_res = attacker.attackAll()
    q.put({'_idx': idx, 'idx': (candle, channel), 'data': adv_res})
    print('*** total detection: %s' % len(adv_res['suc_gaf']))

def multi_process(paras, tasks, pro_num):
    tasks_num = len(tasks)
    pro_ls, res_ls = [], []
    run_n, next_idx = 0, 0

    q = mp.Queue()
    for idx, task in enumerate(tasks):
        p = mp.Process(target=job, args=(paras, task[0], task[1], idx, q,))
        pro_ls.append(p)
    
    while True:
        if len(res_ls) >= tasks_num:
            del pro_ls
            break

        if q.qsize():
            print('[ Info ] start joinning processor ...')
            data = q.get()
            res_ls.append(data)
            pro_ls[data['_idx']].join()
            run_n -= 1
            print('[ Info ] finish joinning processor %s !' % data['_idx'])
            print('[ info ] current data collection: %s / %s' % (len(res_ls), tasks_num))

        if (tasks_num - len(res_ls) - run_n > 0) & (run_n < pro_num):
            if tasks_num < pro_num:
                diff = tasks_num
            else:
                diff = pro_num - run_n
            for _ in range(diff):
                pro_ls[next_idx].start()
                run_n += 1
                next_idx += 1
    return res_ls

def main(paras):
    # preprocess data
    paras = process_data(paras)
    # extract attacked labels
    paras = extract_atk_labels_data(paras)
    # combine candles & channels as tasks
    cdl_chn_pairs = get_cdl_chn_pairs(paras)
    # attack all tasks through multiple processors
    res_ls = multi_process(paras, cdl_chn_pairs, paras['pro_num'])
    # intergate data from all processors & save
    adv_res = util_pro.aggregate_multi_res(res_ls, save=True, fn=paras['res_fn'])
    print(len(adv_res['suc_gaf']))
    count_res = util_pro.print_attack_results(paras, adv_res)

if __name__ == '__main__':
    '''
    - To use the adversarial attacker without controls:
        - set `mode` = 0
        - control attack labels at `atk_labels`
        - control attack candles at `atk_candles`
        - control attack channels at `atk_channels`
    - To use the adversarial attacker with controls:
        - set `mode` = 1, and only the combinations of `atk_labels`, `atk_candles`, and `atk_channels` will be attacked
    - All perturbated data will automatically collect by generator to pickle file `res_fn`
    '''
    PARAS = dict()
    PARAS['data_fn'] = './data/ETH_gaf.pkl'
    PARAS['model_fn'] = './model/kmodel_0.9531.h5'
    PARAS['res_fn'] = './data/adv_res.pkl'
    PARAS['saver_ls'] = ['suc_gaf', 'suc_label', 'suc_logit', 'org_ohlc', 'org_label', 'atk_candle', 'atk_channel']
    PARAS['pro_num'] = 6
    PARAS['mode'] = 0  # 0: attack & collect all combinations / 1: only atk_cus_pairs
    PARAS['atk_labels'] = [0, 1, 2, 3, 4, 5, 6, 7]
    # --------------- mode 0 ---------------
    PARAS['atk_candles'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # All
    PARAS['atk_channels'] = [0, 1, 2, 3]  # CULR
    # --------------- mode 1 ---------------
    PARAS['atk_cus_pairs'] = [(7, 0), (7, 1), (8, 2), (8, 3), (9, 0), (9, 3)]

    start_time = time.time()
    main(PARAS)
    print('[ info ] total runtime: %.2f mins' % ((time.time() - start_time) / 60))

