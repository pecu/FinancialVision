from tqdm import trange, tqdm
from scipy import stats
import pandas as pd
import numpy as np
import time

from utils import util_multi as mul
from utils import util_process as pro


def rename(data):
    rename_dc = {'Gmt time': 'timestamp'}
    data.rename(columns=rename_dc, inplace=True)
    data.columns = [c.lower() for c in data.columns]
    return data


def process_data(data, slope=True):
    '''Including calculation of CLUR, Quartiles, and cus trend
    Args:
        data (dataframe): csv data from assets. With column names open, high, low, close.

    Returns:
        dataframe.
    '''
    if slope:
        # process slpoe
        data['diff'] = data['close'] - data['open']
        data = data.query('diff != 0').reset_index(drop=True)
        data['direction'] = np.sign(data['diff'])
        data['ushadow_width'] = 0
        data['lshadow_width'] = 0

        for idx in trange(len(data)):
            if data.loc[idx, 'direction'] == 1:
                data.loc[idx, 'ushadow_width'] = data.loc[idx, 'high'] - data.loc[idx, 'close']
                data.loc[idx, 'lshadow_width'] = data.loc[idx, 'open'] - data.loc[idx, 'low']
            else:
                data.loc[idx, 'ushadow_width'] = data.loc[idx, 'high'] - data.loc[idx, 'open']
                data.loc[idx, 'lshadow_width'] = data.loc[idx, 'close'] - data.loc[idx, 'low']

            if idx <= 50:
                data.loc[idx, 'body_per'] = stats.percentileofscore(abs(data['diff']), abs(data.loc[idx,'diff']), 'rank')
                data.loc[idx, 'upper_per'] = stats.percentileofscore(data['ushadow_width'], data.loc[idx,'ushadow_width'], 'rank')
                data.loc[idx, 'lower_per'] = stats.percentileofscore(data['lshadow_width'], data.loc[idx,'lshadow_width'], 'rank')
            else:
                data.loc[idx, 'body_per'] = stats.percentileofscore(abs(data.loc[idx-50:idx, 'diff']),abs(data.loc[idx, 'diff']), 'rank')
                data.loc[idx, 'upper_per'] = stats.percentileofscore(data.loc[idx-50:idx, 'ushadow_width'], data.loc[idx, 'ushadow_width'], 'rank')
                data.loc[idx, 'lower_per'] = stats.percentileofscore(data.loc[idx-50:idx, 'lshadow_width'], data.loc[idx, 'lshadow_width'], 'rank')

        data['slope'] = data['close'].rolling(7).apply(pro.get_slope, raw=False)
        data.dropna(inplace=True)
    else:
        # process trend
        data['trend'] = data['slope'].rolling(1).apply(pro.get_trend, raw=False)
        data['previous_trend'] = data['trend'].shift(1).fillna(0)
    return data


def detect_evening_star(data, q=None, multi=False, short_per=35, long_per=65):
    '''Detect evening star pattern    
    Args:
        short_per (int): percentile for determination.
        long_per (int): percentile for determination.

    Returns:
        dataframe.
    '''
    print('[ Info ] : detecting evening star')
    temp = data[(data['previous_trend'] == 1) & (data['direction'] == 1)].index
    data['evening'] = 0
    try:
        for idx in tqdm(temp):
            cond1 = (data.loc[idx, 'body_per'] >= long_per)
            cond2 = (data.loc[idx+1, 'body_per'] <= short_per)
            cond3 = (data.loc[idx+2, 'direction'] == -1)
            cond4 = (data.loc[idx+1, 'close'] + data.loc[idx+1, 'open'])/2 >= data.loc[idx, 'close']
            cond5 = data.loc[idx+2, 'close'] <= ((data.loc[idx, 'open'] + data.loc[idx, 'close'])/2)
            # cond6 = (data.loc[idx+2, 'body_per'] >= long_per)
            cond7 = (data.loc[idx+2, 'open'] <= (data.loc[idx+1, 'open'] + data.loc[idx+1, 'close'])/2)
            if cond1 & cond2 & cond3 & cond4 & cond5 & cond7:
                data.loc[idx+2, 'evening'] = 1
    except:
        pass

    if multi:
        q.put({'evening': np.array(data['evening'])})
    else:
        return data


def detect_morning_star(data, q=None, multi=False, short_per=35, long_per=65):
    '''Detect morning star pattern    
    Args:
        short_per (int): percentile for determination.
        long_per (int): percentile for determination.

    Returns:
        dataframe.
    '''
    print('[ Info ] : detecting morning star')
    temp = data[(data['previous_trend'] == -1) & (data['direction'] == -1)].index
    data['morning'] = 0
    try:
        for idx in tqdm(temp):
            cond1 = (data.loc[idx, 'body_per'] >= long_per)
            cond2 = (data.loc[idx+1, 'body_per'] <= short_per)
            cond3 = (data.loc[idx+2, 'direction'] == 1)
            # cond4 = max(data.loc[idx+1, 'close'], data.loc[idx+1, 'open']) <= data.loc[idx, 'close']
            cond4 = (data.loc[idx+1, 'close'] + data.loc[idx+1, 'open'])/2 <= data.loc[idx, 'close']
            cond5 = data.loc[idx+2, 'close'] >= ((data.loc[idx, 'open'] + data.loc[idx, 'close'])/2)
            # cond6 = (data.loc[idx+2, 'body_per'] >= long_per)
            cond7 = (data.loc[idx+2, 'open'] >= (data.loc[idx+1, 'open'] + data.loc[idx+1, 'close'])/2)
            if cond1 & cond2 & cond3 & cond4 & cond5 & cond7:
                data.loc[idx+2, 'morning'] = 1
    except:
        pass

    if multi:
        q.put({'morning': np.array(data['morning'])})
    else:
        return data


def detect_shooting_star(data, q=None, multi=False, short_per=35, long_per=65):
    '''Detect shooting star pattern    
    Args:
        short_per (int): percentile for determination.
        long_per (int): percentile for determination.

    Returns:
        dataframe.
    '''
    print('[ Info ] : detecting shooting star')
    data['shooting_star'] = 0
    temp = data[(data['previous_trend'] == 1) & (data['direction'] == 1)].index
    try:
        for idx in tqdm(temp):
            cond1 = (data.loc[idx, 'body_per'] >= long_per)
            cond2 = (data.loc[idx, 'direction'] == 1)
            cond3 = (data.loc[idx+1, 'ushadow_width'] > 2 * abs(data.loc[idx+1, 'diff']))
            cond4 = (min(data.loc[idx+1, 'open'], data.loc[idx+1, 'close']) > ((data.loc[idx, 'close'] + data.loc[idx, 'open']) / 2))
            cond5 = (data.loc[idx+1, 'lower_per'] <= short_per - 10)  # 25
            cond6 = (data.loc[idx+1, 'upper_per'] >= long_per)
            if cond1 & cond2 & cond3 & cond4 & cond5 & cond6:
                data.loc[idx+1, 'shooting_star'] = 1
    except:
        pass

    if multi:
        q.put({'shooting_star': np.array(data['shooting_star'])})
    else:
        return data


def detect_hanging_man(data, q=None, multi=False, short_per=35, long_per=65):
    '''Detect hanging man pattern    
    Args:
        short_per (int): percentile for determination.
        long_per (int): percentile for determination.

    Returns:
        dataframe.
    '''
    print('[ Info ] : detecting hanging man')
    data['hanging_man'] = 0
    temp = data[(data['previous_trend'] == 1) & (data['direction'] == 1)].index
    try:
        for idx in tqdm(temp):
            cond1 = (data.loc[idx, 'lshadow_width'] > 2 * abs(data.loc[idx, 'diff']))
            cond2 = (data.loc[idx, 'body_per'] <= short_per)
            cond3 = (data.loc[idx, 'upper_per'] <= (short_per - 10))
            cond4 = (data.loc[idx, 'lower_per'] >= long_per)
            if cond1 & cond2 & cond3 & cond4:
                data.loc[idx, 'hanging_man'] = 1
    except:
        pass

    if multi:
        q.put({'hanging_man': np.array(data['hanging_man'])})
    else:
        return data


def detect_bullish_engulfing(data, q=None, multi=False, short_per=35, long_per=65):
    '''Detect bullish engulfing pattern
    Args:
        short_per (int): percentile for determination.
        long_per (int): percentile for determination.
    
    Returns:
        dataframe.
    '''
    print('[ Info ] : detecting bullish engulfing')
    data['bullish_engulfing'] = 0
    temp = data[(data['previous_trend'] == -1) & (data['direction'] == -1)].index
    try:
        for idx in tqdm(temp):
            cond1 = (data.loc[idx, 'direction'] == -1)
            cond2 = (data.loc[idx, 'body_per'] >= long_per)
            cond3 = (data.loc[idx+1, 'direction'] == 1)
            cond4 = (data.loc[idx+1, 'close'] > data.loc[idx, 'open'])
            cond5 = (data.loc[idx+1, 'open'] < data.loc[idx, 'close'])
            if cond1 & cond2 & cond3 & cond4 & cond5:
                data.loc[idx+1, 'bullish_engulfing'] = 1
    except:
        pass

    if multi:
        q.put({'bullish_engulfing': np.array(data['bullish_engulfing'])})
    else:
        return data


def detect_bearish_engulfing(data, q=None, multi=False, short_per=35, long_per=65):
    '''Detect bearish engulfing pattern
    Args:
        short_per (int): percentile for determination.
        long_per (int): percentile for determination.

    Returns:
        dataframe.
    '''
    print('[ Info ] : detecting bearish engulfing')
    data['bearish_engulfing'] = 0
    temp = data[(data['previous_trend'] == 1) & (data['direction'] == 1)].index
    try:
        for idx in tqdm(temp):
            cond1 = (data.loc[idx, 'direction'] == 1)
            cond2 = (data.loc[idx, 'body_per'] >= long_per)
            cond3 = (data.loc[idx+1, 'direction'] == -1)
            cond4 = (data.loc[idx+1, 'close'] < data.loc[idx, 'open'])
            cond5 = (data.loc[idx+1, 'open'] > data.loc[idx, 'close'])
            if cond1 & cond2 & cond3 & cond4 & cond5:
                data.loc[idx+1, 'bearish_engulfing'] = 1
    except:
        pass

    if multi:
        q.put({'bearish_engulfing': np.array(data['bearish_engulfing'])})
    else:
        return data


def detect_hammer(data, q=None, multi=False, short_per=35, long_per=65):
    '''Detect hammer pattern    
    Args:
        short_per (int): percentile for determination.
        long_per (int): percentile for determination.

    Returns:
        dataframe.
    '''
    print('[ Info ] : detecting hammer')
    data['hammer'] = 0
    temp = data[(data['previous_trend'] == -1) & (data['direction'] == -1)].index
    try:
        for idx in tqdm(temp):
            cond1 = (data.loc[idx, 'lshadow_width'] > 2 * abs(data.loc[idx, 'diff']))
            cond2 = (data.loc[idx, 'body_per'] <= short_per)
            cond3 = (data.loc[idx, 'upper_per'] <= (short_per - 15))
            cond4 = (data.loc[idx, 'lower_per'] >= long_per)
            if cond1 & cond2 & cond3 & cond4:
                data.loc[idx, 'hammer'] = 1
    except:
        pass

    if multi:
        q.put({'hammer': np.array(data['hammer'])})
    else:
        return data    


def detect_inverted_hammer(data, q=None, multi=False, short_per=35, long_per=65):
    '''Detect inverted hammer pattern    
    Args:
        short_per (int): percentile for determination.
        long_per (int): percentile for determination.

    Returns:
        dataframe.
    '''
    print('[ Info ] : detecting inverted hammer')
    data['inverted_hammer'] = 0
    temp = data[(data['previous_trend'] == -1) & (data['direction'] == -1)].index
    try:
        for idx in tqdm(temp):
            cond1 = (data.loc[idx, 'direction'] == -1)
            cond2 = (data.loc[idx, 'body_per'] >= long_per)
            cond3 = (data.loc[idx+1, 'ushadow_width'] > 2 * abs(data.loc[idx+1, 'diff']))
            cond4 = (max(data.loc[idx+1, 'open'], data.loc[idx+1, 'close']) < ((data.loc[idx, 'close'] + data.loc[idx, 'open']) / 2))
            cond5 = (data.loc[idx+1, 'lower_per'] <= short_per)
            cond6 = (data.loc[idx+1, 'upper_per'] >= long_per)
            if cond1 & cond2 & cond3 & cond4 & cond5 & cond6:
                data.loc[idx+1, 'inverted_hammer'] = 1
    except:
        pass

    if multi:
        q.put({'inverted_hammer': np.array(data['inverted_hammer'])})
    else:
        return data    


def detect_bullish_harami(data, q=None, multi=False, short_per=35, long_per=65):
    '''Detect inverted bullish harami pattern    
    Args:
        short_per (int): percentile for determination.
        long_per (int): percentile for determination.
        
    Returns:
        dataframe.
    '''
    print('[ Info ] : detecting bullish harami')
    data['bullish_harami'] = 0
    temp = data[(data['previous_trend'] == -1) & (data['direction'] == -1)].index
    try:
        for idx in tqdm(temp):
            cond1 = (data.loc[idx, 'direction'] == -1)
            cond2 = (data.loc[idx, 'body_per'] >= long_per)
            cond3 = (data.loc[idx+1, 'direction'] == 1)
            cond4 = (data.loc[idx+1, 'close'] >= ((data.loc[idx, 'open'] + data.loc[idx, 'close'])/2))
            cond5 = (data.loc[idx+1, 'close'] < data.loc[idx, 'open'])
            cond6 = (data.loc[idx+1, 'open'] > data.loc[idx, 'close'])
            cond7 = (data.loc[idx+1, 'open'] <= ((data.loc[idx, 'open'] + data.loc[idx, 'close'])/2))
            cond8 = (data.loc[idx+1, 'body_per'] >= long_per)
            if cond1 & cond2 & cond3 & cond4 & cond5 & cond6 & cond7 & cond8:
                data.loc[idx+1, 'bullish_harami'] = 1
    except:
        pass

    if multi:
        q.put({'bullish_harami': np.array(data['bullish_harami'])})
    else:
        return data    


def detect_bearish_harami(data, q=None, multi=False, short_per=35, long_per=65):
    '''Detect inverted bearish harami pattern    
    Args:
        short_per (int): percentile for determination.
        long_per (int): percentile for determination.

    Returns:
        dataframe.
    '''
    print('[ Info ] : detecting bearish harami')
    data['bearish_harami'] = 0
    temp = data[(data['previous_trend'] == 1) & (data['direction'] == 1)].index
    try:
        for idx in tqdm(temp):
            cond1 = (data.loc[idx, 'direction'] == 1)
            cond2 = (data.loc[idx, 'body_per'] >= long_per)
            cond3 = (data.loc[idx+1, 'direction'] == -1)
            cond4 = (data.loc[idx+1, 'close'] <= ((data.loc[idx, 'open'] + data.loc[idx, 'close'])/2))
            cond5 = (data.loc[idx+1, 'close'] > data.loc[idx, 'open'])
            cond6 = (data.loc[idx+1, 'open'] < data.loc[idx, 'close'])
            cond7 = (data.loc[idx+1, 'open'] >= ((data.loc[idx, 'open'] + data.loc[idx, 'close'])/2))
            cond8 = (data.loc[idx+1, 'body_per'] >= long_per)
            if cond1 & cond2 & cond3 & cond4 & cond5 & cond6 & cond7 & cond8:
                data.loc[idx+1, 'bearish_harami'] = 1
    except:
        pass

    if multi:
        q.put({'bearish_harami': np.array(data['bearish_harami'])})
    else:
        return data       


def detect_all(data, tasks_ls=None, multi=False, pro_num=2):
    '''
    Args:
        data (dataframe): csv data after `process_data` function.
        multi (bool): use multiprocessing or not.
        pro_num (int): how many processes to be used.

    Returns:
        data (dataframe): dataframe with detections.
    '''
    if multi:
        res_ls = mul.auto_multi(data, tasks_ls, pro_num)
        print('[ Info ] join finished !')

        dc = {}
        for res in res_ls:
            for key, value in res.items():
                dc[key] = value
        df = pd.DataFrame(dc)
        data = pd.concat([data, df], axis=1)
    else:
        data = detect_evening_star(data)
        data = detect_morning_star(data)
        data = detect_shooting_star(data)
        data = detect_hanging_man(data)
        data = detect_bullish_engulfing(data)
        data = detect_bearish_engulfing(data)
        data = detect_hammer(data)
        data = detect_inverted_hammer(data)
        data = detect_bullish_harami(data)
        data = detect_bearish_harami(data)
    return data


def detection_result(data):
    '''Print numbers of detection    
    Args:
        data (dataframe): csv data after `process_data` function.

    Returns:
        data (dataframe): dataframe with detections.
    '''
    print('\n[ Info ] : number of evening star is %s' % np.sum(data['evening']))
    print('[ Info ] : number of morning star is %s' % np.sum(data['morning']))
    print('[ Info ] : number of shooting star is %s' % np.sum(data['shooting_star']))
    print('[ Info ] : number of hanging man is %s' % np.sum(data['hanging_man']))
    print('[ Info ] : number of bullish engulfing is %s' % np.sum(data['bullish_engulfing']))
    print('[ Info ] : number of bearish engulfing is %s' % np.sum(data['bearish_engulfing']))
    print('[ Info ] : number of hammer is %s' % np.sum(data['hammer']))
    print('[ Info ] : number of inverted hammer is %s' % np.sum(data['inverted_hammer']))
    print('[ Info ] : number of bullish harami is %s' % np.sum(data['bullish_harami']))
    print('[ Info ] : number of bearish harami is %s' % np.sum(data['bearish_harami']))


if __name__ == "__main__":
    TASLS_LS = [detect_evening_star, detect_morning_star, detect_shooting_star, 
                detect_hanging_man, detect_bullish_engulfing, detect_bearish_engulfing,
                detect_hammer, detect_inverted_hammer, detect_bullish_harami,
                detect_bearish_harami]
    
    # load raw ohlc data
    data = pd.read_csv('./data/eurusd_2010_2017_raw.csv')
    data = rename(data)

    # calculate features & slope
    data = process_data(data, slope=True)

    # calculate trend (depend on slopes)
    data = process_data(data, slope=False)

    # save current data
    # data.to_csv('./data/eurusd_2010_2017_process.csv', index=False)

    # detect with customized rules
    data = detect_all(data, TASLS_LS, multi=True, pro_num=4)
    data.to_csv('./data/eurusd_2010_2017_patterns.csv', index=False)
    detection_result(data)
