import pandas as pd
import numpy as np
import talib

class Market(object):
    def __init__(self, data_path, skip_volume_0=True, indicators={}, pip=0.00001, start_index=0):
        '''
        data_path: [string]
        skip_volume_0: [boolean]
        indicators: {[string, indicator_name] : [int, window_size], ...}
        '''
        self.raw_data = pd.read_csv(data_path)
        self.raw_data['Gmt time'] = pd.to_datetime(self.raw_data['Gmt time'])
        self.pip = pip
        
        self.reset(skip_volume_0=skip_volume_0, indicators=indicators, start_index=start_index)

    def _preprocess(self):
        self.data = self.raw_data.copy()
        if self.skip_volume_0: # delete all volume=0 data or not
            self.data = self.data[['Gmt time','Open','High','Low','Close','Volume']][ self.data['Volume'].apply(lambda x: x > 0) ]
        if len(self.indicators) > 0:
            for ind in self.indicators.keys():
                if ind == 'ATR':
                    self.data['ATR'] = talib.ATR(self.data['High'],self.data['Low'],self.data['Close'],timeperiod=self.indicators[ind])
                elif ind == 'ADX':
                    self.data['ADX'] = talib.ADX(self.data['High'],self.data['Low'],self.data['Close'],timeperiod=self.indicators[ind])
                elif ind == 'RSI':
                    self.data['RSI'] = talib.RSI(self.data['Close'],timeperiod=self.indicators[ind])
            
        self.data.dropna(inplace=True)
        self.data.reset_index(inplace=True, drop=True)
    
    def reset(self, skip_volume_0=None, indicators=None, start_index=0):
        if skip_volume_0 != None:
            self.skip_volume_0 = skip_volume_0
        if indicators != None:
            self.indicators = indicators
        self.current_index = start_index - 1
        self._preprocess()

    def next(self):
        self.current_index += 1
        return self.current_index < len(self.data) - 1 #preventing from boundary-issues

    def activate_check(self, order):
        '''
        order: [dict]
        '''
        if not order['activated']:
            price_range = (self.data['Low'][self.current_index],self.data['High'][self.current_index])
            if self._price_check(order['price'], price_range): #check if can buy/sell
                return True
        return False

    def SLTP_check(self, order):
        '''
        order: [dict]
        '''
        if order['activated']:
            price_range = (self.data['Low'][self.current_index],self.data['High'][self.current_index])
            if order['activated']:
                if self._price_check(order['TP'], price_range): #check if reached TP
                    return 'TP'
                elif self._price_check(order['SL'], price_range): #check if reached SL
                    return 'SL'
        return False

    def _price_check(self, the_price, price_range):
        return price_range[0] <= the_price and the_price <= price_range[1]

    def get_ohlc(self, size=1):
        if size <= 1:
            return self.data[['Open','High','Low','Close']].iloc[self.current_index]
        else:
            return self.data[['Open','High','Low','Close']].iloc[self.current_index-size+1:self.current_index+1]

    def get_market_price(self):
        return self.data['Close'].iloc[self.current_index]

    def get_indicators(self, size=1):
        if size <= 1:
            return self.data[[ind for ind in self.indicators.keys()]].iloc[self.current_index]
        else:
            return self.data[[ind for ind in self.indicators.keys()]].iloc[self.current_index-size+1:self.current_index+1]

    def get_datetime(self):
        return self.data['Gmt time'].iloc[self.current_index]

    def get_pip(self, q=1):
        return q*self.pip

    def get_current_index(self):
        return self.current_index

    def get_data(self):
        return self.data

    def get_data_length(self):
        return len(self.data)
