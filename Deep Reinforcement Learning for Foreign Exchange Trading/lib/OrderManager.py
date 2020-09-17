import logging

class OrderManager(object):
    def __init__(self, market, record):
        '''
        market: [Market]
        record: [Record]
        '''
        self.market = market #important! call by reference
        self.record = record #important! call by reference
        self.orders = []
        self.SerialNumber = 0

    def reset(self, new_record=None):
        if new_record != None:
            self.record = new_record
        self.clear_orders()
        self.SerialNumber = 0
    
    def clear_orders(self):
        if len(self.orders) > 0: #close/cancel all orders before reset
            for order in self.orders:
                self._close_order(order)
        self.orders = [] #the internal function_close_order(order) will not clear the list for you...

    def create_order(self, quantity, price=0, order_type='BUY', SL_pip=0, TP_pip=0, price_diff=0):
        '''
        order_type: 'BUY' or 'SELL'
        price: [float]=limit price, 0=market price
        SL_pip: [int]=profit pips(+), 0=None
        TP_pip: [int]=loss pips(+), 0=None
        price_diff: [int]=pips(+/-), for the use of limit order (market_price+price_diff)
        '''
        if price == 0:
            price = self.market.get_market_price() + self.market.get_pip()*price_diff
        if order_type == 'BUY':
            SL = -1e9 if SL_pip == 0 else -SL_pip*self.market.get_pip()+price
            TP = 1e9 if TP_pip == 0 else price+TP_pip*self.market.get_pip()
        elif order_type == 'SELL':
            SL = 1e9 if SL_pip == 0 else price+SL_pip*self.market.get_pip()
            TP = -1e9 if TP_pip == 0 else -TP_pip*self.market.get_pip()+price

        this_order = {
                        'SN': self.SerialNumber,
                        'order_type': order_type,
                        'quantity': quantity,
                        'price': price,
                        'SL': SL,
                        'TP': TP,
                        'activated': False,
                        'create_time': self.market.get_datetime()
                    }
        
        self.SerialNumber += 1
        self._add_order(this_order)
        return this_order['SN']

    def _add_order(self, this_order):
        if self._SN_valid(this_order) and self._price_valid(this_order) and \
            self._SL_valid(this_order) and self._TP_valid(this_order):
            self.orders.append(this_order)
    
    def _SN_valid(self, this_order): #check if SN is duplicated
        for order in self.orders:
            if order['SN'] == this_order['SN']:
                return False
        return True

    def _price_valid(self, this_order): #check if price is in market
        if this_order['price'] <= 0:
            logging.warning('OrderUnvalid: price must be greater than zero')
            return False
        else:
            return True

    def _SL_valid(self, this_order): #check if SL is too close to price or unreasonable
        if this_order['order_type'] == 'BUY':
            if this_order['SL'] >= this_order['price']-self.market.get_pip(2): #must be greater than 2 pips
                logging.warning('OrderUnvalid: stop loss must be 2 pips less than buy price')
                return False
        elif this_order['order_type'] == 'SELL':
            if this_order['SL'] <= this_order['price']+self.market.get_pip(2): #must be greater than 2 pips
                logging.warning('OrderUnvalid: stop loss must be 2 pips greater than buy price')
                return False
        return True

    def _TP_valid(self, this_order): #check if TP is too close to price or unreasonable
        if this_order['order_type'] == 'BUY':
            if this_order['TP'] <= this_order['price']+self.market.get_pip(2): #must be greater than 2 pips
                logging.warning('OrderUnvalid: take profit must be 2 pips greater than buy price')
                return False
        elif this_order['order_type'] == 'SELL':
            if this_order['TP'] >= this_order['price']-self.market.get_pip(2): #must be greater than 2 pips
                logging.warning('OrderUnvalid: take profit must be 2 pips less than buy price')
                return False
        return True
    
    def orders_check(self):
        finished_order_index = []
        for i in range(len(self.orders)):
            this_order = self.orders[i]
            if self.market.activate_check(this_order): #if order can be activated
                self.orders[i]['activate_time'] = self.market.get_datetime()
                self.orders[i]['activated'] = True
                self.record.order_activated(self.orders[i]) #record buy/sell
                logging.info('OrderActivated#%04d: %s@%f (TP=%f,SL=%f)'%(self.orders[i]['SN'], self.orders[i]['order_type'], \
                                                                        self.orders[i]['price'], self.orders[i]['TP'], \
                                                                        self.orders[i]['SL']))
            SLTP_status = self.market.SLTP_check(this_order)
            if SLTP_status == 'TP' or SLTP_status == 'SL': #if order reached TP or SL
                self._close_order(this_order, SLTP_status)
                finished_order_index.append(i)

        if len(finished_order_index) > 0:
            for i in sorted(finished_order_index, reverse=True):
                self.orders.pop(i)

    def _close_order(self, this_order, status='X'):
        '''
        this_order: [dict]
        status: 'TP' or 'SL' or 'X'
        -
        new order details: end_price, profit, end_time
        important!!!! -> order will not be deleted from list by this function
        '''
        if status == 'TP':
            this_order['end_price'] = this_order['TP']
            if this_order['order_type'] == 'BUY':
                this_order['profit'] = (this_order['TP'] - this_order['price']) * this_order['quantity']
                logging.info('OrderTakeProfit#%04d: win %f'%(this_order['SN'], this_order['profit']))
            elif this_order['order_type'] == 'SELL':
                this_order['profit'] = (this_order['price'] - this_order['TP']) * this_order['quantity']
                logging.info('OrderTakeProfit#%04d: win %f'%(this_order['SN'], this_order['profit']))
        elif status == 'SL':
            this_order['end_price'] = this_order['SL']
            if this_order['order_type'] == 'BUY':
                this_order['profit'] = (this_order['SL'] - this_order['price']) * this_order['quantity']
                logging.info('OrderStopLoss#%04d: lose %f'%(this_order['SN'], this_order['profit']))
            elif this_order['order_type'] == 'SELL':
                this_order['profit'] = (this_order['price'] - this_order['SL']) * this_order['quantity']
                logging.info('OrderStopLoss#%04d: lose %f'%(this_order['SN'], this_order['profit']))
        elif status == 'X':
            if this_order['activated'] == True: #if not cancelled
                this_order['end_time'] = self.market.get_datetime()
                this_order['end_price'] = self.market.get_market_price()
                if this_order['order_type'] == 'BUY':
                    this_order['profit'] = (this_order['end_price'] - this_order['price']) * this_order['quantity']
                    logging.info('OrderClosed#%04d: (%f-%f)*%d=%f'%(this_order['SN'], this_order['end_price'], \
                                                                this_order['price'], this_order['quantity'], \
                                                                this_order['profit']))
                elif this_order['order_type'] == 'SELL':
                    this_order['profit'] = (this_order['price'] - this_order['end_price']) * this_order['quantity']
                    logging.info('OrderClosed#%04d: (%f-%f)*%d=%f'%(this_order['SN'], this_order['price'], \
                                                               this_order['end_price'], this_order['quantity'], \
                                                               this_order['profit']))
            else: # if cancelled
                logging.info('OrderCancelled#%04d: %s@%f'%(this_order['SN'], this_order['order_type'], this_order['price']))
            
        this_order['end_time'] = self.market.get_datetime()
        self.record.to_history(this_order)

    def close_order(self, SN):
        for i in range(len(self.orders)):
            if self.orders[i]['SN'] == SN:
                if self.orders[i]['activated'] == True: #if not cancel 
                    self.orders[i]['end_time'] = self.market.get_datetime()
                    self.orders[i]['end_price'] = self.market.get_market_price()
                    if self.orders[i]['order_type'] == 'BUY':
                        self.orders[i]['profit'] = (self.orders[i]['end_price'] - self.orders[i]['price']) * self.orders[i]['quantity']
                        logging.info('OrderClosed#%04d: %f-%f=%f'%(self.orders[i]['SN'], self.orders[i]['end_price'], \
                                                                    self.orders[i]['price'],self.orders[i]['profit']))
                    elif self.orders[i]['order_type'] == 'SELL':
                        self.orders[i]['profit'] = (self.orders[i]['price'] - self.orders[i]['end_price']) * self.orders[i]['quantity']
                        logging.info('OrderClosed#%04d: %f-%f=%f'%(self.orders[i]['SN'], self.orders[i]['price'], \
                                                                    self.orders[i]['end_price'],self.orders[i]['profit']))
                self.record.to_history(self.orders[i])
                self.orders.pop(i)
                break
    
    def get_orders_SN(self):
        return [order['SN'] for order in self.orders]

    def get_order_detail(self, SN):
        for i in range(len(self.orders)):
            if self.orders[i]['SN'] == SN:
                return self.orders[i]
                
    def get_history_order(self, amount=1):
        return self.record.get_history(amount=amount)
            
            
            
            
            
            
            
            
            
            
        