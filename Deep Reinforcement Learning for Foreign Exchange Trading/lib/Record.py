

class Record(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.trades = {'win':0, 'lose':0}
        self.quantity = {'buy':0, 'sell':0}
        self.points = {'buy':0, 'sell':0, 'loss': 0, 'profit': 0}
        self.history_order = []

    def to_history(self, this_order):
        if this_order['activated'] == True: #is not canceled order
            if this_order['profit'] <= 0: #lose trade
                self.trades['lose'] += 1
                self.points['loss'] += this_order['profit']
            elif this_order['profit'] > 0: #win trade
                self.trades['win'] += 1
                self.points['profit'] += this_order['profit']
            if this_order['order_type'] == 'BUY':
                self.quantity['sell'] += this_order['quantity']
                self.points['sell'] += this_order['quantity'] * this_order['end_price']
            elif this_order['order_type'] == 'SELL':
                self.quantity['buy'] += this_order['quantity']
                self.points['buy'] += this_order['quantity'] * this_order['end_price']
        self.history_order.append(this_order)

    def order_activated(self, this_order):
        self.quantity[this_order['order_type'].lower()] += this_order['quantity']
        self.points[this_order['order_type'].lower()] += this_order['quantity'] * this_order['price']

    def show_details(self):
        print('Total Trades: %d'%(self.trades['win']+self.trades['lose']))
        print('Win Trades : Lose Trades = %d : %d'%(self.trades['win'], self.trades['lose']))
        pf = '--- (loss=0)' if self.points['loss'] == 0 else "%.2f"%(self.points['profit']/-self.points['loss'])
        print('Profit Factor: %s'%pf)
        print('Net Profit: %f'%(self.points['profit'] + self.points['loss']))
        
        hist_profit = 0.0
        peak = 0.0
        max_drawdown = 0.0
        for hist in self.history_order:
            if hist['activated']:
                this_return = round(hist['profit'], 5)
                hist_profit += this_return
                if this_return < 0:
                    drawdown = peak-hist_profit
                    max_drawdown = max(drawdown, max_drawdown)
                else:
                    peak = hist_profit if hist_profit > peak else peak
        
        print('Max Drawdown: %f%%'%((max_drawdown/hist_profit)*100))
        return {'total_trades': self.trades['win']+self.trades['lose'],
                'win_trades': self.trades['win'], 'lose_trades': self.trades['lose'],
                'profit_factor': pf, 'net_profit': self.points['profit'] + self.points['loss'],
                'max_drawdown': max_drawdown}

    def get_net_profit(self):
        return self.points['profit'] + self.points['loss']

    def get_profit_factor(self, look_back=0):
        pf = 2.0
        if look_back == 0:
            pf = pf if self.points['loss'] == 0 else self.points['profit']/-self.points['loss']
        else:
            profit = 0.0
            loss = 0.0
            for i in range(len(self.history_order)-1, -1, -1):
                if self.history_order[i]['activated']:
                    value = self.history_order[i]['profit']
                    if value < 0:
                        loss += value
                    else:
                        profit += value
                look_back -= 1
                if look_back <= 0:
                    break
            pf = pf if loss == 0 else profit/-loss
                
        return pf
        
    def get_history(self, amount=-1):
        if amount <= 0 or amount >= len(self.history_order):
            return self.history_order.copy()
        else:
            return self.history_order[len(self.history_order)-amount:].copy()
    
        