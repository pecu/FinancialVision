
# BASE CLASS OBJECT
class Trader(object):
    def __init__(self, orderManager):
        '''
        orderManager: [OrderManager]
        '''
        self.orderManager = orderManager #important! call by reference

    def create_order(self, quantity=1, price=0, order_type='BUY', SL_pip=0, TP_pip=0, price_diff=0):
        SN = self.orderManager.create_order(quantity=quantity,
                                            price=price,
                                            order_type=order_type,
                                            SL_pip=SL_pip,
                                            TP_pip=TP_pip,
                                            price_diff=price_diff)
        return SN

    def close_all_orders(self):
        if len(self.orderManager.get_orders_SN()) > 0:
            self.orderManager.clear_orders()
    
    def get_orders_detail(self):
        order_details = []
        for sn in self.orderManager.get_orders_SN():
            order_detail = self.orderManager.get_order_detail(sn)
            order_details.append(order_detail)
        return order_details


class SureFireTrader(Trader):
    def __init__(self, orderManager):
        super().__init__(orderManager)
        self.quantity_level = [1,3,6,12,24,48,96]
        self.max_level = 4
        self.current_level = 0
    
    def new_trade(self, SL_pip, TP_pip, start_order_type):
        self.close_all_orders()
        self.current_level = 0
        self.create_order(quantity=self.quantity_level[0], order_type=start_order_type, \
                            SL_pip=SL_pip, TP_pip=TP_pip) #create a market order
        self.add_reverse_order(price_diff=int(SL_pip/2), SL_pip=SL_pip, TP_pip=TP_pip) #create a limit order
        

    def status_check(self):
        '''
        -
        return: [string, status]
                        'NONE'= nothing happens
                        'TRADE_OVER'=trade is over/not started yet, need to start a new trade
                        'ADD_ORDER'=the last order has been activated, add a reverse order
                        'ERROR'=unexpected error occurs
        '''
        status = 'NONE'
        detail = []
        orders_SN = self.orderManager.get_orders_SN()
        if len(orders_SN) == 0:
            status = 'TRADE_OVER'
            detail = self.orderManager.get_history_order(amount=6)
            if len(detail) > 0:
                unactivated_list = []
                for i in range(len(detail)):
                    if not detail[i]['activated']:
                        unactivated_list.append(i)
                for i in sorted(unactivated_list, reverse=True):
                    detail.pop(i)
                for i in range(len(detail)-1, -1, -1):
                    if detail[i]['quantity'] == 1:
                        detail = detail[i:]
                        break
        elif len(orders_SN) == 1: #only one unactivated order left
            this_SN = orders_SN[0]
            order_detail = self.orderManager.get_order_detail(this_SN)
            if not order_detail['activated']:
                status = 'TRADE_OVER'
                detail = self.orderManager.get_history_order(amount=6)
                
                unactivated_list = []
                for i in range(len(detail)):
                    if not detail[i]['activated']:
                        unactivated_list.append(i)
                for i in sorted(unactivated_list, reverse=True):
                    detail.pop(i)
                for i in range(len(detail)-1, -1, -1):
                    if detail[i]['quantity'] == 1:
                        detail = detail[i:]
                        break
                
            else:
                status = 'TRADE_OVER' # actually, its an ERROR...
                detail = self.orderManager.get_history_order(amount=6)
                
                unactivated_list = []
                for i in range(len(detail)):
                    if not detail[i]['activated']:
                        unactivated_list.append(i)
                for i in sorted(unactivated_list, reverse=True):
                    detail.pop(i)
                for i in range(len(detail)-1, -1, -1):
                    if detail[i]['quantity'] == 1:
                        detail = detail[i:]
                        break
        else: #check if need add a reverse order
            this_SN = orders_SN[-1]
            order_detail = self.orderManager.get_order_detail(this_SN)
            if order_detail['activated'] == True and self.current_level <= self.max_level:
                status = 'ADD_ORDER'

        return status, detail


    def add_reverse_order(self, SL_pip, TP_pip, price=0, price_diff=0):
        the_orders = self.orderManager.get_orders_SN()
        if len(the_orders) == 0:
            return #pass
        elif self.orderManager.get_order_detail(the_orders[-1])['order_type'] == 'BUY':
            next_order_type = 'SELL'
            price_diff = -price_diff
        else:
            next_order_type = 'BUY'
        
        self.current_level += 1
        self.create_order(quantity=self.quantity_level[self.current_level],
                            order_type=next_order_type,
                            price=price,
                            SL_pip=SL_pip,
                            TP_pip=TP_pip,
                            price_diff=price_diff)

    def set_max_level(self, max_level):
        self.max_level = max_level

