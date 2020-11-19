#test
from lib.Market import Market
from lib.Record import Record
from lib.OrderManager import OrderManager
from lib.Trader import SureFireTrader
from lib.Series2GAF import gaf_encode
from tensorforce.agents import Agent
from tensorforce.agents.constant_agent import ConstantAgent
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
import json
import os, sys
import pickle

plt.style.use('ggplot')
#logging.basicConfig(level=logging.INFO)

def main():
    
    AGENT_METHOD = sys.argv[1] # "ppo"
    CURRENCY_PAIR = sys.argv[2] # "EURUSD"

    theMarket = Market(data_path="data/%s_Candlestick_4_Hour_BID_01.12.2018-31.12.2018.csv"%CURRENCY_PAIR)#, indicators={'ADX': 12})
    MyRecord = Record()
    MyOrderManager = OrderManager(market=theMarket, record=MyRecord)
    MyTrader = SureFireTrader(orderManager=MyOrderManager)

    SLTP_pips = [20,25,30]
    start_order_type = ['BUY','SELL']
    max_level_limit = [2,3,4]
    window_size = 12
    
    output_record = {'episode':[], 'total_trades': [], 'win_trades': [],
                    'lose_trades': [], 'profit_factor': [], 'net_profit': [],
                    'max_drawdown': [], 'trading_rounds': []}
                    
    if AGENT_METHOD != "constant":                
        iter_range = range(0,100+1,100)
    else:
        iter_range = range(1)
        
    for the_episode in iter_range:
        # Create a RL agent
        if AGENT_METHOD != "constant":
            with open("config/%s.json"%AGENT_METHOD, 'r') as fp:
                agent_config = json.load(fp=fp)
            with open("config/conv2d.json", 'r') as fp:
                network_config = json.load(fp=fp)
            agent = Agent.from_spec(
                spec=agent_config,
                kwargs=dict(
                    states=dict(type='float', shape=(window_size,window_size,4)),               #[Open, High, Low, Close]
                    actions=dict(
                        SLTP_pips=dict(type='int', num_actions=len(SLTP_pips)),                 #[20,25,30]
                        start_order_type=dict(type='int', num_actions=len(start_order_type)),   #['BUY','SELL']
                        max_level_limit=dict(type='int', num_actions=len(max_level_limit))      #[2,3,4,5]
                    ),
                    network=network_config
                )

            )
            agent.restore_model("save_model"+"/"+AGENT_METHOD+"/"+'%04d'%the_episode)
        else:
            agent = ConstantAgent(
                states=dict(type='float', shape=(window_size,window_size,4)),               #[Open, High, Low, Close]
                actions=dict(
                    SLTP_pips=dict(type='int', num_actions=len(SLTP_pips)),                 #[20,25,30]
                    start_order_type=dict(type='int', num_actions=len(start_order_type)),   #['BUY','SELL']
                    max_level_limit=dict(type='int', num_actions=len(max_level_limit))      #[2,3,4]
                ),
                action_values={'SLTP_pips': 2, 'max_level_limit': 2, 'start_order_type': 0}
            )

        profit_history = []
        idle_count = 0
        round_count = 0
        episode_end = False
        max_idle_limit = 12 #future action
        MyRecord.reset()
        MyOrderManager.reset()
        theMarket.reset(start_index=window_size)
        
        #pbar = tqdm()
        while(theMarket.next()): #main loop, essential

            #pbar.update(1) # simple-GUI
            
            ################### ROUTINES ################### 
            MyOrderManager.orders_check() #routine, after market.next
            trade_status, other_detail = MyTrader.status_check() #routine, after orders_check
            ################################################


            ################### GET STATE ##################
            ohlc = theMarket.get_ohlc(size=window_size)
            indicators = theMarket.get_indicators(size=window_size)
            O, H, L, C = gaf_encode(ohlc['Open']), gaf_encode(ohlc['High']), \
                            gaf_encode(ohlc['Low']),gaf_encode(ohlc['Close'])
            #ADX = gaf_encode(indicators['ADX'])
            state = np.stack((O,H,L,C),axis=-1)
            ################################################


            ################## TAKE ACTION #################
            if trade_status == 'TRADE_OVER':
                
                ############ GET REWARD & TRAIN ################

                action = agent.act(state) # Get prediction from agent, execute
                SL_pip = SLTP_pips[action['SLTP_pips']]*2
                TP_pip = SLTP_pips[action['SLTP_pips']]
                MyTrader.set_max_level(max_level_limit[action['max_level_limit']])
                first_order_type = start_order_type[action['start_order_type']]
                ################################################

                MyTrader.new_trade(SL_pip=SL_pip, TP_pip=TP_pip, start_order_type=first_order_type)

                round_count += 1
                idle_count = 0
                logging.info("NewTradeStarted: current net profit=%f (price@%f)"%(MyRecord.get_net_profit(), theMarket.get_market_price()))
            
            elif trade_status == 'ADD_ORDER':
                last_order = MyTrader.get_orders_detail()[-1]
                if last_order['order_type'] == 'BUY':
                    price = last_order['price'] - theMarket.get_pip(TP_pip)
                elif last_order['order_type'] == 'SELL':
                    price = last_order['price'] + theMarket.get_pip(TP_pip)
                MyTrader.add_reverse_order(price=price, SL_pip=SL_pip, TP_pip=TP_pip)
                idle_count = 0
            
            elif trade_status == 'ERROR':
                logging.warning("SureFireError: order issues...")
            
            elif trade_status == 'NONE':
                idle_count += 1
                if idle_count >= max_idle_limit:
                    
                    action = agent.act(state) # Get prediction from agent, execute
                    SL_pip = SLTP_pips[action['SLTP_pips']]*2
                    TP_pip = SLTP_pips[action['SLTP_pips']]
                    MyTrader.set_max_level(max_level_limit[action['max_level_limit']])
                    first_order_type = start_order_type[action['start_order_type']]
                    

                    MyTrader.new_trade(SL_pip=SL_pip, TP_pip=TP_pip, start_order_type=first_order_type)
                    idle_count = 0
                    logging.info("NewTradeStarted: current net profit=%f (price@%f)"%(MyRecord.get_net_profit(), theMarket.get_market_price()))
            ################################################

        
            profit_history.append(MyRecord.get_net_profit()) #for plotting
        
        #pbar.close()
        my_details = MyRecord.show_details()
        print("Rounds of Tradings: %d\n"%round_count)
        print('---')
        
        output_record['episode'].append(the_episode)
        output_record['trading_rounds'].append(round_count)
        for k in my_details:
            output_record[k].append(my_details[k])
        
        plt.plot(range(len(profit_history)), profit_history)
        plt.show()
        
        
    for k in output_record:
        plt.plot(output_record['episode'], output_record[k])
        plt.title(k)
        plt.show()
    
    pd.DataFrame(output_record).to_csv('test_detail-episode.csv', index=False)
    
    
if __name__=="__main__":
    main()
