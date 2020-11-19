from lib.Market import Market
from lib.Record import Record
from lib.OrderManager import OrderManager
from lib.Trader import SureFireTrader
from lib.Series2GAF import gaf_encode
from tensorforce.agents import Agent
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import numpy as np
import logging
import json
import os
import pickle
import sys
plt.style.use('ggplot')
#logging.basicConfig(level=logging.INFO)

#SETTINGS
AGENT_METHOD = sys.argv[1] #'ppo' 
CURRENCY_PAIR =  sys.argv[2]  #'EURUSD'


def main():
    theMarket = Market(data_path="data/%s_Candlestick_4_Hour_BID_01.08.2018-30.11.2018.csv"%CURRENCY_PAIR)#, indicators={'ADX': 12})
    MyRecord = Record()
    MyOrderManager = OrderManager(market=theMarket, record=MyRecord)
    MyTrader = SureFireTrader(orderManager=MyOrderManager)

    SLTP_pips = [20,25,30]
    start_order_type = ['BUY','SELL']
    max_level_limit = [2,3,4]
    window_size = 12

    # Create a RL agent
    if AGENT_METHOD is not 'constant':
        with open("config/%s.json"%AGENT_METHOD, 'r') as fp:
            agent_config = json.load(fp=fp)
    with open("config/conv2d.json", 'r') as fp:
        network_config = json.load(fp=fp)
        
    if AGENT_METHOD is not 'constant':
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
    if AGENT_METHOD is 'constant':
        agent = ConstantAgent(
                    states=dict(type='float', shape=(window_size,window_size,4)),               #[Open, High, Low, Close]
                    actions=dict(
                        SLTP_pips=dict(type='int', num_actions=len(SLTP_pips)),                 #[20,25,30]
                        start_order_type=dict(type='int', num_actions=len(start_order_type)),   #['BUY','SELL']
                        max_level_limit=dict(type='int', num_actions=len(max_level_limit))      #[2,3,4]
                    ),
                    action_values={'SLTP_pips': 2, 'max_level_limit': 2, 'start_order_type': 0}
                )
    if not os.path.exists("save_model/%s/trades"%AGENT_METHOD):
        os.makedirs("save_model/%s/trades"%AGENT_METHOD)

    reward_history = []
    for episode in trange(200+1, ascii=True):

        profit_history = []
        this_reward_history = []
        idle_count = 0
        round_count = 0
        episode_end = False
        max_idle_limit = 12 #future action
        MyRecord.reset()
        MyOrderManager.reset()
        theMarket.reset(start_index=window_size)
        
        pbar = tqdm()
        while(theMarket.next()): #main loop, essential

            pbar.update(1) # simple-GUI
            
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
                if theMarket.get_current_index() > window_size:
                    '''
                    profit = sum(round(order['profit'],5) for order in other_detail if order['profit']>0)
                    loss = sum(round(order['profit'],5) for order in other_detail if order['profit']<0)
                    
                    this_profit_factor = MyRecord.get_profit_factor()
                    this_trade_length = len(MyRecord.get_history())
                    reward = this_profit_factor*np.sqrt(this_trade_length)#SQN
                    '''
                    raw_reward = (MyRecord.get_net_profit()-profit_history[-1])/theMarket.get_pip()
                    penalty = 1.0-0.1*len(other_detail)
                    if raw_reward > 0:
                        reward = raw_reward*penalty
                    else:
                        if len(other_detail) == 0:
                            reward = 0
                        else:
                            reward = -np.abs(other_detail[0]['TP']-other_detail[0]['price'])/theMarket.get_pip()
                    
                    if theMarket.get_current_index() >= theMarket.get_data_length() - max_idle_limit*max_level_limit[-1]:
                        episode_end = True
                    agent.observe(reward=reward, terminal=episode_end) # Add experience, agent automatically updates model according to batch size
                    this_reward_history.append(reward)
                    if episode_end == True:
                        if episode%100 == 0:
                            this_dir = 'save_model/%s/%04d'%(AGENT_METHOD, episode)
                            if not os.path.exists(this_dir):
                                os.makedirs(this_dir)
                            agent.save_model(this_dir+'/model')
                        pbar.close()
                        reward_history.append(this_reward_history)
                        with open('save_model/%s/trades/episode_%04d.pkl'%(AGENT_METHOD, episode), 'wb') as f:
                            pickle.dump(MyRecord.get_history(),f,protocol=-1)
                        break
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

                    ############ GET REWARD & TRAIN ################
                    '''
                    profit = sum(round(order['profit'],5) for order in other_detail if order['profit']>0)
                    loss = sum(round(order['profit'],5) for order in other_detail if order['profit']<0)
                    
                    this_profit_factor = MyRecord.get_profit_factor()
                    this_trade_length = len(MyRecord.get_history())
                    reward = this_profit_factor*np.sqrt(this_trade_length)#SQN
                    '''
                    raw_reward = (MyRecord.get_net_profit()-profit_history[-1])/theMarket.get_pip()
                    penalty = 1.0-0.1*len(other_detail)
                    if raw_reward > 0:
                        reward = raw_reward*penalty
                    else:
                        if len(other_detail) == 0:
                            reward = 0
                        else:
                            reward = -np.abs(other_detail[0]['TP']-other_detail[0]['price'])/theMarket.get_pip()
                    
                    if theMarket.get_current_index() >= theMarket.get_data_length() - max_idle_limit*max_level_limit[-1]:
                        episode_end = True
                    agent.observe(reward=reward, terminal=episode_end) # Add experience, agent automatically updates model according to batch size
                    this_reward_history.append(reward)
                    if episode_end == True:
                        if episode%100 == 0:
                            this_dir = 'save_model/%s/%04d'%(AGENT_METHOD, episode)
                            if not os.path.exists(this_dir):
                                os.makedirs(this_dir)
                            agent.save_model(this_dir+'/model')
                        pbar.close()
                        reward_history.append(this_reward_history)
                        with open('save_model/%s/trades/episode_%04d.pkl'%(AGENT_METHOD, episode), 'wb') as f:
                            pickle.dump(MyRecord.get_history(),f,protocol=-1)
                        break
                    
                    action = agent.act(state) # Get prediction from agent, execute
                    SL_pip = SLTP_pips[action['SLTP_pips']]*2
                    TP_pip = SLTP_pips[action['SLTP_pips']]
                    MyTrader.set_max_level(max_level_limit[action['max_level_limit']])
                    first_order_type = start_order_type[action['start_order_type']]
                    ################################################

                    MyTrader.new_trade(SL_pip=SL_pip, TP_pip=TP_pip, start_order_type=first_order_type)
                    idle_count = 0
                    logging.info("NewTradeStarted: current net profit=%f (price@%f)"%(MyRecord.get_net_profit(), theMarket.get_market_price()))
            ################################################

        
            profit_history.append(MyRecord.get_net_profit()) #for plotting

        #MyRecord.show_details()
        #print("Rounds of Tradings: %d\n"%round_count)
        
    with open('save_model/%s/trades/profit_history.pkl'%AGENT_METHOD, 'wb') as f:
        pickle.dump(profit_history,f,protocol=-1)
        
    with open('save_model/%s/trades/reward_history.pkl'%AGENT_METHOD, 'wb') as f:
        pickle.dump(reward_history,f,protocol=-1)

#     plt.plot(range(len(profit_history)), profit_history)
#     plt.show()

#     plt.plot(range(len(reward_history)),reward_history)
#     plt.show()


if __name__=="__main__":
    main()
