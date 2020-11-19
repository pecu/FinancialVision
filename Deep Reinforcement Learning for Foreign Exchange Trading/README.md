# Deep Reinforcement Learning for Foreign Exchange Trading

Yun-Cheng Tsai(pecutsai@gm.scu.edu.tw), Chun-Chieh Wang, Fu-Min Szu, and Kuan-Jen Wang
    
[[ Springer ]](https://link.springer.com/chapter/10.1007/978-3-030-55789-8_34?fbclid=IwAR3AwDyvvqv-SSsFbq5cLde6EPbp58tpjQGA4WqEtfzmp2MZZk-qTLq--dE)

Reinforcement learning can interact with the environment and is suitable for applications in decision control systems. Therefore, we used the reinforcement learning method to establish
a foreign exchange transaction, avoiding the long-standing problem of unstable trends in deeplearning predictions. In the system design, we optimized the Sure-Fire statistical arbitrage policy,
set three different actions, encoded the continuous price over some time into a heat-map view of the Gramian Angular Field (GAF), and compared the Deep Q Learning (DQN) and Proximal
Policy Optimization (PPO) algorithms. To test feasibility, we analyzed three currency pairs, namely EUR/USD, GBP/USD, and AUD/USD. We trained the data in units of four hours from 1 August
2018 to 30 November 2018 and tested model performance using data between 1 December 2018 and 31 December 2018. The test results of the various models indicated that favorable investment
performance achieves as long as the model can handle complex and random processes, and the state can describe the environment, validating the feasibility of reinforcement learning in the development
of trading strategies.

## Usages
#### Train models with clean & merged examples respectively (100 times will be trained in default)
    $ pip install -r requirements.txt
    $ python trainer.py "ppo" "EURUSD"
    $ python tester.py "ppo" "EURUSD"
