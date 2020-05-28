# Adversarial Robustness of Deep Convolutional Candlestick Learner

[Jun-Hao Chen](o1r2g3a4n5i6z7e8@gmail.com), [Samuel Yen-Chi Chen](ycchen1989@gmail.com), [Yun-Cheng Tsai](pecu610@gmail.com), and [Chih-Shiang Shur](aaa123848@gmail.com)    
    
[[ ArXiv ]](https://arxiv.org/abs/2001.02767)


We provide a framework explaining the way that learned model determine the specific candlestick patterns,    
which is based on local search adversarial attacks.
    
    

## Implementations


## Results


## Requirements
* Numpy == 1.18.1
* TensorFlow == 1.15.0
* Keras == 2.3.1

## Usages
#### 1. Clone and install the requirements    
    $ git clone https://github.com/pecu/FinancialVision.git
    $ cd FinancialVision/
    $ pip install -r requirements.txt
#### 2. Download data from [here](https://drive.google.com/drive/folders/1hbA3EaMrf9CZBgU6VqQcAseBHuEuQgi-?fbclid=IwAR1dqeY7Q4DCYsdTGBWopDb3W4o6-ixCzRKlUNslHMZjQKuYg_JOHeWxRJs).
- EURUSD_10bar_train1500_test500_val200.zip
#### 3. Attack the model     
    $ python main.py


## References
1. Foolbox open-source (<https://github.com/bethgelab/foolbox>)
