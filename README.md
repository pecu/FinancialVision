# Explainable Deep Convolutional Candlestick Learner
##### We provide a framework explaining the way that learned model determine the specific candlestick patterns,
##### which is based on local search adversarial attacks.
Jun-Hao Chen, Samuel Yen-Chi Chen, Yun-Cheng Tsai, and Chih-Shiang Shur    
    
[[ArXiv]](https://arxiv.org/pdf/2001.02767)

## Implementations


## Results
| Label  | Success Rate | Percent (%) |
| ------------- | ------------- | ------------- |
| 1  | 631 / 1500  | 42.1 |
| 2  | 972 / 1500  | 64.8 |
| 3  | 1079 / 1500  | 71.9 |
| 4  | 1319 / 1500  | 87.9 |
| 5  | 602 / 1500  | 40.1 |
| 6  | 932 / 1500  | 62.1 |
| 7  | 953 / 1500  | 63.5 |
| 8  | 1238 / 1500  | 82.5 |

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
