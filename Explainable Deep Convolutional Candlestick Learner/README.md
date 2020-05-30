# Explainable Deep Convolutional Candlestick Learner

[Jun-Hao Chen](o1r2g3a4n5i6z7e8@gmail.com), [Samuel Yen-Chi Chen](ycchen1989@gmail.com), [Yun-Cheng Tsai](pecu610@gmail.com), and [Chih-Shiang Shur](aaa123848@gmail.com)    
    
[[ ArXiv ]](https://arxiv.org/abs/2001.02767)

Candlesticks are graphical representations of price movements for a given period. The traders can discover the trend of the asset by looking at the candlestick patterns. Although deep convolutional neural networks have achieved great success for recognizing the candlestick patterns, their reasoning hides inside a black box. The traders cannot make sure what the model has learned. In this contribution, we provide a framework which is to explain the reasoning of the learned model determining the specific candlestick patterns of time series. Based on the local search adversarial attacks, we show that the learned model perceives the pattern of the candlesticks in a way similar to the human trader.  


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
    $ cd FinancialVision/Explainable Deep Convolutional Candlestick Learner/
    $ pip install -r requirements.txt
#### 2. Download data from [here](https://drive.google.com/drive/folders/1hbA3EaMrf9CZBgU6VqQcAseBHuEuQgi-?fbclid=IwAR1dqeY7Q4DCYsdTGBWopDb3W4o6-ixCzRKlUNslHMZjQKuYg_JOHeWxRJs).
- label8_eurusd_10bar_1500_500_val200_gaf_culr.zip
#### 3. Attack the model     
    $ python main.py


## References
1. Foolbox open-source (<https://github.com/bethgelab/foolbox>)

To cite this study:
```BibTeX
@article{chen2020explainable,
  title={Explainable Deep Convolutional Candlestick Learner},
  author={Chen, Jun-Hao and Chen, Samuel Yen-Chi and Tsai, Yun-Cheng and Shur, Chih-Shiang},
  journal={arXiv preprint arXiv:2001.02767},
  year={2020}
}
```
