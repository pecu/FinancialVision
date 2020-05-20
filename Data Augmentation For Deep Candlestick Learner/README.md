# Data Augmentation For Deep Candlestick Learner

[Chia-Ying Tsao](), [Jun-Hao Chen](o1r2g3a4n5i6z7e8@gmail.com), [Samuel Yen-Chi Chen](ycchen1989@gmail.com), and [Yun-Cheng Tsai](pecu610@gmail.com)

[[ ArXiv ]](https://arxiv.org/abs/2005.06731)

Deep learning models require large data set to perform well,  
yet sometimes it is difficult to get sufficient data in some field such as financial trading.  

We propose a Modified Local Search AttackSampling method to augment the candlestick data,  
which is high-quality data and hard to distinguish by human.  

It will open a new way for finance community to employ existing machine learning techniques even if the dataset is small.

## Implementations

## Results
* Dependent Paired T Test  
       ---                 |  N  |   Mean   |  Std     |  T-value |  P-value  
  ----                     | ---   | ----       |  ---       | ---------  |  -------  
  H0  | 245 |  -0.0575 |  0.2386  | -3.7736 |  0.0002

![Alt text](./images/result/adv_examples.PNG)

## Requirements
* Numpy == 1.17.0  
* Tensorflow-gpu == 1.14.0  
* Keras == 2.2.5  
## Usages
**1. Clone and install the requirements**  
```
$ git clone https://github.com/FinancialVision.git
$ cd FinancialVision/
$ pip install -r requirements.txt
```
**2. Download data from here**  
## References
