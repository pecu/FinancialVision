# Adversarial Robustness of Deep Convolutional Candlestick Learner

[Jun-Hao Chen](o1r2g3a4n5i6z7e8@gmail.com), [Samuel Yen-Chi Chen](ycchen1989@gmail.com), [Yun-Cheng Tsai](pecu610@gmail.com), and [Chih-Shiang Shur](aaa123848@gmail.com)    
    
[[ ArXiv ]](https://arxiv.org/abs/2006.03686)

Deep learning (DL) has been applied extensively in a wide range of fields. However, it has been shown that DL models are susceptible to a certain kinds of perturbations called **adversarial attacks**. To fully unlock the power of DL in critical fields such as financial trading, it is necessary to address such issues. In this paper, we present a method of constructing perturbed examples and use these examples to boost the robustness of the model. Our algorithm increases the stability of DL models for candlestick classification with respect to perturbations in the input
data.
    

## Implementations
<p align="center">
  <img src="https://i.imgur.com/idN1awP.jpg" alt="alt text">
<p>

## Results


## Requirements
* Numpy == 1.18.1
* TensorFlow == 1.15.0
* Keras == 2.3.1

## Usages
#### 1. Clone and install the requirements    
    $ git clone https://github.com/pecu/FinancialVision.git
    $ cd FinancialVision/Adversarial Robustness of Deep Convolutional Candlestick Learner/
    $ pip install -r requirements.txt
#### 2. Download data from [here](https://drive.google.com/drive/folders/1hbA3EaMrf9CZBgU6VqQcAseBHuEuQgi-?fbclid=IwAR1dqeY7Q4DCYsdTGBWopDb3W4o6-ixCzRKlUNslHMZjQKuYg_JOHeWxRJs).
- For **clean examples**:
    - label8_eurusd_10bar_1500_500_val200_gaf_culr.zip
- For **merged examples**:
    - merged_examples.zip
#### 3.1 Train models with clean & merged examples respectively (100 times will be trained in default)
    $ python clean_examples_training.py
    $ python merged_examples_training.py
#### 3.2 Attack the model     
    $ python attack.py

## References
1. Foolbox open-source (<https://github.com/bethgelab/foolbox>)
