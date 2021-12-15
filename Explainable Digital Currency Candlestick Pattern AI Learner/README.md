# Explainable Cryptocurrency Candlestick Pattern AI Learner

[Jun-Hao Chen](o1r2g3a4n5i6z7e8@gmail.com), [Cheng-Hen Wu](chwu0796@gmail.com), [Samuel Yen-Chi Chen](ycchen1989@gmail.com), and [Yun-Cheng Tsai](pecu610@gmail.com)   
    
[[Arxiv]]

More and more hedge funds have integrated AI techniques into the traditional trading strategy to speculate on cryptocurrency. Among the conventional technical analysis, candlestick pattern recognition is a critical financial trading technique by visual judgment on graphical price movement. A model with high accuracy still can not meet the demand under the highly regulated financial industry that requires understanding the decision-making and quantifying the potential risk. 
Despite the deep convolutional neural networks (CNNs) have a significant performance. Especially in a highly speculative market, blindly trusting a black-box model will incur lots of troubles. Therefore, it is necessary to incorporate explainability into a DNN-based classic trading strategy, candlestick pattern recognition. It can make an acceptable justification for traders in the cryptocurrency market. The paper exposes the black box and provides two algorithms as following. The first is an Adversarial Interpreter to explore the explainability. The second is an Adversarial Generator to enhance the model's explainability. To trust in the AI model and understand its judgment, the participant adopts powerful AI techniques to create more possibilities for AI in the cryptocurrency market.


## Implementations


## Results


### Data
  - Data can be downloaded from here ([ETH_gaf.pkl](https://drive.google.com/file/d/1ZtShhD_TkEaBQNFH-D4HQ2QH8Pllhsy9/view?usp=sharing)).

## Usages
#### 1. Create virtual environment & install dependencies
    $ git clone https://github.com/pecu/FinancialVision.git
    $ cd FinancialVision/Explainable Digital Currency Candlestick Pattern AI Learner/
    $ conda create -n py37 python=3.7
    $ activate py37
    $ pip install -r requirements.txt 
#### 2. Run multi-threads code (more illustrations are in `perturb_multi.py`)
    $ python perturb_multi.py


## References

To cite this study:

