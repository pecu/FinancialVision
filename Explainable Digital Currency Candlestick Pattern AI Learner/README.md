# Explainable Cryptocurrency Candlestick Pattern AI Learner

[Jun-Hao Chen](o1r2g3a4n5i6z7e8@gmail.com), [Cheng-Hen Wu](chwu0796@gmail.com), [Samuel Yen-Chi Chen](ycchen1989@gmail.com), and [Yun-Cheng Tsai](pecu610@gmail.com)   
    
[[ IEEE Xplore ]](https://ieeexplore.ieee.org/document/9727231?fbclid=IwAR3doeRuCKiY19_yQbSFqeiKBnurg4n1eK9tPEETEpiCXp2kolE1hYB-I7M)

More and more hedge funds have integrated AI techniques into the traditional trading strategy to speculate on cryptocurrency. Among the conventional technical analysis, candlestick pattern recognition is a critical financial trading technique by visual judgment on graphical price movement. A model with high accuracy still can not meet the demand under the highly regulated financial industry that requires understanding the decision-making and quantifying the potential risk. 
Despite the deep convolutional neural networks (CNNs) have a significant performance. Especially in a highly speculative market, blindly trusting a black-box model will incur lots of troubles. Therefore, it is necessary to incorporate explainability into a DNN-based classic trading strategy, candlestick pattern recognition. It can make an acceptable justification for traders in the cryptocurrency market. The paper exposes the black box and provides two algorithms as following. The first is an Adversarial Interpreter to explore the explainability. The second is an Adversarial Generator to enhance the model's explainability. To trust in the AI model and understand its judgment, the participant adopts powerful AI techniques to create more possibilities for AI in the cryptocurrency market.


## Results
The Figure can be seem as an important feature heatmap. The successful attack ratio ranges from 0 to 100 percent. The data point is more critical when the percentage is higher.

* The attack result of the original model:  
The overall ratios are low; only a few blocks are significantly red. Especially the ratios of knowledge-based data points that locate in the last three candlesticks are low and unevenly distributed.
<p align="left">
  <img src="https://github.com/pecu/FinancialVision/blob/master/Explainable%20Digital%20Currency%20Candlestick%20Pattern%20AI%20Learner/images/Attacking%20Result%20of%20the%20Original%20Model.png" width = "600" height = "300">
<p>

* The attack result of the model in experiment 1:  
The result is closer to the prior knowledge rule, since the ratios of knowledge-based data points that locate in the last three candlesticks are significantly enhanced. It means that the retrain model has learned the critical pattern rules and apply them to the classification.
<p align="left">
  <img src="https://github.com/pecu/FinancialVision/blob/master/Explainable%20Digital%20Currency%20Candlestick%20Pattern%20AI%20Learner/images/Attacking%20Result%20of%20the%20Experiment%201%20Model.png" width = "600" height = "300">
<p>

* The attack result of the model in experiment 2
<p align="left">
  <img src="https://github.com/pecu/FinancialVision/blob/master/Explainable%20Digital%20Currency%20Candlestick%20Pattern%20AI%20Learner/images/Attacking%20Result%20of%20the%20Experiment%202%20Model.png" width = "600" height = "300">
<p>


## Data
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
```BibTeX
@inproceedings{chen2022explainable,
  title={Explainable Digital Currency Candlestick Pattern AI Learner},
  author={Chen, Jun-Hao and Wu, Cheng-Han and Tsai, Yun-Chneg and Chen, Samuel Yen-Chi},
  booktitle={2022 14th International Conference on Knowledge and Smart Technology (KST)},
  pages={91--96},
  year={2022},
  organization={IEEE}
}
```
