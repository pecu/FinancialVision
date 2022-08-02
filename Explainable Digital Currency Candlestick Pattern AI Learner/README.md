# Explainable Cryptocurrency Candlestick Pattern AI Learner

[Jun-Hao Chen](o1r2g3a4n5i6z7e8@gmail.com), [Cheng-Hen Wu](chwu0796@gmail.com), [Samuel Yen-Chi Chen](ycchen1989@gmail.com), and [Yun-Cheng Tsai](pecu610@gmail.com)   
    
[[ IEEE Xplore ]](https://ieeexplore.ieee.org/document/9727231?fbclid=IwAR3doeRuCKiY19_yQbSFqeiKBnurg4n1eK9tPEETEpiCXp2kolE1hYB-I7M)

More and more hedge funds have integrated AI techniques into the traditional trading strategy to speculate on cryptocurrency. Among the conventional technical analysis, candlestick pattern recognition is a critical financial trading technique by visual judgment on graphical price movement. However, the deep neural network (DNN) model with high accuracy still can not meet the demand under the highly regulated financial industry that requires understanding the decision-making and quantifying the potential risk. Especially in a highly speculative market, blindly trusting a black-box model will incur lots of troubles. Therefore, the study incorporates explainability into a DNN-based classic trading strategy, candlestick pattern recognition and aims to make an acceptable justification for traders in the cryptocurrency market. The paper exposes the black box and provides two algorithms. The first one is an Adversarial Interpreter that will explore the model's explainability. The second is an Adversarial Generator that will further enhance the model's explainability. Both algorithms will help the DNN model's decision-making clearer and trustable for humans. With more trust in the AI model and understanding in its judgment, more traders will accept the AI technique and create more possibilities for AI in the cryptocurrency market.


## Results
The Figure can be seem as an important feature heatmap. The successful attack ratio ranges from 0 to 100 percent. The data point is more critical when the percentage is higher. In summary, retraining the model with the perturbated adversarial samples without human's control can significantly improve the feature importance of prior-knowledge-based data points and makes the model more trustable for the human trader.

* The attack result of the original model:  
The overall ratios are low; only a few blocks are significantly red. Especially the ratios of knowledge-based data points that locate in the last three candlesticks are low and unevenly distributed.
<p align="left">
  <img src="https://github.com/pecu/FinancialVision/blob/master/Explainable%20Digital%20Currency%20Candlestick%20Pattern%20AI%20Learner/images/Attacking%20Result%20of%20the%20Original%20Model.png" width = "600" height = "300">
<p>

* The attack result of the model in experiment 1:  
The result is much closer to the prior knowledge rule, since the ratios of knowledge-based data points that locate in the last three candlesticks are significantly enhanced. It means that the re-trained model has learned the critical pattern rules and applied them to the classification.
<p align="left">
  <img src="https://github.com/pecu/FinancialVision/blob/master/Explainable%20Digital%20Currency%20Candlestick%20Pattern%20AI%20Learner/images/Attacking%20Result%20of%20the%20Experiment%201%20Model.png" width = "600" height = "300">
<p>

* The attack result of the model in experiment 2:  
The important feature heatmap does not have a considerable improvement in general, which means that it is not easy to control the learning process of deep learning.
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
