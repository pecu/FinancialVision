# Encoding Candlesticks as Images for Patterns Classification Using Convolutional Neural Networks

[Jun-Hao Chen](o1r2g3a4n5i6z7e8@gmail.com) and [Yun-Cheng Tsai](pecu610@gmail.com)
    
[[Arxiv]](https://arxiv.org/abs/1901.05237)

Candlestick charts display the high, low, opening, and closing prices in a specific period. Candlestick patterns emerge because human actions and reactions are patterned and continuously replicate. These patterns capture information on the candles. According to Thomas Bulkowski's Encyclopedia of Candlestick Charts, there are 103 candlestick patterns. Traders use these patterns to determine when to enter and exit. Candlestick pattern classification approaches take the hard work out of visually identifying these patterns. To highlight its capabilities, we propose a two-steps approach to recognize candlestick patterns automatically. The first step uses the Gramian Angular Field (GAF) to encode the time series as different types of images. The second step uses the Convolutional Neural Network (CNN) with the GAF images to learn eight critical kinds of candlestick patterns. In this paper, we call the approach GAF-CNN. In the experiments, our method can identify the eight types of candlestick patterns with 90.7% average accuracy automatically in real-world data, outperforming the LSTM model.


## Implementations


## Results


### Data
  - The example of rule-based code shows in `rulebase.py`, but you still need to finish the encoding and sampling or use the data we already processed.
  - [label8_eurusd_10bar_1500_500_val200_gaf_culr.pkl](https://drive.google.com/open?id=1cCym8Re1aPDep29_cj9kUavCrYzpGV-U)

## Usages
#### 1. Environments    
    $ git clone https://github.com/pecu/FinancialVision.git
    $ cd FinancialVision/Encoding Candlesticks as Images for Patterns Classification Using Convolutional Neural Networks/
    $ pip install -r requirements.txt 
#### 2. Train the model
    $ python main.py


## References

To cite this study:
```BibTeX
@article{tsai2019encoding,
  title={Encoding candlesticks as images for patterns classification using convolutional neural networks},
  author={Chen, Jun-Hao and Tsai, Yun-Cheng},
  journal={arXiv preprint arXiv:1901.05237},
  year={2019}
}
```
