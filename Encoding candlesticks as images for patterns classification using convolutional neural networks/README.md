# Encoding Candlesticks as Images for Patterns Classification Using Convolutional Neural Networks

[Yun-Cheng Tsai](pecu610@gmail.com), [Jun-Hao Chen](o1r2g3a4n5i6z7e8@gmail.com), and [Chun-Chieh Wang](philip27020012@gmail.com)
    
[[ ArXiv ]](https://arxiv.org/abs/2001.02767)

Candlestick charts display the high, low, open and closing prices for a specific time series. Candlestick patterns emerge because human actions and reactions are patterned and continuously replicate and captured in the formation of the candles. According to Thomas Bulkowski's Encyclopedia of Candlestick Charts, there are 103 candlestick patterns. Traders use these patterns to determine when to enter and exit. However, the candlestick patterns classification takes the hard work out of identifying them visually. In this paper, we propose an extend Convolutional Neural Networks (CNN) approach "GASF-CNN" to recognize the candlestick patterns automatically. We use the Gramian Angular Field (GAF) to encode the time series as different types of images. Then we use the CNN with the GAF encoding images to learn eight critical kinds of candlestick patterns. The simulation and experimental results evidence that our approach can find the eight types of candlestick patterns over eighty percent accuracy automatically. 


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
  author={Tsai, Yun-Cheng and Chen, Jun-Hao and Wang, Chun-Chieh},
  journal={arXiv preprint arXiv:1901.05237},
  year={2019}
}
```
