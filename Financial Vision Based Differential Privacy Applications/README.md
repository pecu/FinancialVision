# Financial Vision Based Differential Privacy Applications

[Jun-Hao Chen](o1r2g3a4n5i6z7e8@gmail.com), [Yi-Jen Wang](yiren513@gmail.com), [Yun-Cheng Tsai](pecu610@gmail.com), and [Samuel Yen-Chi Chen](ycchen1989@gmail.com)
    
[[ ArXiv ]](https://arxiv.org/abs/2112.14075?fbclid=IwAR0sNsMn8umjyEkm2GVfK79ww-klERPr_UOM6wac0FRWlk0BMdXBx0pCnqM)

The importance of deep learning data privacy has gained significant attention in recent years. It is probably to suffer data breaches when applying deep learning to cryptocurrency that lacks supervision of financial regulatory agencies. However, there is little relative research in the financial area to our best knowledge. We apply two representative deep learning privacy-privacy frameworks proposed by Google to financial trading data. We designed the experiments with several different parameters suggested from the original studies. In addition, we refer the degree of privacy to Google and Apple companies to estimate the results more reasonably. The results show that DP-SGD performs better than the PATE framework in financial trading data. The tradeoff between privacy and accuracy is low in DP-SGD. The degree of privacy also is in line with the actual case. Therefore, we can obtain a strong privacy guarantee with precision to avoid potential financial loss.

## Step 1. Create python 3.7 environment
conda create -m -n envname python=3.7
## Step 2. Install packages
pip install -r requirements.txt
## Step 3. Run PATE
python PATE_final.py
## Step 4. Run DP-SGD
python DPSGD_final.py

## Attack DP-SGD models 
We use the [ml\_privacy\_meter](https://github.com/privacytrustlab/ml_privacy_meter) package to attack the baseline model and the DP-SGD models to understand how much data privacy these models can provide. A tutorial can be found [Here](https://github.com/privacytrustlab/ml_privacy_meter/tree/master/archive). The below sample plot will in the logs/plots folder. The membership probability predictions for training set and population set by the model are also saved as numpy files in the logs folder as member_probs.npy and nonmember_probs.npy.

![figure](attack_result.png)

We defined the probability of training data being defined correctly by the attack model as recognition rate. The recognition rate can be visualizing by calling
``` 
python recognition_rate.py
```

![figure](recognition_rate.png)

## References

To cite this study:
```BibTeX
@inproceedings{Chen2021FinancialVB,
  title={Financial Vision Based Differential Privacy Applications},
  author={Jun-Hao Chen and Yi-Jen Wang and Yun-Cheng Tsai and Samuel Yen-Chi Chen},
  year={2021}
}
```
