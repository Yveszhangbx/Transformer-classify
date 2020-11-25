# Transformer-classify
Reimplement of transformer on session classification

**Reference:**

Sheil et al. (2018) "Predictingpurchasingintent:AutomaticFeatureLearningusing RecurrentNeuralNetworks "

## Prepare Dataset
The dataset we use is the 'yoo-choose' dataset in recsys2015

You can download it directly from the [official websiteof recsys2015](https://2015.recsyschallenge.com/challenge.html) , which comprises three .dat files: yoochoose-buys.dat, yoochoose-clicks.dat and yoochoose-test.dat

## Data Preprocess
The raw data need to be converted to sequences according to there session before we use it.

You can use **preprocess.py** to do this job. 
