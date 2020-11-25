# Transformer-classify
Reimplement of transformer on session classification

**Reference:**

Sheil et al. (2018) "Predictingpurchasingintent:AutomaticFeatureLearningusing RecurrentNeuralNetworks "

## Download Dataset
The dataset we use is the 'yoo-choose' dataset in recsys2015

You can download it directly from the [official websiteof recsys2015](https://2015.recsyschallenge.com/challenge.html) , which comprises three .dat files: yoochoose-buys.dat, yoochoose-clicks.dat and yoochoose-test.dat

## Data Preprocess
The raw data need to be converted to sequences according to there session before we use it.

You can use **preprocess.py** to do this job, which takes the 1st argument as the input directory and the  2nd argument as the output directory.

`python preprocess.py /transformer/input_data /transformer/output_data`

Then you will get the preprocessed data in the target output directory.

## Train Model
To train the model, run the following code

 `python train.py /transformer/output_data`
 
 The training parameters are specified in **config.py**.
 
 The model checkpoints and tensorboeard visualization result will be stored respectively in 'checkpoints' subdirectory and 'logs' subdirectory in the directory where you run this command.
