# Transformer-Classify
Reimplement of transformer on session classification

**Reference:**

Sheil, H., Rana, O., & Reilly, R. (2018). Predicting purchasing intent: Automatic Feature Learning using Recurrent Neural Networks. https://arxiv.org/abs/1807.08207v1

## Environment
Check **environment.txt** to see the environment and packages required

## Download Dataset
The dataset we use is the 'yoo-choose' dataset in recsys2015

You can download it directly from the [official websiteof recsys2015](https://2015.recsyschallenge.com/challenge.html) , which comprises three .dat files: yoochoose-buys.dat, yoochoose-clicks.dat and yoochoose-test.dat

## Data Preprocess
The raw data need to be converted to sequences according to there session before we use it.

You can use **preprocess.py** to do this job, which takes the 1st argument as the input directory and the  2nd argument as the output directory.

`python preprocess.py /transformer/input_data /transformer/output_data`

Then you will get the preprocessed data in the target output directory.

## Train Model
To train a Transformer model, run the following code

 `python main.py train Transformer /transformer/train_data`

else to train a LSTM model, run

 `python main.py train LSTM /transformer/train_data`
 
 The training parameters are specified in **config.py**.
 
 The model checkpoints and tensorboard visualization result will be stored respectively in 'checkpoints' subdirectory and 'logs' subdirectory in the directory where you run this command. And 3 tokenizers will be automatically saved in the train_data directory.
 
 ## Test Model
 To test a model, you only need to substitute the 2nd parameter in the code above with a 'test'


Before testing, you **should** copy the pickle files of 3 tokenizers in the train_data directory and paste them in the test_data directory so that test_data and be tokeized in the same way.
