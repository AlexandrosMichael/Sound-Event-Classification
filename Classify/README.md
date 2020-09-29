# Classify

The Python modules in this directory are used to carry out the model training and evaluation procedure.

The scripts in this directory provide a  comprehensive testing suite during which models are trained, evaluated and their results are stored in the Results directory.

## VGGish

The models make use of the pre-trained VGGish which is too large to be included in the submission. Download the files in the following link, and place them in the Classify directory. [Link to VGGish](https://universityofstandrews907-my.sharepoint.com/:f:/g/personal/am425_st-andrews_ac_uk/En7-TAeAYDNGpmTQzIhJAYoBOvL08IEukQOsIrERLxFUTA?e=BRkZHL)

## VGGish+ Usage

To train and evaluate a VGGish+ model use the following:

 ```bash
python classify_retrain.py [-p True] [-t <int>] [-a True] [-s True] [-r True]
```
Where the optional flags are as follows:
    
-p : Generate plots.

-t : Set the number of trainable layers. Count starting from the final layer of the network. If absent, VGGish+ will not be fine-tuned.

-a : Train using the dataset  augmented using pitch and volume shifting. If not present, the model will be trained using the  non-augmented dataset.

-s : Train using the spectrograms with SpecAugment applied on them. If  not present, SpecAugment will not be  used.

-r : Evaluate the model using the RealSound dataset. If false, it will evaluate on a subset of the FreeSound dataset.

Run the test-suite provided using the following: 

 ```bash
./run_suite.sh
```
 
## LSTM Usage

Similarly, to train and evaluate an LSTM model use the following:

 ```bash
python lstm_train.py [-p True] [-a True] [-s True] [-r True]
```

Run the test-suite provided  using the following:

 ```bash
./run_lstm_suite.sh
```

## Saving Models

A directory Models is included which may be used to hold trained models. The option to save models is not  automated and it is hard-coded in the classify_retrain.py and the lstm_train.py modules. 
