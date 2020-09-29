#! /bin/bash

export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64
python lstm_train.py -r True -v True > Results/Baseline/lstm_vggish_baseline_rs.txt

export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64
python lstm_train.py -r True -a True -v True > Results/AugmentedDataset/lstm_vggish_aug_rs.txt

export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64
python lstm_train.py -r True -s True -v True > Results/SpecAugment/lstm_vggish_sa_rs.txt

export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64
python lstm_train.py -r True -s True -a True -v True > Results/BothAugmentations/lstm_vggish_aug_sa_rs.txt