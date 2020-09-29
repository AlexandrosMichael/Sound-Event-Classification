#! /bin/bash

# ---------------------------------------------------------------------------------------------------------------------
# MODELS TRAINED AND TESTED ON FREESOUNDS

# BASELINE MODELS NO AUGMENTATION

# No fine-tuning
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64
python classify_retrain.py -p True > Results/Baseline/no_retrain.txt

# Fine-tune fc layers
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64
python classify_retrain.py -t 4 -p True > Results/Baseline/fc_retrain.txt

# Fine-tune CNN & fc layers
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64
python classify_retrain.py -t 7 -p True > Results/Baseline/cnn_retrain.txt


# MODELS ON AUGMENTED DATASET

# No fine-tuning
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64
python classify_retrain.py -a True -p True > Results/AugmentedDataset/aug_no_retrain.txt

# Fine-tune fc layers
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64
python classify_retrain.py -a True -t 4 -p True > Results/AugmentedDataset/aug_fc_retrain.txt

# Fine-tune CNN & fc layers
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64
python classify_retrain.py -a True -t 7 -p True > Results/AugmentedDataset/aug_cnn_retrain.txt


# MODELS WITH SPEC AUGMENT

# No fine-tuning
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64
python classify_retrain.py -s True -p True > Results/SpecAugment/sa_no_retrain.txt

# Fine-tune fc layers
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64
python classify_retrain.py -s True -t 4 -p True > Results/SpecAugment/sa_fc_retrain.txt

# Fine-tune CNN & fc layers
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64
python classify_retrain.py -s True -t 7 -p True > Results/SpecAugment/sa_cnn_retrain.txt


# MODELS ON BOTH AUGMENTATIONS

# No fine-tuning
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64
python classify_retrain.py -s True -a True -p True > Results/BothAugmentations/aug_sa_no_retrain.txt

# Fine-tune fc layers
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64
python classify_retrain.py -s True -a True -t 4 -p True > Results/BothAugmentations/aug_sa_fc_retrain.txt

#Fine-tune CNN & fc layers
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64
python classify_retrain.py -s True -a True -t 7 -p True > Results/BothAugmentations/aug_sa_cnn_retrain.txt



# ---------------------------------------------------------------------------------------------------------------------

# MODELS TRAINED ON FREESOUNDS AND TESTED ON REALSOUNDS
# BASELINE MODELS NO AUGMENTATION

# No fine-tuning
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64
python classify_retrain.py -r True -p True > Results/Baseline/no_retrain_rs.txt

# Fine-tune fc layers
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64
python classify_retrain.py -t 4 -r True -p True > Results/Baseline/fc_retrain_rs.txt

# Fine-tune CNN & fc layers
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64
python classify_retrain.py -t 7 -r True -p True > Results/Baseline/cnn_retrain_rs.txt


# MODELS ON AUGMENTED DATASET

# No fine-tuning
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64
python classify_retrain.py -a True -r True -p True > Results/AugmentedDataset/aug_no_retrain_rs.txt

# Fine-tune fc layers
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64
python classify_retrain.py -a True -t 4 -r True -p True > Results/AugmentedDataset/aug_fc_retrain_rs.txt

# Fine-tune CNN & fc layers
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64
python classify_retrain.py -a True -t 7 -r True -p True > Results/AugmentedDataset/aug_cnn_retrain_rs.txt


# MODELS WITH SPEC AUGMENT

# No fine-tuning
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64
python classify_retrain.py -s True -r True -p True > Results/SpecAugment/sa_no_retrain_rs.txt

# Fine-tune fc layers
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64
python classify_retrain.py -s True -t 4 -r True -p True > Results/SpecAugment/sa_fc_retrain_rs.txt

# Fine-tune CNN & fc layers
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64
python classify_retrain.py -s True -t 7 -r True -p True > Results/SpecAugment/sa_cnn_retrain_rs.txt


# MODELS ON BOTH AUGMENTATIONS

# No fine-tuning
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64
python classify_retrain.py -s True -a True -r True -p True > Results/BothAugmentations/aug_sa_no_retrain_rs.txt

# Fine-tune fc layers
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64
python classify_retrain.py -s True -a True -t 4 -r True -p True > Results/BothAugmentations/aug_sa_fc_retrain_rs.txt

#Fine-tune CNN & fc layers
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64
python classify_retrain.py -s True -a True -t 7 -r True -p True > Results/BothAugmentations/aug_sa_cnn_retrain_rs.txt