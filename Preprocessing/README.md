# Preprocessing

The Python modules in this directory are used to carry out the data preprocessing procedure. 

The scripts are provided to automate the process. They should be used in the following order.


## Remove Silence

The remove_silence.py module is used to remove the silent sections of the sound files in the FreeSounds dataset. The resulting dataset will be stored in the Dataset/FreeSoundsSilent directory. 

A script is provided to do this automatically. Run using:

```bash
./remove_silence.sh
```

## Extend Sound Files

In order to extend sound files less than 10 seconds long, we use the loop_sound.py module. The resulting dataset will be stored in the Dataset/FreeSoundsExtended directory.

A script is provided to do this automatically. Run using:

```bash
./extend_sound_files.sh
```

## Data Augmentation (optional)

In order to perform the sound file manipulation data augmentation technique, which includes shifting the pitch and volume of the sound, we use the augmentation.py module. The resulting dataset will be stored in the FreeSoundsAugmented dataset.

A script is provided to do this automatically. Run using:

```bash
./augment.sh
```


## Generate Spectrograms

Finally, the last part of preprocessing is generating the spectrograms using the generate_spectrogram.py module. Special care must be taken when using this as we need to specify the path of the dataset whose spectrograms will be generated and we need to specify in the code, the location where spectrograms will be stored.

For example if we would like to generate the spectrograms for Keyboard category of the non-augmented FreeSounds dataset we will use the following: 

```bash
python generate_spectrogram.py Dataset/FreeSoundsExtended/Keyboard 
```
Furthermore, you will have to edit the path_ending variable of the module to:

```python
path_ending = "Dataset/FreeSoundsSpectrograms/" + str(sound_category)
```
 Similarly, if we want to generate the spectrograms for the augmented counterpart of the Keyboard category, we will use:
```bash
python generate_spectrogram.py Dataset/FreeSoundsAugmented/Keyboard 
```

And we also need to edit the path_ending variable to: 
```python
path_ending = "Dataset/FreeSoundsAugmentedSpectrograms/" + str(sound_category)
```
In order to automate the process, we pass a script that can be edited accordingly to generate spectrograms for the right dataset. This can be run with:

```bash
./generate_spectrograms.sh
```

