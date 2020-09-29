import sys
import numpy as np
import librosa
import matplotlib.pyplot as plt
from pydub import AudioSegment
from preprocessing_settings import *

from random import randint

n = 0


def load_audio_file(file_path):
    data = librosa.core.load(file_path)[0]
    return data


def plot_time_series(data):
    plt.figure(figsize=(14, 8))
    plt.title('Raw wave ')
    plt.ylabel('Amplitude')
    plt.plot(np.linspace(0, 1, len(data)), data)
    plt.show()


def pitch_change(fname, path, magnification, up, new_name):
    sound = AudioSegment.from_file(fname, format="wav")
    # shift the pitch up by half an octave (speed will increase proportionally)
    octaves = 0.25 * magnification
    new_sample_rate = int(sound.frame_rate * (2.0 ** octaves))
    new_sample_rate2 = int(sound.frame_rate / (2.0 ** octaves))
    # keep the same samples but tell the computer they ought to be played at the
    # new, higher sample rate. This file sounds like a chipmunk but has a weird sample rate.
    pitch_sound = sound._spawn(sound.raw_data, overrides={'frame_rate': new_sample_rate})
    pitch_sound2 = sound._spawn(sound.raw_data, overrides={'frame_rate': new_sample_rate2})
    # convert it to a common sample rate (44.1k - standard audio CD)
    pitch_sound = pitch_sound.set_frame_rate(44100)
    pitch_sound2 = pitch_sound2.set_frame_rate(44100)
    # export / save pitch changed sound
    if up:
        pitch_sound.export(path + "/" + new_name, format="wav")
    else:
        pitch_sound2.export(path + "/" + new_name, format="wav")


def volume_change(fname, path, magnification, up, new_name):
    sound = AudioSegment.from_file(fname, format="wav")
    if up:
        sound_vol = sound + 5 * magnification
    else:
        sound_vol = sound - 5 * magnification
    sound_vol.export(path + "/" + new_name, format="wav")

def original(fname, path, new_name):
    sound = AudioSegment.from_file(fname, format="wav")

    sound.export(path + "/" + new_name, format="wav")


# take in all of the command line arguments
arguments = sys.argv[1:]

# path in which the sound files exist
path_to_directory = arguments[0]

# augmentation to perform

# get path to directory with sound files
path = os.path.join(DATA_DIR, path_to_directory)

# get directory contents
dir_contents = os.listdir(path)

# for each sound file in the directory
for file in dir_contents:
    if file[0] != ".":
        print(file)
        file_path = os.path.join(path, file)
        # file path to original sound
        print(file_path)
        # with contextlib.closing(wave.open(file_path, 'r')) as f:
        #
        file_tokens = file.replace('.', '_').split('_')
        sound_class = file_tokens[0]
        path_ending = "Dataset/FreeSoundsAugmented/" + str(sound_class)
        print(path_ending)
        new_path = os.path.join(DATA_DIR, path_ending)
        print(new_path)
        magnification_choice = randint(1, 2)

        file_tokens = file.replace('.', '_').split('_')
        sound_class = file_tokens[0]
        sound_class_file = int(file_tokens[1])
        sound_file_components = {
            'sound_class': sound_class,
            'sound_class_file': sound_class_file,
        }

        original(file_path, new_path, sound_file_components['sound_class'] + "_" + str(sound_file_components['sound_class_file']) + ".wav" )
        pitch_change(file_path, new_path, magnification_choice, True, sound_file_components['sound_class'] + "_" + str(sound_file_components['sound_class_file']) + "-PCU" + ".wav")
        volume_change(file_path, new_path, magnification_choice, True, sound_file_components['sound_class'] + "_" + str(sound_file_components['sound_class_file']) + "-VCU" + ".wav")
        pitch_change(file_path, new_path, magnification_choice, False, sound_file_components['sound_class'] + "_" + str(sound_file_components['sound_class_file']) + "-PCD" + ".wav")
        volume_change(file_path, new_path, magnification_choice, False, sound_file_components['sound_class'] + "_" + str(sound_file_components['sound_class_file']) + "-VCD" + ".wav")
