import os
import sys
from preprocessing_settings import *
from pydub import AudioSegment
from pydub.silence import split_on_silence

# take in all of the command line arguments
arguments = sys.argv[1:]

# path in which the sound files exist
path_to_directory = arguments[0]

# get path to directory with sound files
path = os.path.join(DATA_DIR, path_to_directory)

# get directory contents
dir_contents = os.listdir(path)

# for each sound file in the directory
for file in dir_contents:
    if file[0] != ".":
        print("File Name", file)
        file_path = os.path.join(path, file)
        # file path to original sound
        print("File Path", file_path)
        file_tokens = file.replace('.', '_').split('_')
        sound_class = file_tokens[0]
        path_ending = "Dataset/FreeSoundsSilent/" + str(sound_class)
        print("Path ending", path_ending)
        new_path = os.path.join(DATA_DIR, path_ending)
        print("New Path", new_path)

        file_tokens = file.replace('.', '_').split('_')
        sound_class = file_tokens[0]
        sound_class_file = int(file_tokens[1])
        sound_file_components = {
            'sound_class': sound_class,
            'sound_class_file': sound_class_file,
        }

        sound_data = AudioSegment.from_wav(file_path)
        print('dbfs', sound_data.dBFS)
        audio_chunks = split_on_silence(sound_data,
                                        # must be silent for at least 1000ms
                                        min_silence_len=1000,
                                        # consider it silent if quieter than average loudness
                                        silence_thresh=-16)
        for chunk in audio_chunks:
            print('after', chunk.dBFS)
        new_sound_data = AudioSegment.empty()
        if len(audio_chunks) == 0:
            new_sound_data = sound_data
        else:
            for i, chunk in enumerate(audio_chunks):
                new_sound_data += chunk

        new_final_path = os.path.join(new_path, file)
        new_sound_data.export(new_final_path, format="wav")

