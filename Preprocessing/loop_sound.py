import os
import sys
from preprocessing_settings import *
from pydub import AudioSegment

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
        path_ending = "Dataset/FreeSoundsExtended/" + str(sound_class)
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
        audio_chunk = sound_data
        while len(sound_data) < 10000:
            sound_data += audio_chunk
        new_final_path = os.path.join(new_path, file)
        print(new_final_path)
        sound_data.export(new_final_path, format="wav")
