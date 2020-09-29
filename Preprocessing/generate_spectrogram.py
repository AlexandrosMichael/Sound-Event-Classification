import sys
import numpy as np
from preprocessing_settings import *
from vggish_input import wavfile_to_examples

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

        # magic to split by both - and _ for augmented files
        sound_file_temp = file.replace("-", "_")
        sound_category = sound_file_temp.split('_')[0]

        path_to_file = os.path.join(path, file)

        print("File Name", file)
        file_path = os.path.join(path, file)

        path_ending = "Dataset/FreeSoundsSpectrograms/" + str(sound_category)
        print("Path ending", path_ending)
        new_path = os.path.join(DATA_DIR, path_ending)

        data = wavfile_to_examples(path_to_file)

        new_file_name = file[:-4]
        new_final_path = os.path.join(new_path, new_file_name)

        print(new_final_path)

        np.save(new_final_path, data)
