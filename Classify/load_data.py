from classify_settings import *
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# loading the free sounds files and splitting them into training and testing sets
def train_test_fs():
    # getting the training set
    print("Getting FS set...")

    file_names = []
    file_categories = []

    path_to_training = os.path.join(DATA_DIR, TRAINING_DIR)
    dir_training_contents = os.listdir(path_to_training)

    for sub_dir in dir_training_contents:
        if sub_dir[0] != "." and (sub_dir in CLASSIFICATION_CLASSES):
            path_to_sound_files = os.path.join(path_to_training, sub_dir)
            sound_files = os.listdir(path_to_sound_files)
            for sound_file in sound_files:
                if sound_file[0] != '.':
                    file_names.append(sound_file)
                    file_categories.append(sub_dir)

    file_names, train_categories = shuffle(file_names, file_categories, random_state=1)

    X_train, X_test, y_train, y_test = train_test_split(file_names, train_categories, test_size=0.2,
                                                        stratify=train_categories,
                                                        random_state=42)

    sound_dict = {
        'training_files': X_train,
        'training_categories': y_train,
        'testing_files': X_test,
        'testing_categories': y_test
    }
    print('Loaded FreeSounds train-test set!')
    print('Number of files in training set:', len(X_train))
    print('Number of files in testing set:', len(X_test))

    return sound_dict


# loading the free sounds files for training set and real sounds for testing set
def train_fs_test_rs():
    # getting the training set
    print("Getting RealSounds set...")

    file_names = []
    file_categories = []

    path_to_training = os.path.join(DATA_DIR, TRAINING_DIR)
    dir_training_contents = os.listdir(path_to_training)

    for sub_dir in dir_training_contents:
        if sub_dir[0] != "." and (sub_dir in CLASSIFICATION_CLASSES):
            path_to_sound_files = os.path.join(path_to_training, sub_dir)
            sound_files = os.listdir(path_to_sound_files)
            for sound_file in sound_files:
                if sound_file[0] != '.':
                    file_names.append(sound_file)
                    file_categories.append(sub_dir)

    train_file_names, train_categories = shuffle(file_names, file_categories, random_state=1)

    file_names = []
    file_categories = []

    print("Getting RealSounds set...")
    path_to_testing = os.path.join(DATA_DIR, TESTING_DIR)
    dir_testing_contents = os.listdir(path_to_testing)

    for sub_dir in dir_testing_contents:
        if sub_dir[0] != "." and (sub_dir in CLASSIFICATION_CLASSES):
            path_to_sound_files = os.path.join(path_to_testing, sub_dir)
            sound_files = os.listdir(path_to_sound_files)
            for sound_file in sound_files:
                if sound_file[0] != '.':
                    file_names.append(sound_file)
                    file_categories.append(sub_dir)

    test_file_names, test_categories = shuffle(file_names, file_categories, random_state=1)

    sound_dict = {
        'training_files': train_file_names,
        'training_categories': train_categories,
        'testing_files': test_file_names,
        'testing_categories': test_categories
    }
    print('Loaded FreeSounds train set and RealSounds test set!')
    print('Number of files in training set:', len(train_file_names))
    print('Number of files in testing set:', len(test_file_names))

    return sound_dict
