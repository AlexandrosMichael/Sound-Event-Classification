from classify_settings import *
import os
import argparse
import numpy as np
import time

np.random.seed(1337)  # for reproducibility
import pandas as pd
import seaborn as sn
from load_data import train_test_fs, train_fs_test_rs
from spec_augment import get_augmented_examples_mask, get_augmented_examples_warp
from sklearn.metrics import confusion_matrix
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, Flatten
from keras.preprocessing import sequence
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from keras.utils import to_categorical
from vggish_input import wavfile_to_examples
from keras.utils import plot_model
from vggish import VGGish


def create_vggish_extraction_model():
    # Create the model
    # Add the vgg convolutional base model
    model = VGGish(include_top=True)
    print(model.summary())

    for layer in model.layers:
        layer.trainable = False

    model.compile(loss='categorical_crossentropy',
                   optimizer='rmsprop',
                   metrics=['accuracy'])

    model.summary()
    return model


def create_LSTM_model(max_timesteps):
    model = Sequential()
    # model.add(Masking(mask_value=-1.0, input_shape=(max_timesteps, 128)))
    model.add(LSTM(units=64, return_sequences=True, input_shape=(max_timesteps, 128)))
    model.add(LSTM(units=64, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(len(CLASSIFICATION_CLASSES), activation='softmax'))
    print("Compiling ...")
    model.compile(loss="categorical_crossentropy", optimizer='rmsprop', metrics=["accuracy"])
    print(model.summary())
    plot_model(model, to_file='lstm_model.png', show_shapes=False, show_layer_names=True)
    return model


def add_augmented_files(X_files, y_categories):
    path_to_dataset = os.path.join(DATA_DIR, AUGMENTATION_DIR)
    augmented_features = []
    augmented_categories = []
    for sound_file in X_files:
        if sound_file[0] != '.':
            sound_category = sound_file.split('_')[0]
            path_to_directory = os.path.join(path_to_dataset, sound_category)
            path_to_file = os.path.join(path_to_directory, sound_file)
            # get all features of the sound files
            features = wavfile_to_examples(path_to_file)
            features = np.array(features)
            augmented_features.append(features)
            augmented_categories.append()


# gets a list of sound files returns a list of their features as extracted from VGGish
def generate_features(sound_files, directory, spec_augment):
    path_to_dataset = os.path.join(DATA_DIR, directory)
    sound_file_features = []
    for sound_file in sound_files:
        if sound_file[0] != '.':
            # magic to split by both - and _ for augmented files
            sound_file_temp = sound_file.replace("-", "_")
            sound_category = sound_file_temp.split('_')[0]
            path_to_directory = os.path.join(path_to_dataset, sound_category)
            path_to_file = os.path.join(path_to_directory, sound_file)
            # get all features of the sound files
            examples = np.load(path_to_file)
            if spec_augment:
                augmented_examples_mask = get_augmented_examples_mask(examples)
                augmented_examples_warp = get_augmented_examples_warp(examples)
                # print('augmented examples shape: ', augmented_examples.shape)
                # features = extract_examples_batch(path_to_file)
                examples = np.concatenate((examples, augmented_examples_mask, augmented_examples_warp))
                sound_file_features.append(examples)
            else:
                sound_file_features.append(examples)

    return sound_file_features


def generate_augmented_features(sound_files, directory, spec_augment):
    # print(sound_files)
    path_to_dataset = os.path.join(DATA_DIR, directory)
    sound_file_features = []
    for sound_file in sound_files:
        if sound_file[0] != '.':
            # magic to split by both - and _ for augmented files
            sound_category = sound_file.split('_')[0]
            path_to_directory = os.path.join(path_to_dataset, sound_category)
            sound_files_in_dir = os.listdir(path_to_directory)
            # print(sound_files)
            aug_features = []
            # print('sound file', sound_file)
            for sound_file_aug in sound_files_in_dir:
                # magic to match augmented file and original file
                sound_file_aug_tokens = sound_file_aug.replace('-', '.').split('.')
                if sound_file_aug_tokens[0] == sound_file.split('.')[0]:
                    # print('generate feature for', sound_file_aug)
                    path_to_file = os.path.join(path_to_directory, sound_file_aug)
                    # get all features of the sound files
                    examples = np.load(path_to_file)
                    if spec_augment:
                        augmented_examples_mask = get_augmented_examples_mask(examples)
                        augmented_examples_warp = get_augmented_examples_warp(examples)
                        examples = np.concatenate((examples, augmented_examples_mask, augmented_examples_warp))
                    for example in examples:
                        aug_features.append(example)
            sound_file_features.append(aug_features)
    return sound_file_features


def prepare_data(X_train_raw, X_test_raw, y_train_raw, y_test_raw):
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    for index, sound_file_features in enumerate(X_train_raw):
        for sample in sound_file_features:
            X_train.append(sample)
            y_train.append(y_train_raw[index])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    # y_train = to_categorical(np.array(y_train), len(CLASSIFICATION_CLASSES))

    for index, sound_file_features in enumerate(X_test_raw):
        for sample in sound_file_features:
            X_test.append(sample)
            y_test.append(y_test_raw[index])
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    # y_test = to_categorical(np.array(y_test), len(CLASSIFICATION_CLASSES))

    processed_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    return processed_data


def plot_graphs(history, args, df_cm):
    augmented_dataset = args.augmentation
    spec_augment = args.spec_augment
    trainable_layers = args.trainable_layers
    real_sounds = args.real_sounds

    file_str = ''

    if augmented_dataset and spec_augment:
        file_str = 'lstm_aug_sa'
    elif augmented_dataset:
        file_str = 'lstm_aug'
    elif spec_augment:
        file_str = 'lstm_sa'
    else:
        file_str = 'lstm_base'

    if trainable_layers == 0:
        file_str = file_str + '_no_retrain'
    elif trainable_layers == 4:
        file_str = file_str + '_fc_retrain'
    else:
        file_str = file_str + '_cnn_retrain'

    if real_sounds == 0:
        file_str = file_str + '_rs.png'
    else:
        file_str = file_str + '.png'


    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')

    if augmented_dataset and spec_augment:
        plt.savefig('Results/BothAugmentations/' + 'acc_' + file_str)
    elif augmented_dataset:
        plt.savefig('Results/AugmentedDataset/' + 'acc_' + file_str)
    elif spec_augment:
        plt.savefig('Results/SpecAugment/' +  'acc_' + file_str)
    else:
        plt.savefig('Results/Baseline/' + 'acc_' + file_str)

    plt.clf()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')

    if augmented_dataset and spec_augment:
        plt.savefig('Results/BothAugmentations/' + 'loss_' + file_str)
    elif augmented_dataset:
        plt.savefig('Results/AugmentedDataset/' + 'loss_' + file_str)
    elif spec_augment:
        plt.savefig('Results/SpecAugment/' +  'loss_' + file_str)
    else:
        plt.savefig('Results/Baseline/' + 'loss_' + file_str)

    plt.clf()

    plt.figure(figsize=(20, 14))
    sn.set(font_scale=1)  # for label size

    cf_fig = sn.heatmap(df_cm, cmap="Blues", annot=True, annot_kws={"size": 10}, fmt=".2f").get_figure()
    if augmented_dataset and spec_augment:
        cf_fig.savefig('Results/BothAugmentations/' + 'cm_' + file_str)
    elif augmented_dataset:
        cf_fig.savefig('Results/AugmentedDataset/' + 'cm_' + file_str)
    elif spec_augment:
        cf_fig.savefig('Results/SpecAugment/' + 'cm_' + file_str)
    else:
        cf_fig.savefig('Results/Baseline/' + 'cm_' + file_str)


def extract_features(args, train_files_with_labels, test_files_with_labels):
    augmented_dataset = args.augmentation
    spec_augment = args.spec_augment
    generate_plots = args.generate_plots
    real_sounds = args.real_sounds

    if augmented_dataset:
        training_dir = AUGMENTATION_DIR
    else:
        training_dir = TRAINING_DIR

    print('Directory used for training:', training_dir)

    if real_sounds:
        testing_dir = TESTING_DIR
    else:
        testing_dir = TRAINING_DIR

    print('Directory used for testing:', testing_dir)

    X_train = np.array(train_files_with_labels.get('X_train'))
    y_train = np.array(train_files_with_labels.get('y_train'))

    X_test = np.array(test_files_with_labels.get('X_test'))
    y_test = np.array(test_files_with_labels.get('y_test'))

    # training and test indices for sound files (their generated features)
    feature_extractor = create_vggish_extraction_model()

    if augmented_dataset:
        X_train_features = generate_augmented_features(X_train, training_dir, spec_augment)
    else:
        X_train_features = generate_features(X_train, training_dir, spec_augment)

    X_test_features = generate_features(X_test, TRAINING_DIR, False)

    prepared_data = prepare_data(X_train_features, X_test_features, y_train, y_test)

    X_train = prepared_data.get('X_train')
    X_test = prepared_data.get('X_test')
    # y_train = prepared_data.get('y_train')
    # y_test = prepared_data.get('y_test')

    y_train = to_categorical(y_train, len(CLASSIFICATION_CLASSES))
    y_test = to_categorical(y_test, len(CLASSIFICATION_CLASSES))

    X_train = X_train.reshape(X_train.shape[0], 96, 64, 1)
    X_test = X_test.reshape(X_test.shape[0], 96, 64, 1)

    print(X_train.shape, "train fold shape")
    print(X_test.shape, "test fold shape")

    X_train_extracted_features = feature_extractor.predict(X_train)
    print('X train extracted features shape', X_train_extracted_features.shape)

    all_features = []
    count = 0
    max_example_len = 0
    for example in X_train_features:
        file_features = []
        for i in range(len(example)):
            file_features.append(X_train_extracted_features[i + count])
        count += len(example)
        if len(example) > max_example_len:
            max_example_len = len(example)
        file_features = np.array(file_features)
        all_features.append(file_features)

    print('max example len', max_example_len)
    all_features = sequence.pad_sequences(all_features, maxlen=max_example_len, dtype="float32", value=-1.0)
    print('all features shape after padding', all_features.shape)

    # print('features shape', all_features.shape)

    X_test_extracted_features = feature_extractor.predict(X_test)
    print('X test extracted features shape', X_test_extracted_features.shape)

    all_test_features = []
    count = 0
    max_example_len = max_example_len
    for example in X_test_features:
        file_features = []
        for i in range(len(example)):
            file_features.append(X_test_extracted_features[i + count])
        count += len(example)
        if len(example) > max_example_len:
            max_example_len = len(example)
        file_features = np.array(file_features)
        all_test_features.append(file_features)

    print('max example len', max_example_len)
    all_test_features = sequence.pad_sequences(all_test_features, maxlen=max_example_len, dtype="float32", value=-1.0)

    model = create_LSTM_model(max_example_len)
    all_features = np.array(all_features)
    print('all features shape:', all_features.shape)
    print('a feature shape:', all_features[0].shape)

    start = time.time()
    history = model.fit(all_features, y_train, epochs=10, batch_size=1, validation_split=0.2, verbose=2,
                        shuffle=True)
    end = time.time()
    print('Training time elapsed in seconds', end - start)
    all_test_features = np.array(all_test_features)

    scores = model.evaluate(all_test_features, y_test, verbose=0)

    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    y_pred = model.predict(all_test_features)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=-1)

    y_pred_labels = []
    for p in y_pred:
        y_pred_labels.append(CLASSIFICATION_CLASSES[p])
    y_test_labels = []
    for t in y_test:
        y_test_labels.append(CLASSIFICATION_CLASSES[t])

    cr = classification_report(y_test_labels, y_pred_labels)
    print(cr)

    cf = confusion_matrix(y_test_labels, y_pred_labels, labels=CLASSIFICATION_CLASSES, normalize='true')
    # generate heatmap for confusion matrix
    df_cm = pd.DataFrame(cf, columns=np.unique(y_test_labels), index=np.unique(y_test_labels))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'

    if generate_plots:
        plot_graphs(history, args, df_cm)

def train_and_classify(args):
    # get file names and categories
    real_sounds = args.real_sounds
    if real_sounds:
        sound_dict = train_fs_test_rs()
    else:
        sound_dict = train_test_fs()

    X_train_files = sound_dict.get('training_files')
    y_train_categories = sound_dict.get('training_categories')

    X_test_files = sound_dict.get('testing_files')
    y_test_categories = sound_dict.get('testing_categories')

    y_train_labels = []

    # turn category names into labels
    for sample in y_train_categories:
        y_train_labels.append(CLASSIFICATION_CLASSES.index(sample))

    y_test_labels = []

    # turn category names into labels
    for sample in y_test_categories:
        y_test_labels.append(CLASSIFICATION_CLASSES.index(sample))

    training_files_with_labels = {
        'X_train': X_train_files,
        'y_train': y_train_labels
    }

    testing_files_with_labels = {
        'X_test': X_test_files,
        'y_test': y_test_labels
    }

    # cross_validate(training_files_with_labels)
    extract_features(args, training_files_with_labels, testing_files_with_labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-a', '--augment', dest='augmentation', type=bool, default=False,
                        help='use augmented dataset')
    parser.add_argument('-s', '--spec_augment', dest='spec_augment', type=bool, default=False, help='use spec augment')
    parser.add_argument('-r', '--real_sounds', dest='real_sounds', type=bool, default=False,
                        help='evaluate on real sounds')
    parser.add_argument('-p', '--plots', dest='generate_plots', type=bool, default=False,
                        help='evaluate on real sounds')

    args = parser.parse_args()
    print(args)
    # train_and_classify(args)
    create_LSTM_model(1)