from classify_settings import *
import os
import time
import argparse
import numpy as np
import pandas as pd
import seaborn as sn
np.random.seed(1337)  # for reproducibility
from load_data import train_test_fs, train_fs_test_rs
from spec_augment import get_augmented_examples_mask, get_augmented_examples_warp
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from keras.utils import plot_model

from keras.models import Model
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt

from sklearn.metrics import classification_report
from keras.utils import to_categorical
from sklearn.utils import shuffle
from vggish_input import wavfile_to_examples
from vggish import VGGish


# instantiate the VGGish+ model
def create_vggish_model(trainable_layers):
    # Create the model
    # Add the vgg convolutional base model
    vggish = VGGish(include_top=True)

    # Freeze the layers except the last 4 layers
    for layer in vggish.layers:
        layer.trainable = False

    if trainable_layers > 0:
        for layer in vggish.layers[-trainable_layers:]:
            layer.trainable = True
    # Check the trainable status of the individual layers
    for layer in vggish.layers:
        print(layer, layer.trainable)

    x = vggish.output

    predictions = Dense(len(CLASSIFICATION_CLASSES), activation='softmax')(x)
    model = Model(input=vggish.input, output=predictions)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    print(model.summary())
    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
    return model


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


# gets a list of sound files returns a list of their features as extracted from VGGish and uses SpecAugment on them
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
            aug_features = []
            for sound_file_aug in sound_files_in_dir:
                # magic to match augmented file and original file
                sound_file_aug_tokens = sound_file_aug.replace('-', '.').split('.')
                if sound_file_aug_tokens[0] == sound_file.split('.')[0]:
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


# prepare the data for model training
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

    for index, sound_file_features in enumerate(X_test_raw):
        for sample in sound_file_features:
            X_test.append(sample)
            y_test.append(y_test_raw[index])
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    processed_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    return processed_data


# method used to cross validate model.
# Used during development only, not for any of the experiments.
def cross_validate(sound_files_with_labels):
    X_train = np.array(sound_files_with_labels.get('X_train'))
    y_train = np.array(sound_files_with_labels.get('y_train'))

    k_fold = StratifiedKFold(n_splits=5, shuffle=False)

    cv_scores = []

    # training and test indices for sound files (their generated features)
    for train, test in k_fold.split(X_train, y_train):
        model = create_vggish_model()
        X_train_files = X_train[train]
        X_val_files = X_train[test]

        y_train_categories = y_train[train]
        y_val_categories = y_train[test]

        X_train_features = generate_features(X_train_files, TRAINING_DIR)
        X_val_features = generate_features(X_val_files, TRAINING_DIR)

        prepared_data = prepare_data(X_train_features, X_val_features, y_train_categories, y_val_categories)
        X_train_fold = prepared_data.get('X_train')
        X_test_fold = prepared_data.get('X_test')
        y_train_fold = prepared_data.get('y_train')
        y_test_fold = prepared_data.get('y_test')

        X_train_fold = X_train_fold
        X_test_fold = X_test_fold

        X_train_fold, y_train_fold = shuffle(X_train_fold, y_train_fold, random_state=42)
        X_test_fold, y_test_fold = shuffle(X_test_fold, y_test_fold, random_state=23)

        y_train_fold = to_categorical(y_train_fold, len(CLASSIFICATION_CLASSES))
        y_test_fold = to_categorical(y_test_fold, len(CLASSIFICATION_CLASSES))

        X_train_fold = X_train_fold.reshape(X_train_fold.shape[0], 96, 64, 1)
        X_test_fold = X_test_fold.reshape(X_test_fold.shape[0], 96, 64, 1)

        print(X_train_fold.shape, "train fold shape")
        print(X_test_fold.shape, "test fold shape")

        early_stopping = EarlyStopping(monitor='val_loss', patience=2)

        start = time.time()
        model.fit(X_train_fold, y_train_fold, epochs=10, batch_size=8, verbose=0, callbacks=[early_stopping],
                  validation_data=(X_test_fold,
                                   y_test_fold),
                  shuffle=True)

        end = time.time()
        print('Training time elapsed in seconds', end-start)

        # evaluate the model
        scores = model.evaluate(X_test_fold, y_test_fold, verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        cv_scores.append(scores[1] * 100)

    print("VGGish tuning CV Scores")
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cv_scores), np.std(cv_scores)))


# method used to plot graphs for each of the experiments
def plot_graphs(history, args, df_cm):
    augmented_dataset = args.augmentation
    spec_augment = args.spec_augment
    trainable_layers = args.trainable_layers
    real_sounds = args.real_sounds

    file_str = ''

    if augmented_dataset and spec_augment:
        file_str = 'aug_sa'
    elif augmented_dataset:
        file_str = 'aug'
    elif spec_augment:
        file_str = 'sa'
    else:
        file_str = 'base'

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


# method  which trains a model and then carries out evaluation on the model
def train_and_evaluate(args, train_files_with_labels, test_files_with_labels):

    augmented_dataset = args.augmentation
    spec_augment = args.spec_augment
    trainable_layers = args.trainable_layers
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
    model = create_vggish_model(trainable_layers)

    if augmented_dataset:
        X_train_features = generate_augmented_features(X_train, training_dir, spec_augment)
    else:
        X_train_features = generate_features(X_train, training_dir, spec_augment)

    X_test_features = generate_features(X_test, testing_dir, False)

    prepared_data = prepare_data(X_train_features, X_test_features, y_train, y_test)

    X_train = prepared_data.get('X_train')
    X_test = prepared_data.get('X_test')
    y_train = prepared_data.get('y_train')
    y_test = prepared_data.get('y_test')

    X_train, y_train = shuffle(X_train, y_train, random_state=42)

    X_test, y_test = shuffle(X_test, y_test, random_state=23)

    y_train = to_categorical(y_train, len(CLASSIFICATION_CLASSES))
    y_test = to_categorical(y_test, len(CLASSIFICATION_CLASSES))

    X_train = X_train.reshape(X_train.shape[0], 96, 64, 1)
    X_test = X_test.reshape(X_test.shape[0], 96, 64, 1)

    print(X_train.shape, "train fold shape")
    print(X_test.shape, "test fold shape")

    start = time.time()
    history = model.fit(X_train, y_train, epochs=20, batch_size=8, verbose=2, shuffle=True,
                        validation_split=0.1)
    end = time.time()
    print('Training time elapsed in seconds', end-start)

    # evaluate the model
    scores = model.evaluate(X_test, y_test, verbose=0)

    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    y_pred = model.predict(X_test)
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

    # path_to_model = os.path.join(MODELS_DIR, 'vgg_fc_retrain.h5')
    # model.save(filepath=path_to_model, overwrite=True)


def main(args):
    # get file names and categories

    # if real sounds parameter - evaluate model on real sounds
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
    train_and_evaluate(args, training_files_with_labels, testing_files_with_labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-a', '--augment', dest='augmentation', type=bool, default=False,
                        help='use augmented dataset')
    parser.add_argument('-s', '--spec_augment', dest='spec_augment', type=bool, default=False, help='use spec augment')
    parser.add_argument('-t', '--trainable', dest='trainable_layers', type=int, default=0,
                        help='number of trainable layers')
    parser.add_argument('-r', '--real_sounds', dest='real_sounds', type=bool, default=False,
                        help='evaluate on real sounds')
    parser.add_argument('-p', '--plots', dest='generate_plots', type=bool, default=False,
                        help='evaluate on real sounds')

    args = parser.parse_args()
    print(args)
    main(args)
    create_vggish_model(0)