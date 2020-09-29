# Settings file for classification
# Set according to the paths and directory structure of your machine
import os

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')

SYNTH_DIR = "Dataset/FreeSounds"
REAL_DIR = "Dataset/RealSounds"

TRAINING_DIR = "Dataset/FreeSoundsSpectrograms"
AUGMENTATION_DIR = "Dataset/FreeSoundsAugmentedSpectrograms"
TESTING_DIR = "Dataset/RealSoundsSpectrograms"

DATA_DIR = "/cs/tmp/am425/SH/sound"
# DATA_DIR = "/Users/alexandrosmichael/Desktop/CS/SH/Code"

# MODELS_DIR = "/Users/alexandrosmichael/Desktop/CS/SH/Code/Classify/Models"
MODELS_DIR = "/cs/tmp/am425/SH/Models"


# removed car, dishwasher, conversation, phone ringing and photocopier
CLASSIFICATION_CLASSES = ["Chair", "ClearThroat", "CoffeeMachine", "Coughing", "DoorKnock", "DoorSlam",
                          "Drawer", "FallingObject", "FootSteps", "Keyboard", "Laughing", "MilkSteamer", "Sink", "Sneezing", "Stiring"]

