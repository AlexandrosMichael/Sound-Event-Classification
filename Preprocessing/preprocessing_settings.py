import os

# Settings file for classification
# Set according to the paths and directory structure of your machine

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = "/cs/tmp/am425/SH/sound"
DATA_DIR = "/Users/alexandrosmichael/Desktop/CS/SH/Code"

SEGMENT_LENGTH = 5000

OVERLAP_RATE = 0.25

EXPORT_RATE = 16000

EXPORT_BIT_RATE = 16