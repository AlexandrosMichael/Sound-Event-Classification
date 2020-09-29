from __future__ import print_function
import vggish_input


def extract_examples_batch(wav_file):
    examples_batch = vggish_input.wavfile_to_examples(wav_file)
    return examples_batch
