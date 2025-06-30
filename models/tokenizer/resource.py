import os.path


def get_vocab_size():
    return 1000

VOCAB_SIZE = get_vocab_size()
CURRENT_DIR = os.path.dirname(__file__)
TOKENIZER_PATH = CURRENT_DIR