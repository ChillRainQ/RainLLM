import os

CURRENT_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(CURRENT_DIR, "dataset")
OUT_DIR = os.path.join(CURRENT_DIR, "out")
PRETRAIN_DIR = os.path.join(OUT_DIR, "pretrain")
SFTTRAIN_DIR = os.path.join(OUT_DIR, "sft")
MODELS_DIR = os.path.join(CURRENT_DIR, "models")
TOKENIZER_PATH = os.path.join(MODELS_DIR, "new_tokenizer")
PRETRAIN_DATA_PATH = os.path.join(DATASET_DIR, "new_pre_hq.jsonl")
SFT_DATA_PATH = os.path.join(DATASET_DIR, "sft_512.jsonl")