import os

CURRENT_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(CURRENT_DIR, "dataset")
OUT_DIR = os.path.join(CURRENT_DIR, "out")
PRETRAIN_DIR = os.path.join(OUT_DIR, "pretrain")
SFTTRAIN_DIR = os.path.join(OUT_DIR, "sft")
MODELS_DIR = os.path.join(CURRENT_DIR, "models")
TOKENIZER_PATH = os.path.join(MODELS_DIR, "tokenizer")
PRETRAIN_DATA_PATH = os.path.join(DATASET_DIR, "pretrain_hq.jsonl")