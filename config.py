import torch

# ========= paramètres utile le long du projet =========

data_train = "dataset/train.txt"
data_val = "dataset/val.txt"
data_test = "dataset/test.txt"

# Modes disponibles :
# "train" | "oneword" | "shortseq" | "dataloader" | "grid"
MODE = "train"

# Hyperparamètres d'entraînement
EPOCHS = 20
LR = 0.0001
BATCH_SIZE = 64
MAX_LEN = 35
MIN_FREQ = 3
HIDDEN_SIZE = 384
EMB_SIZE = 128
DROPOUT = 0.2
SEP = ";"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
