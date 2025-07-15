# src/config.py
import torch
from rdkit.Chem import rdFingerprintGenerator

# Настройки устройства
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Глобальный генератор отпечатков Morgan
MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# Константы для пайплайна
N_MOLECULES_TO_FIND = 5
PRETRAIN_EPOCHS = 5
RL_ITERATIONS = 10
BATCH_SIZE = 16
MAX_LEN_SELFIE = 120 # Максимальная длина SELFIES

# URL и пути
MOSES_DATA_URL = "https://media.githubusercontent.com/media/molecularsets/moses/refs/heads/master/data/train.csv"
KEAP1_TARGET_ID = 'CHEMBL2069156'
EGFR_TARGET_ID = 'CHEMBL203'
IKKB_TARGET_ID = 'CHEMBL2094'

# Пути для сохранения моделей
KEAP1_MODEL_PATH = 'models/keap1_activity_model.pkl'
EGFR_MODEL_PATH = 'models/egfr_activity_model.pkl'
IKKB_MODEL_PATH = 'models/ikkb_activity_model.pkl'
GENERATOR_PRETRAINED_PATH = 'models/generator_pretrained.pth'
GENERATOR_FINAL_PATH = 'models/generator_final.pth'
