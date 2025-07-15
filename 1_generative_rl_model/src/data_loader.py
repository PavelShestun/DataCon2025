# src/data_loader.py
import pandas as pd
import numpy as np
import requests
import io
from chembl_webresource_client.new_client import new_client
from .config import KEAP1_TARGET_ID, EGFR_TARGET_ID, IKKB_TARGET_ID, MOSES_DATA_URL
import logging

logger = logging.getLogger(__name__)

def load_chembl_data(target_id, limit=None):
    """Загружает данные об активности для указанного ChEMBL ID."""
    try:
        activity = new_client.activity
        res = activity.filter(target_chembl_id=target_id, standard_type="IC50").only('canonical_smiles', 'standard_units', 'standard_value')
        if limit:
            res = res[:limit]
        return pd.DataFrame(res)
    except Exception as e:
        logger.error(f"Не удалось загрузить данные для {target_id}: {e}")
        return pd.DataFrame()

def preprocess_for_modeling(df):
    """Выполняет предварительную обработку DataFrame для моделирования."""
    if df.empty:
        return None
    df = df.dropna(subset=['canonical_smiles', 'standard_value', 'standard_units'])
    df = df[df['standard_units'] == 'nM']
    df['standard_value'] = pd.to_numeric(df['standard_value'], errors='coerce')
    df = df.dropna(subset=['standard_value'])
    df = df[df['standard_value'] > 0]
    df = df.drop_duplicates(subset=['canonical_smiles'])
    df['pIC50'] = -np.log10(df['standard_value'] * 1e-9)
    return df

def get_all_datasets(max_retries=3):
    """Загружает и обрабатывает все необходимые наборы данных."""
    for attempt in range(max_retries):
        keap1_df = load_chembl_data(KEAP1_TARGET_ID)
        egfr_df = load_chembl_data(EGFR_TARGET_ID, limit=5000)
        ikkb_df = load_chembl_data(IKKB_TARGET_ID, limit=5000)

        if not keap1_df.empty and not egfr_df.empty and not ikkb_df.empty:
            keap1_clean = preprocess_for_modeling(keap1_df)
            egfr_clean = preprocess_for_modeling(egfr_df)
            ikkb_clean = preprocess_for_modeling(ikkb_df)
            return keap1_clean, egfr_clean, ikkb_clean

        logger.warning(f"Попытка {attempt + 1}/{max_retries} не удалась. Повторная попытка...")
    return None, None, None

def load_moses_data():
    """Загружает набор данных MOSES."""
    try:
        moses_df = pd.read_csv(io.StringIO(requests.get(MOSES_DATA_URL).text), header=0)
        return moses_df['SMILES'].dropna().unique().tolist()
    except Exception as e:
        logger.error(f"Не удалось загрузить MOSES: {e}")
        return []
