# 2_comparison_diffsbdd/src/evaluator.py
import logging
import joblib
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, Crippen, rdFingerprintGenerator
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from tqdm import tqdm
from openbabel import pybel

class MoleculeEvaluator:
    """
    Класс для оценки свойств сгенерированных молекул.
    """
    def __init__(self, config: dict):
        self.config = config
        self.morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        self.pains_catalog = self._init_pains_catalog()
        self.models = self._load_predictor_models()

    def _init_pains_catalog(self):
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        return FilterCatalog(params)
    
    def _load_predictor_models(self):
        """Загружает обученные модели для предсказания pIC50."""
        models = {}
        model_paths = {
            "keap1": self.config["KEAP1_MODEL_PATH"],
            "egfr": self.config["EGFR_MODEL_PATH"],
            "ikkb": self.config["IKKB_MODEL_PATH"],
        }
        for name, path in model_paths.items():
            try:
                models[name] = joblib.load(path)
                logging.info(f"Модель-предиктор для {name.upper()} успешно загружена.")
            except FileNotFoundError:
                logging.error(f"Файл модели не найден: {path}. Убедитесь, что вы сначала запустили пайплайн '1_generative_rl_model'.")
                models[name] = None
            except Exception as e:
                logging.error(f"Ошибка загрузки модели {path}: {e}")
                models[name] = None
        return models

    def sdf_to_smiles(self, sdf_path: str) -> list:
        """Читает SDF файл и извлекает из него SMILES строки."""
        try:
            mols = [m for m in pybel.readfile("sdf", sdf_path)]
            smiles_list = [m.write("smi").split('\t')[0] for m in mols]
            logging.info(f"Извлечено {len(smiles_list)} SMILES из файла {sdf_path}.")
            return smiles_list
        except Exception as e:
            logging.error(f"Ошибка чтения SDF файла {sdf_path}: {e}")
            return []

    def evaluate_smiles_list(self, smiles_list: list) -> pd.DataFrame:
        """Применяет полную оценку свойств ко списку SMILES."""
        properties_list = []
        for smiles in tqdm(smiles_list, desc="Оценка свойств молекул"):
            props = self._calculate_properties_for_smiles(smiles)
            if props:
                properties_list.append(props)
        return pd.DataFrame(properties_list)

    def _calculate_properties_for_smiles(self, smiles: str) -> dict or None:
        """Расчитывает полный набор свойств для одной молекулы."""
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return None

        # Простые фильтры
        if self.pains_catalog.HasMatch(mol): return None
        
        # Расчет свойств
        try:
            features = self._get_features(mol)
            result = {
                'SMILES': smiles,
                'QED': QED.qed(mol),
                'SA_Score': Descriptors.CalcCrippenDescriptors(mol)[1], # SA Score из Crippen
                'MolWt': Descriptors.MolWt(mol),
                'LogP': Crippen.MolLogP(mol),
                'pIC50_KEAP1': self.models['keap1'].predict(features)[0] if self.models['keap1'] else 0,
                'pIC50_EGFR': self.models['egfr'].predict(features)[0] if self.models['egfr'] else 0,
                'pIC50_IKKb': self.models['ikkb'].predict(features)[0] if self.models['ikkb'] else 0,
            }
            result['Selectivity_Score'] = result['pIC50_KEAP1'] - max(result['pIC50_EGFR'], result['pIC50_IKKb'])
            return result
        except Exception as e:
            logging.warning(f"Ошибка расчета свойств для {smiles}: {e}")
            return None

    def _get_features(self, mol: Chem.Mol) -> np.ndarray:
        """Генерирует признаки (отпечатки и дескрипторы) для модели."""
        fp = np.array(self.morgan_gen.GetFingerprint(mol)).reshape(1, -1)
        desc_list = [
            Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA,
            Descriptors.NumHAcceptors, Descriptors.NumHDonors, Descriptors.NumRotatableBonds,
            Descriptors.FractionCSP3, Descriptors.RingCount, Descriptors.NumAromaticRings,
            Descriptors.HeavyAtomCount
        ]
        desc = np.array([func(mol) for func in desc_list]).reshape(1, -1)
        return np.hstack((fp, desc))
    
    def filter_and_score(self, df: pd.DataFrame, top_n=10) -> pd.DataFrame:
        """Применяет финальную фильтрацию и скоринг."""
        if df.empty: return pd.DataFrame()
        
        # Здесь можно добавить более сложные фильтры, если нужно
        
        # Расчет итоговой оценки (простая версия для примера)
        df['Score'] = df['pIC50_KEAP1'] + df['Selectivity_Score'] * 0.5 + df['QED'] - df['SA_Score'] * 0.1
        
        return df.sort_values(by='Score', ascending=False).reset_index(drop=True)
