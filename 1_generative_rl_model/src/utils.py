# src/utils.py
import numpy as np
import selfies as sf
from rdkit import Chem, DataStructs
from .config import MORGAN_GENERATOR

def pareto_frontier(df, objectives=['pIC50_KEAP1', 'QED', 'BBB_Score', 'CNS_MPO', 'SA_Score'], maximize=[True, True, True, True, False]):
    """Фильтрует DataFrame для нахождения границы Парето."""
    points = df[objectives].values
    n_points = len(points)
    is_efficient = np.ones(n_points, dtype=bool)

    for i in range(n_points):
        if not is_efficient[i]:
            continue
        for j in range(n_points):
            if i == j:
                continue
            dominates = True
            for k, max_flag in enumerate(maximize):
                if max_flag:
                    if points[j][k] > points[i][k]:
                        dominates = False
                        break
                else:
                    if points[j][k] < points[i][k]:
                        dominates = False
                        break
            if dominates:
                is_efficient[i] = False
                break
    return df[is_efficient].reset_index(drop=True)

def calculate_metrics(smiles_list, reference_smiles):
    """Рассчитывает разнообразие, новизну и уникальность."""
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_list if Chem.MolFromSmiles(smi)]
    ref_mols = [Chem.MolFromSmiles(smi) for smi in reference_smiles if Chem.MolFromSmiles(smi)]
    fps = [MORGAN_GENERATOR.GetFingerprint(mol) for mol in mols]

    # Разнообразие
    diversity = 0.0
    if len(fps) >= 2:
        similarities = [1 - DataStructs.TanimotoSimilarity(fps[i], fps[j]) for i in range(len(fps)) for j in range(i + 1, len(fps))]
        diversity = np.mean(similarities) if similarities else 0.0

    # Новизна
    ref_fps = [MORGAN_GENERATOR.GetFingerprint(mol) for mol in ref_mols]
    novel_count = sum(1 for fp in fps if all(DataStructs.TanimotoSimilarity(fp, ref_fp) < 0.4 for ref_fp in ref_fps))
    novelty = novel_count / max(1, len(fps))

    # Уникальность
    uniqueness = len(set(smiles_list)) / max(1, len(smiles_list))

    return {'diversity': diversity, 'novelty': novelty, 'uniqueness': uniqueness}

def augment_smiles(smiles, num_variants=2):
    """Аугментация SMILES путем генерации случайных вариантов."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return [Chem.MolToSmiles(mol, doRandom=True) for _ in range(num_variants) if mol]
    except:
        return [smiles]

def smiles_to_selfies(smiles_string):
    """Безопасное кодирование SMILES в SELFIES."""
    try:
        return sf.encoder(smiles_string)
    except:
        return None

def selfies_to_smiles(selfies_string):
    """Безопасное декодирование SELFIES в SMILES."""
    try:
        return sf.decoder(selfies_string)
    except:
        return None
