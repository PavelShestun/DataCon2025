import os
import pandas as pd
import logging
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

def generate_3d_conformers(csv_path: str, output_dir: str):
    """
    Читает SMILES из CSV файла, генерирует для них 3D конформеры
    и сохраняет в формате PDB.
    """
    logging.info(f"Начало подготовки лигандов из файла: {csv_path}")
    os.makedirs(output_dir, exist_ok=True)

    try:
        df = pd.read_csv(csv_path)
        if 'SMILES' not in df.columns:
            logging.error(f"В CSV файле '{csv_path}' отсутствует колонка 'SMILES'.")
            return
    except FileNotFoundError:
        logging.error(f"Файл не найден: {csv_path}. Убедитесь, что вы сначала запустили пайплайн генерации.")
        return

    logging.info(f"Найдено {len(df)} молекул для подготовки.")
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Подготовка лигандов"):
        smi = row['SMILES']
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            logging.warning(f"Не удалось распознать SMILES: {smi} (строка {index + 2})")
            continue

        mol_id = f"ligand_{index + 1}"
        mol = Chem.AddHs(mol)

        # Используем ETKDG v3 для лучшей генерации конформеров
        embed_params = AllChem.ETKDGv3()
        embed_params.randomSeed = 42 # Для воспроизводимости
        embed_status = AllChem.EmbedMolecule(mol, embed_params)
        
        if embed_status == -1: # -1 означает ошибку
            logging.warning(f"Не удалось сгенерировать конформер для {mol_id} ({smi})")
            continue

        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except ValueError as e:
            logging.warning(f"Ошибка оптимизации для {mol_id} ({smi}): {e}")
            continue

        pdb_filename = os.path.join(output_dir, f"{mol_id}.pdb")
        Chem.MolToPDBFile(mol, pdb_filename)

    logging.info(f"Подготовка лигандов завершена. Файлы сохранены в: {output_dir}")
