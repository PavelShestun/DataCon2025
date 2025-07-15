# 2_comparison_diffsbdd/main.py

import sys
import os
import logging
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

# Добавляем корневую директорию проекта в путь
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

from src.generator import DiffSBDDGenerator
from src.evaluator import MoleculeEvaluator

def main():
    """
    Основной пайплайн для генерации молекул с DiffSBDD и их последующей оценки.
    """
    # Настройка логирования
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("--- ЗАПУСК ПАЙПЛАЙНА СРАВНЕНИЯ С DIFFSBDD ---")
    
    # --- Шаг 1: Генерация молекул с помощью DiffSBDD ---
    generator = DiffSBDDGenerator(config.DIFFSBDD_CONFIG)
    generated_sdf_path = generator.run_generation()
    if not generated_sdf_path:
        logging.error("Генерация молекул не удалась. Завершение работы.")
        return

    # --- Шаг 2: Оценка сгенерированных молекул ---
    evaluator = MoleculeEvaluator(config.DIFFSBDD_CONFIG)
    smiles_list = evaluator.sdf_to_smiles(generated_sdf_path)
    
    evaluated_df = evaluator.evaluate_smiles_list(smiles_list)
    if evaluated_df.empty:
        logging.error("Оценка молекул не дала результатов. Завершение работы.")
        return

    # --- Шаг 3: Фильтрация и сохранение результатов ---
    top_molecules_df = evaluator.filter_and_score(evaluated_df)
    
    # Сохраняем полный датафрейм с оценками
    evaluated_df.to_csv(config.DIFFSBDD_CONFIG["OUTPUT_CSV"], index=False)
    logging.info(f"Полные результаты оценки сохранены в: {config.DIFFSBDD_CONFIG['OUTPUT_CSV']}")

    # --- Шаг 4: Визуализация лучших молекул ---
    if not top_molecules_df.empty:
        logging.info("Топ-10 лучших молекул по результатам оценки:")
        print(top_molecules_df.head(10).to_string())
        
        # Визуализация для отчета (необязательно, но полезно)
        mols = [Chem.MolFromSmiles(smi) for smi in top_molecules_df.head(10)['SMILES']]
        img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(200, 200))
        img_path = os.path.join(os.path.dirname(config.DIFFSBDD_CONFIG["OUTPUT_CSV"]), "top_molecules_grid.png")
        img.save(img_path)
        logging.info(f"Изображение с топ-молекулами сохранено в: {img_path}")
    else:
        logging.warning("После фильтрации не осталось молекул-кандидатов.")

    logging.info("Пайплайн сравнения с DiffSBDD успешно завершен.")


if __name__ == "__main__":
    main()
