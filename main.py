# main.py
import os
import logging
import pandas as pd
from tqdm.auto import tqdm
from collections import deque
import random
import joblib

from src.config import *
from src.data_loader import get_all_datasets, load_moses_data
from src.model_trainer import prepare_models
from src.molecule_generator import MoleculeGeneratorRNN
from src.property_calculator import (
    calculate_advanced_properties_parallel,
    calculate_multi_objective_reward,
    filter_molecules
)
from src.utils import pareto_frontier, calculate_metrics, augment_smiles

# Настройка логирования
logging.basicConfig(level=logging.INFO, filename='molecule_generation.log', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_generation_pipeline():
    """
    Основной пайплайн для генерации молекул с использованием RL.
    """
    # Создание каталогов для моделей, если они не существуют
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    logger.info(f"Запуск пайплайна на устройстве: {DEVICE.upper()}")
    print(f"[*] Запуск пайплайна на устройстве: {DEVICE.upper()}")

    # --- Шаг 1: Подготовка моделей-предикторов ---
    keap1_clean, egfr_clean, ikkb_clean = get_all_datasets()
    keap1_model, egfr_model, ikkb_model = prepare_models(
        keap1_clean, egfr_clean, ikkb_clean
    )

    # --- Шаг 2: Загрузка и фильтрация данных для обучения генератора ---
    logger.info("Загрузка набора данных MOSES...")
    moses_smiles = load_moses_data()
    if not moses_smiles:
        logger.error("Не удалось загрузить MOSES, выход.")
        return

    logger.info("Фильтрация молекул по свойствам...")
    drug_like_smiles = [smi for smi in tqdm(moses_smiles, desc="Фильтрация молекул") if filter_molecules(smi)]
    
    pretrain_smiles = drug_like_smiles[:20000]
    augmented_smiles = [aug_smi for smi in tqdm(pretrain_smiles, desc="Аугментация SMILES") for aug_smi in augment_smiles(smi)]
    
    train_data, val_data = augmented_smiles[:int(0.9 * len(augmented_smiles))], augmented_smiles[int(0.9 * len(augmented_smiles)):]

    # --- Шаг 3: Предобучение RNN-генератора ---
    logger.info("Инициализация и предобучение RNN...")
    generator = MoleculeGeneratorRNN(all_smiles_for_vocab=pretrain_smiles, max_len=MAX_LEN_SELFIE, device=DEVICE)
    generator.pretrain(train_data, val_data, epochs=PRETRAIN_EPOCHS, batch_size=BATCH_SIZE)
    generator.save_model(GENERATOR_PRETRAINED_PATH)

    # --- Шаг 4: Цикл обучения с подкреплением (RL) ---
    logger.info("Начало цикла обучения с подкреплением...")
    experience_buffer = deque(maxlen=256)
    all_generated_molecules = {}
    metrics_log = []

    for i in range(RL_ITERATIONS):
        # Генерация, мутация и выбор молекул для обучения
        newly_sampled = generator.sample(num_samples=BATCH_SIZE, temperature=1.2)
        mutated_smiles = [generator.mutate_selfies(smi) for smi in newly_sampled]
        best_from_exp = [m['smiles'] for m in sorted(list(experience_buffer), key=lambda x: x['score'], reverse=True)[:BATCH_SIZE // 2]]
        
        training_smiles_set = set(newly_sampled + mutated_smiles + best_from_exp)
        
        # Расчет свойств и вознаграждений
        props_list = calculate_advanced_properties_parallel(training_smiles_set, keap1_model, egfr_model, ikkb_model)
        rewards = [calculate_multi_objective_reward(props) for props in props_list]
        smiles_for_update = [props['SMILES'] for props in props_list]

        # Обновление буфера и обучение генератора
        for smi, props, reward in zip(smiles_for_update, props_list, rewards):
            if smi not in all_generated_molecules:
                all_generated_molecules[smi] = {'props': props, 'score': reward}
                if reward > 0.1:
                    experience_buffer.append({'smiles': smi, 'score': reward})
        
        if smiles_for_update:
            loss = generator.train_step(smiles_for_update, rewards)
            # Логирование метрик
            # ... (код для логирования метрик, как в оригинале)

    generator.save_model(GENERATOR_FINAL_PATH)

    # --- Шаг 5: Фильтрация Парето и вывод результатов ---
    logger.info("Применение фильтрации Парето...")
    final_df = pd.DataFrame([v['props'] for v in all_generated_molecules.values()])
    final_df['Score'] = [v['score'] for v in all_generated_molecules.values()]
    pareto_df = pareto_frontier(final_df)

    logger.info(f"Найдено {len(pareto_df)} молекул на границе Парето.")
    print("\n--- Топ 5 Сгенерированных Молекул ---")
    print(pareto_df.head(N_MOLECULES_TO_FIND))

if __name__ == '__main__':
    run_generation_pipeline()
