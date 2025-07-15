# 3_validation_docking/main.py

import sys
import os
import logging

# Добавляем корневую директорию проекта в путь, чтобы можно было импортировать config
# Это делает приложение переносимым
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# Импортируем нашу логику из папки src
from src.ligand_preparer import generate_3d_conformers
from src.docking_runner import DockingRunner

def main():
    """
    Основной пайплайн для подготовки лигандов и запуска молекулярного докинга.
    """
    # Настройка логирования для вывода в консоль
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

    logging.info("--- ЗАПУСК ПАЙПЛАЙНА МОЛЕКУЛЯРНОГО ДОКИНГА ---")
    
    # Используем настройки из общего config файла
    docking_config = config.DOCKING_CONFIG

    # --- Шаг 1: Подготовка 3D структур лигандов ---
    generate_3d_conformers(
        csv_path=docking_config["INPUT_CSV"], 
        output_dir=docking_config["PREPARED_LIGANDS_DIR"]
    )

    # --- Шаг 2: Инициализация и подготовка для докинга ---
    runner = DockingRunner(docking_config)
    runner.prepare_receptor()
    box_params = runner.calculate_docking_box()

    # --- Шаг 3: Запуск докинга ---
    runner.run_docking_batch(box_params)

    logging.info("Пайплайн докинга успешно завершен.")
    logging.info(f"Результаты сохранены в: {docking_config['DOCKING_RESULTS_DIR']}")

if __name__ == "__main__":
    main()
