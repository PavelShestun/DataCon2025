# 3_validation_docking/src/data_preparer.py
import os
import pandas as pd
import logging

class DataPreparer:
    def __init__(self, config: dict):
        self.config = config

    def combine_and_deduplicate(self) -> str:
        """
        Объединяет несколько CSV файлов, добавляет метку модели, удаляет дубликаты
        по SMILES и сохраняет результат.
        
        Возвращает путь к объединенному файлу.
        """
        logging.info("Начало объединения и дедупликации входных датасетов...")
        
        dataframes = []
        for file_path in self.config["INPUT_CSV_FILES"]:
            try:
                # Определяем имя модели из пути
                model_name = os.path.basename(os.path.dirname(file_path))
                df = pd.read_csv(file_path)
                df['Model'] = model_name
                dataframes.append(df)
                logging.info(f"Загружен файл: {file_path}, добавлено {len(df)} строк с меткой '{model_name}'.")
            except FileNotFoundError:
                logging.warning(f"Файл не найден, пропускаем: {file_path}")
                continue
        
        if not dataframes:
            logging.error("Ни один из входных датасетов не был загружен. Завершение работы.")
            return None

        df_combined = pd.concat(dataframes, ignore_index=True)
        initial_rows = len(df_combined)
        
        # Фильтрация по уникальным SMILES, оставляем первую запись
        df_combined = df_combined.drop_duplicates(subset=['SMILES'], keep='first')
        final_rows = len(df_combined)
        
        removed_count = initial_rows - final_rows
        logging.info(f"Удалено {removed_count} дубликатов по SMILES. Осталось {final_rows} уникальных молекул.")
        
        # Сохранение объединенного датасета
        output_path = self.config["COMBINED_INPUT_CSV"]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_combined.to_csv(output_path, index=False)
        logging.info(f"Объединенный и очищенный датасет сохранен в: {output_path}")
        
        return output_path
