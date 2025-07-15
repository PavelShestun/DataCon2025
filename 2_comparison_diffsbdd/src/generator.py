# 2_comparison_diffsbdd/src/generator.py
import os
import logging
import subprocess
import requests

class DiffSBDDGenerator:
    """
    Класс для управления генерацией молекул с помощью DiffSBDD.
    """
    def __init__(self, config: dict):
        self.config = config
        self.pdb_path = os.path.join("data", f"{self.config['PDB_ID']}.pdb")
        # Путь к скрипту генерации, предполагается, что он будет лежать рядом
        self.script_path = "generate_ligands.py" 
        
    def _prepare_receptor(self):
        """Скачивает PDB файл, если он не существует."""
        os.makedirs(os.path.dirname(self.pdb_path), exist_ok=True)
        if os.path.exists(self.pdb_path):
            logging.info(f"PDB файл {self.config['PDB_ID']} уже существует.")
            return

        logging.info(f"Скачивание PDB файла {self.config['PDB_ID']}...")
        url = f"https://files.rcsb.org/download/{self.config['PDB_ID']}.pdb"
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(self.pdb_path, "wb") as f:
                f.write(response.content)
            logging.info(f"PDB файл успешно скачан в: {self.pdb_path}")
        except requests.exceptions.RequestException as e:
            logging.error(f"Ошибка скачивания PDB файла: {e}")
            raise
    
    def run_generation(self) -> str or None:
        """
        Запускает скрипт генерации DiffSBDD и возвращает путь к выходному SDF файлу.
        """
        self._prepare_receptor()
        
        # Убедимся, что директория для вывода существует
        os.makedirs(os.path.dirname(self.config["OUTPUT_SDF"]), exist_ok=True)
        
        # ВНИМАНИЕ: Предполагается, что скрипт `generate_ligands.py` и чекпоинт
        # находятся в папке `2_comparison_diffsbdd/`.
        # Для запуска этого кода вам нужно будет скачать репозиторий DiffSBDD
        # и положить `generate_ligands.py` в эту директорию.
        command = [
            "python", self.script_path,
            self.config["CHECKPOINT_FILE"],
            "--pdbfile", self.pdb_path,
            "--outfile", self.config["OUTPUT_SDF"],
            "--n_samples", str(self.config["N_SAMPLES"]),
            "--sanitize" # Важный флаг для получения корректных молекул
        ]

        logging.info("Запуск генерации молекул с помощью DiffSBDD...")
        logging.info(f"Команда: {' '.join(command)}")
        
        try:
            # Запускаем внешний процесс
            subprocess.run(command, check=True, capture_output=True, text=True)
            logging.info(f"Генерация завершена. Молекулы сохранены в {self.config['OUTPUT_SDF']}")
            return self.config["OUTPUT_SDF"]
        except FileNotFoundError:
            logging.error(f"Ошибка: скрипт '{self.script_path}' не найден. "
                          "Убедитесь, что он находится в директории '2_comparison_diffsbdd/'.")
            return None
        except subprocess.CalledProcessError as e:
            logging.error("Ошибка во время выполнения скрипта DiffSBDD.")
            logging.error(f"STDOUT: {e.stdout}")
            logging.error(f"STDERR: {e.stderr}")
            return None
