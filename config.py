# config.py
import os

# --- 0. Общие Настройки Путей ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

# --- 1. Настройки для RL Модели (Приложение 1) ---
RL_CONFIG = {
    "MODELS_DIR": os.path.join(ROOT_DIR, "1_generative_rl_model", "models"),
    "OUTPUT_CSV": os.path.join(RESULTS_DIR, "1_rl_model_outputs", "final_molecules.csv"),
    "METRICS_CSV": os.path.join(RESULTS_DIR, "1_rl_model_outputs", "rl_metrics.csv"),
    # ... другие настройки RL, если нужно ...
}

# --- 2. Настройки для DiffSBDD (Приложение 2) ---
DIFFSBDD_CONFIG = {
    "PDB_ID": "4L7B",
    "CHECKPOINT_FILE": "checkpoints/crossdocked_fullatom_cond.ckpt", # Предполагается, что он будет в папке приложения
    "N_SAMPLES": 50,
    "OUTPUT_SDF": os.path.join(RESULTS_DIR, "2_diffsbdd_outputs", "diffsbdd_generated.sdf"),
    "OUTPUT_CSV": os.path.join(RESULTS_DIR, "2_diffsbdd_outputs", "diffsbdd_evaluated.csv"),
    # Модели-предикторы, которые создает Приложение 1
    "KEAP1_MODEL_PATH": os.path.join(RL_CONFIG["MODELS_DIR"], "keap1_activity_model.pkl"),
    "EGFR_MODEL_PATH": os.path.join(RL_CONFIG["MODELS_DIR"], "egfr_activity_model.pkl"),
    "IKKB_MODEL_PATH": os.path.join(RL_CONFIG["MODELS_DIR"], "ikkb_activity_model.pkl"),
}


# --- 3. Настройки для Докинга (Приложение 3) ---
DOCKING_CONFIG = {
    # ВАЖНО: Укажите здесь, какие CSV-файлы объединять и докировать
    "INPUT_CSV_FILES": [
        os.path.join(RESULTS_DIR, "1_rl_model_outputs", "final_molecules.csv"),
        os.path.join(RESULTS_DIR, "2_diffsbdd_outputs", "diffsbdd_evaluated.csv"),
    ],
    # Имя файла, куда будут сохранены объединенные и уникальные SMILES
    "COMBINED_INPUT_CSV": os.path.join(DATA_DIR, "smiles_inputs", "combined_for_docking.csv"),

    "PDB_ID": "4L7B",
    "RECEPTOR_DIR": os.path.join(DATA_DIR, "receptor"),
    "PREPARED_LIGANDS_DIR": os.path.join(RESULTS_DIR, "3_docking_outputs", "prepared_ligands"),
    "DOCKING_RESULTS_DIR": os.path.join(RESULTS_DIR, "3_docking_outputs", "docking_results"),
    "ANALYSIS_RESULTS_DIR": os.path.join(RESULTS_DIR, "3_docking_outputs", "analysis_plots"), # Папка для графиков
    
    # Настройки докинг-бокса
    "REF_LIGAND_CHAIN": "B",
    "REF_LIGAND_RES_NUM": "701",
    "REF_LIGAND_RES_NAME": "1VV",
    "BOX_PADDING": 10.0,

    # Настройки GNINA
    "GNINA_EXECUTABLE_PATH": "./gnina",
    "CNN_MODE": "fast",
}
