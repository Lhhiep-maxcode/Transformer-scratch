from pathlib import Path
import re

def get_config():
    return {
        "data_path": ['/kaggle/input/machine-translation-en-vi/English_Vietnamese_1.csv'],
        "train_size": 0.05,   
        "val_size": 1 / 30000,     
        "batch_size": 64,
        "num_epochs": 8,
        "peak_lr": 7e-4,
        "warmup_steps": 4000,
        "seq_len": 102,
        "d_model": 512,
        "beam_size": 5,
        "preload_path": None,
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "tokenizer_file": "wordlevel_tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
        "random_seed": 42,
        # wandb log
        "wandb_key": "43f4be04f69239630c28c8a79e7eb42eea331825", 
        "wandb_project_name": "Transformer from scratch",
        "wandb_experiment_name": "Init experiment",
        "wandb_experiment_id": None,
    }

def get_weights_file_path(config, epoch: int):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

def latest_weights_file_path(config):
    model_folder = config['model_folder']
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    
    if len(weights_files) == 0:
        return None
    
    # Extract epoch number from filename using regex
    def extract_epoch(file):
        match = re.search(r'(\d+)', file.stem)  # Extract digits
        return int(match.group(1)) if match else -1  # Convert to int
    
    # Sort based on epoch number
    weights_files.sort(key=extract_epoch)

    return str(weights_files[-1])  # Return latest model