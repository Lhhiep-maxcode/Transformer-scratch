from pathlib import Path
import re

def get_config():
    return {
        "train_size": 100000,   # set -1 to get all
        "val_size": -1,     # set -1 to get all
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 1e-3,
        "seq_len": 150,
        "d_model": 512,
        "beam_size": 5,
        "lang_src": "vi",
        "lang_tgt": "en",
        "preload": "latest",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
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