import torch
import torch.nn as nn
import warnings
import gc
import torch.nn.functional as F
import wandb
import pandas as pd
import random
from torchmetrics.text import CharErrorRate, WordErrorRate, BLEUScore
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from dataset import BilingualDataset, causal_mask
from torch.utils.data import Dataset, DataLoader, random_split
from model import build_transformer
from config import get_config, get_weights_file_path, latest_weights_file_path
from tqdm import tqdm
from infer import greedy_search, beam_search
from pathlib import Path

import random
import numpy as np
import torch
import os

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def get_or_build_tokenizer(config, sentences, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(show_progress=True, special_tokens=['[UNK]', '[PAD]', '[SOS]', '[EOS]'], min_frequency=2)
        tokenizer.train_from_iterator(sentences, trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def shuffle_parallel_lists(en, vi, seed=42):
    assert len(en) == len(vi), "Lists must have the same length"

    paired = list(zip(en, vi))
    random.Random(seed).shuffle(paired)

    en_shuffled, vi_shuffled = zip(*paired)
    return list(en_shuffled), list(vi_shuffled)


def get_ds(config):
    print('='*10, 'Data Preparation', '='*10)
    en_list = []
    vi_list = []
    for path in config['data_path']:
        df = pd.read_csv(path)
        en_list.extend(df['English'].to_list())
        vi_list.extend(df['Vietnamese'].to_list())
    
    en_list, vi_list = shuffle_parallel_lists(en_list, vi_list, seed=config['random_seed'])

    # get tokenizer
    tokenizer_tgt = get_or_build_tokenizer(config, vi_list, 'vi')
    tokenizer_src = get_or_build_tokenizer(config, en_list, 'en')
    
    total_data = len(en_list)

    # filter out sentences' length > config['seq_len']
    filtered_en_list = []
    filtered_vi_list = []
    for en, vi in zip(en_list, vi_list):
        if len(tokenizer_src.encode(en).ids) > config['seq_len']:
            continue
        if len(tokenizer_tgt.encode(vi).ids) > config['seq_len']:
            continue

        filtered_en_list.append(en)
        filtered_vi_list.append(vi)
    
    en_list = filtered_en_list
    vi_list = filtered_vi_list

    print(f"Filter and get {len(en_list)} / {total_data} sentences")
    print("Data Example:")
    count = 0
    for i in range(1, 6):
        print(f"English:     ", en_list[i])
        print(f"Vietnamese:  ", vi_list[i])
        print()
        print(f"English:     ", en_list[-i])
        print(f"Vietnamese:  ", vi_list[-i])
        print()
        count += 1
        if count > 5:
            break

    
    # Train, valid split
    train_en = en_list[:int(config['train_size'] * len(en_list))]
    train_vi = vi_list[:int(config['train_size'] * len(en_list))]

    val_en = en_list[int(config['train_size'] * len(en_list)):int((config['train_size'] + config['val_size']) * len(en_list))]
    val_vi = vi_list[int(config['train_size'] * len(en_list)):int((config['train_size'] + config['val_size']) * len(en_list))]

    train_ds = BilingualDataset(train_en, train_vi, tokenizer_src, tokenizer_tgt, config['seq_len'])
    val_ds = BilingualDataset(val_en, val_vi, tokenizer_src, tokenizer_tgt, config['seq_len'])

    print(f"Training size: {len(train_ds)} sentences")
    print(f"Validation size: {len(val_ds)} sentences")

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False)

    print('='*30)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt
    

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(src_vocab_size=vocab_src_len, tgt_vocab_size=vocab_tgt_len,
                              src_seq_len=config['seq_len'], tgt_seq_len=config['seq_len'],
                              d_model=config['d_model'])
    return model


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total}")
    print(f"Trainable parameters: {trainable}")


def log_validation_results(
    source_texts, expected, predicted_greedy, predicted_beam, 
    log_path="validation_log.txt", epoch=None
):
    # Open file in append mode
    with open(log_path, "a", encoding="utf-8") as f:

        # Optional: write epoch header
        if epoch is not None:
            f.write(f"\n\n====================== EPOCH {epoch} ======================\n\n")

        # Write each sample
        for src, tgt, greedy, beam in zip(source_texts, expected, predicted_greedy, predicted_beam):
            f.write("SRC:      " + src + "\n")
            f.write("TARGET:   " + tgt + "\n")
            f.write("GREEDY:   " + greedy + "\n")
            f.write("BEAM:     " + beam + "\n")
            f.write("-----------------------------------------------------------\n")


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, epoch, global_step, wandb_run, config, num_examples=2):
    model.eval()
    count = 0
    
    source_texts = []
    expected = []
    predicted_greedy = []
    predicted_beam = []

    console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (batch, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (batch, 1, 1, seq_len) => (batch, heads, seq_len, seq_len)

            assert encoder_input.size(0) == 1, "Batch size must be 1"

            model_out_greedy = greedy_search(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)
            model_out_beam = beam_search(model, config['beam_size'], encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text_greedy = tokenizer_tgt.decode(model_out_greedy.detach().cpu().numpy())
            model_out_text_beam = tokenizer_tgt.decode(model_out_beam.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted_greedy.append(model_out_text_greedy)
            predicted_beam.append(model_out_text_beam)

            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED GREEDY: ':>12}{model_out_text_greedy}")
            print_msg(f"{f'PREDICTED BEAM: ':>12}{model_out_text_beam}")

            if count == num_examples:
                print_msg('-'*console_width)
                break

    log_validation_results(
        source_texts, expected, predicted_greedy, predicted_beam, 
        log_path="validation_log.txt", epoch=epoch
    )

    metric = CharErrorRate()
    g_cer = metric(predicted_greedy, expected)

    metric = WordErrorRate()
    g_wer = metric(predicted_greedy, expected)

    metric = BLEUScore()
    g_bleu = metric(predicted_greedy, expected)

    metric = CharErrorRate()
    b_cer = metric(predicted_beam, expected)

    metric = WordErrorRate()
    b_wer = metric(predicted_beam, expected)

    metric = BLEUScore()
    b_bleu = metric(predicted_beam, expected)

    if wandb_run:
        wandb_run.log({
            'greedy search - validation cer': g_cer,
            'greedy search - validation wer': g_wer,
            'greedy search - validation bleu': g_bleu,
            'beam search - validation cer': b_cer,
            'beam search - validation wer': b_wer,
            'beam search - validation bleu': b_bleu,
        })

    print('-----Evaluation-----')
    print('| greedy search - validation cer', g_cer)
    print('| greedy search - validation wer', g_wer)
    print('| greedy search - validation bleu', g_bleu)
    print('| beam search - validation cer', b_cer)
    print('| beam search - validation wer', b_wer)
    print('| beam search - validation bleu', b_bleu)


def train_model(config):
    set_seed(config['random_seed'])
    gc.collect()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif (device == 'mps'):
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")
        print("      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
        print("      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")
    device = torch.device(device)


    Path(f"{config['model_folder']}").mkdir(parents=True, exist_ok=True)
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    count_parameters(model)

    if config['wandb_key'] is not None:
        wandb.login(key=config['wandb_key'])
        run = wandb.init(
            project=config['wandb_project_name'],  # Specify your project
            name=config['wandb_experiment_name'],
            id=config['wandb_experiment_id'],
            resume=("must" if config['wandb_experiment_id'] else None),
            config={                        
                # Track hyperparameters and metadata
                "train_size": config['train_size'],
                "epochs": config['num_epochs'], 
                "batch_size": config['batch_size'],
                "lr": config['lr'],
                "max_seq_len": config['seq_len'],
                "hidden_dim": config['d_model'],
                "beam_size": config['beam_size'],      
            },
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload_path = config['preload_path']
    if preload_path:
        print(f'Preloading model {preload_path}')
        state = torch.load(preload_path)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        if device == "cuda":
            torch.cuda.empty_cache()

        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:04d}")
        losses = []
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

            # Compare the output with the label
            label = batch['label'].to(device) # (B, seq_len)

            # Compute the loss using a simple cross entropy
            # loss((B * seq_len, vocab_size), (B * seq_len))
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            losses.append(loss.item())

            # Log the loss
            if global_step % 10 == 0 and config['wandb_key'] is not None:
                run.log({'Train loss (10 steps)': loss.item()})

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        avg_loss = torch.mean(torch.tensor(losses))
        print('| Average Training-Loss : {:.4f}'.format(avg_loss))
        if config['wandb_key'] is not None:
            run.log({'Train loss (epoch)': avg_loss})
            run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config["seq_len"], device, lambda msg: batch_iterator.write(msg), epoch, global_step, run, config)
        else:
            run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config["seq_len"], device, lambda msg: batch_iterator.write(msg), epoch, global_step, None, config)

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)
    

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
