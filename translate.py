from pathlib import Path
from config import get_config, latest_weights_file_path 
from model import build_transformer
from tokenizers import Tokenizer
from datasets import load_dataset
from dataset import BilingualDataset
from train import beam_search, greedy_search
import torch
import sys

def translate(sentence: str):
    # Define the device, tokenizers, and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    config = get_config()
    tokenizer_src = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_src']))))
    tokenizer_tgt = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_tgt']))))
    model = build_transformer(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(), config["seq_len"], config['seq_len'], d_model=config['d_model']).to(device)

    # Load the pretrained weights
    model_filename = latest_weights_file_path(config)
    state = torch.load(model_filename)
    model.load_state_dict(state['model_state_dict'])

    # if the sentence is a number use it as an index to the test set
    label = ""
    if type(sentence) == int or sentence.isdigit():
        id = int(sentence)
        ds = load_dataset("kaitchup/opus-Vietnamese-to-English")
        ds_raw_train = [data for data in ds['train']]
        ds_raw_valid = [data for data in ds['validation']]
        ds = BilingualDataset(ds_raw_train + ds_raw_valid, tokenizer_src, tokenizer_tgt, config['seq_len'])
        sentence = ds[id]['src_text']
        label = ds[id]["tgt_text"]
    seq_len = config['seq_len']

    # translate the sentence
    model.eval()
    with torch.no_grad():
        # Precompute the encoder output and reuse it for every generation step
        source = tokenizer_src.encode(sentence)
        source = torch.cat([
            torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64), 
            torch.tensor(source.ids, dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[PAD]')] * (seq_len - len(source.ids) - 2), dtype=torch.int64)
        ], dim=0).to(device)
        source_mask = (source != tokenizer_src.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0).int().to(device)

        # Print the source sentence and target start prompt
        if label != "": print(f"{f'ID: ':>12}{id}") 
        print(f"{f'SOURCE: ':>12}{sentence}")
        if label != "": print(f"{f'TARGET: ':>12}{label}") 
        print(f"{f'PREDICTED: ':>12}", end='')

        model_out_greedy = greedy_search(model, source, source_mask, tokenizer_src, tokenizer_tgt, seq_len, device)
        model_out_beam = beam_search(model, config["beam_size"], source, source_mask, tokenizer_src, tokenizer_tgt, seq_len, device)

        model_out_text_greedy = tokenizer_tgt.decode(model_out_greedy.detach().cpu().numpy())
        model_out_text_beam = tokenizer_tgt.decode(model_out_beam.detach().cpu().numpy())

        print(f"{f'GREEDY SEARCH - PREDICTED: ':>12}{model_out_text_greedy}")
        print(f"{f'BEAM SEARCH - PREDICTED: ':>12}{model_out_text_beam}")

    return model_out_text_greedy, model_out_text_beam