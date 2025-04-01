import torch
import torch.nn as nn
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from dataset import BilingualDataset, causal_mask
from torch.utils.data import Dataset, DataLoader, random_split


from pathlib import Path

def text_split(data):
    """
    Example data: {'text': 'Nó cần có... ###>It needs a...'}
    """
    vi, en = data['text'].split("###>")
    vi = vi.strip()
    en = en.strip()
    return vi, en


def get_all_sentences(dataset, lang):
    """
    dataset: dataset get from huggingface
    lang: 
        'vi' - Vietnamese
        'en' - English
    """
    for data in dataset:
        vi, en = text_split(data)
        sentence = vi
        if lang == 'en':
            sentence = en
        yield sentence


def get_or_build_tokenizer(config, dataset, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(show_progess=True, special_tokens=['[UNK]', '[PAD]', '[SOS]', '[EOS]'], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(dataset, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    ds_raw = load_dataset("kaitchup/opus-Vietnamese-to-English")
    ds_raw_train = ds_raw['train']
    ds_raw_valid = ds_raw['validation']

    # get tokenizer
    tokenizer_src = get_or_build_tokenizer(config, ds_raw_train, 'vi')
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw_train, 'en')
    
    max_len_src = 0
    max_len_tgt = 0

    for data in ds_raw_train:
        vi, en = text_split(data)
        vi_ids = tokenizer_src.encode(vi).ids
        en_ids = tokenizer_tgt.encode(en).ids
        max_len_src = max(max_len_src, len(vi_ids))
        max_len_tgt = max(max_len_tgt, len(en_ids))

    for data in ds_raw_valid:
        vi, en = text_split(data)
        vi_ids = tokenizer_src.encode(vi).ids
        en_ids = tokenizer_tgt.encode(en).ids
        max_len_src = max(max_len_src, len(vi_ids))
        max_len_tgt = max(max_len_tgt, len(en_ids))
    
    config_seq_len = config['seq_len']
    max_found_seq_len = max(max_len_src, max_len_tgt)
    seq_len = max(config_seq_len, max_found_seq_len)

    if config_seq_len < max_found_seq_len:
        print(f"Override value provided for seq_len parameter to {seq_len}")
    
    train_ds = BilingualDataset(ds_raw_train, tokenizer_src, tokenizer_tgt, seq_len)
    val_ds = BilingualDataset(ds_raw_valid, tokenizer_src, tokenizer_tgt, seq_len)

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt
    
