import torch
import torch.nn as nn
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

def get_all_sentences(dataset, lang):
    """
    dataset: dataset get from huggingface
    lang: 
        'vi' - Vietnamese
        'en' - English
    """
    for data in dataset:
        vi, en = data['text'].split("###>")
        vi = vi.strip()
        en = en.strip()
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
    
