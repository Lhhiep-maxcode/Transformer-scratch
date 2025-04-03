import torch
import torch.nn as nn
import warnings
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from dataset import BilingualDataset, causal_mask
from torch.utils.data import Dataset, DataLoader, random_split
from model import build_transformer
from torch.utils.tensorboard import SummaryWriter
from config import get_config, get_weights_file_path, latest_weights_file_path
from tqdm import tqdm


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

    if config_seq_len < max_found_seq_len:
        raise Exception(f"Max founded sequence length is {max_found_seq_len}, but config seq_len is {config_seq_len}. Please increase the config seq_len.")
    
    train_ds = BilingualDataset(ds_raw_train, tokenizer_src, tokenizer_tgt, config['seq_len'])
    val_ds = BilingualDataset(ds_raw_valid, tokenizer_src, tokenizer_tgt, config['seq_len'])

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt
    

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(src_vocab_size=vocab_src_len, tgt_vocab_size=vocab_tgt_len,
                              src_seq_len=config['seq_len'], tgt_seq_len=config['seq_len'],
                              d_model=config['d_model'])
    return model


def train_model(config):
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
    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:04d}")
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

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

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