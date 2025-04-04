import torch
import torch.nn as nn
import warnings
import torchmetrics
import gc
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
    ds_raw_train = [data for data in ds_raw['train']]
    ds_raw_valid = [data for data in ds_raw['validation']]

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
    
    train_ds = BilingualDataset(ds_raw_train[:config['train_size']], tokenizer_src, tokenizer_tgt, config['seq_len'])
    val_ds = BilingualDataset(ds_raw_valid[:config['val_size']], tokenizer_src, tokenizer_tgt, config['seq_len'])

    print(f"Training size: {len(train_ds)} sentences")
    print(f"Validation size: {len(val_ds)} sentences")

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt
    

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(src_vocab_size=vocab_src_len, tgt_vocab_size=vocab_tgt_len,
                              src_seq_len=config['seq_len'], tgt_seq_len=config['seq_len'],
                              d_model=config['d_model'])
    return model


def beam_search(model, beam_size, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_id = tokenizer_tgt.token_to_id('[SOS]')
    eos_id = tokenizer_tgt.token_to_id('[EOS]')
    pad_id = tokenizer_tgt.token_to_id('[PAD]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(encoder_input, encoder_mask)
    # Initialize the decoder input with the sos token
    decoder_initial_input = torch.empty(1, 1).fill_(sos_id).type_as(encoder_input).to(device)

    # Create a candidate list
    # (generated text, score)
    candidates = [(decoder_initial_input, 1)]

    while True:

        # If a candidate has reached the maximum length, it means we have run the decoding for at least max_len iterations, so stop the search
        if any([cand.size(1) == max_len for cand, _ in candidates]):
            break
        
        new_candidates = []

        for candidate, score in candidates:

            # Do not expand candidates that have reached the eos token
            if candidate[0][-1].item() == eos_id:
                continue

            # Build the candidate's mask
            candidate_mask = causal_mask(candidate.size(1)).type_as(encoder_mask).to(device) # (1, length of sequence, length of sequence)
            out = model.decode(encoder_output, encoder_mask, candidate, candidate_mask)
            # get next token probabilities
            prob = model.project(out[:, -1])    # (batch, d_model)
            # get the top k candidates
            topk_prob, topk_idx = torch.topk(prob, beam_size, dim=-1)
            for i in range(beam_size):
                # for each of the top k candidates, get the token and its probability
                # hardcode for evaluation with batch size = 1
                token = topk_idx[0][i].unsqueeze(0).unsqueeze(0)    # (1, 1)   
                token_prob = topk_prob[0][i].item()
                new_candidate = torch.cat([candidate, token], dim=1)     # (1, length of generated sequence + 1)
                # We sum the log probabilities because the probabilities are in log space
                new_candidates.append((new_candidate, score + token_prob))

        # Sort the new candidates by their score
        candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)
        # Keep only the top k candidates
        candidates = candidates[:beam_size]

         # If all the candidates have reached the eos token, stop
        if all([cand[0][-1].item() == eos_id for cand, _ in candidates]):
            break

    # Return the best candidate
    return candidates[0][0].squeeze(0)  # (seq_len)


def greedy_search(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_id = tokenizer_tgt.token_to_id('[SOS]')
    eos_id = tokenizer_tgt.token_to_id('[EOS]')
    pad_id = tokenizer_tgt.token_to_id('[PAD]')

    encoder_output = model.encode(encoder_input, encoder_mask)
    decoder_input = torch.empty((1, 1)).fill_(sos_id).type_as(encoder_input).to(device) # (batch, 1)
    while True:
        if decoder_input.size(1) == max_len:
            break
        
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(encoder_mask).to(device) # (1, length of sequence, length of sequence)
        decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (batch, length of sequence, d_model)
        # print("Decoder out", decoder_output.shape)
        # get next token
        prob = model.project(decoder_output[:, -1]) # (batch, d_model)
        # print("Probability", prob.shape)
        _, next_word = torch.max(prob, dim=-1)
        # print("Next word", next_word)
        decoder_input = torch.cat(
            [decoder_input, torch.empty((1, 1)).type_as(decoder_input).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_id:
            break

    return decoder_input.squeeze(0)   # (seq_len)


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, config, num_examples=2):
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

    if writer:
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted_greedy, expected)
        writer.add_scalar('greedy search - validation cer', cer, global_step)
        writer.flush()

        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted_greedy, expected)
        writer.add_scalar('greedy search - validation wer', wer, global_step)
        writer.flush()

        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted_greedy, expected)
        writer.add_scalar('greedy search - validation bleu', bleu, global_step)
        writer.flush()

        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted_beam, expected)
        writer.add_scalar('beam search - validation cer', cer, global_step)
        writer.flush()

        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted_beam, expected)
        writer.add_scalar('beam search - validation wer', wer, global_step)
        writer.flush()

        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted_beam, expected)
        writer.add_scalar('beam search - validation bleu', bleu, global_step)
        writer.flush()


def train_model(config):
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
        if device == "cuda":
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

        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config["seq_len"], device, lambda msg: batch_iterator.write(msg), global_step, config, writer)

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