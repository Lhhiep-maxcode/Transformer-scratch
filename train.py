import torch
import torch.nn as nn
import warnings
import gc
from torchmetrics.text import CharErrorRate, WordErrorRate, BLEUScore
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
        trainer = WordLevelTrainer(show_progress=True, special_tokens=['[UNK]', '[PAD]', '[SOS]', '[EOS]'], min_frequency=2)
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
    
    # Handle -1 for getting all data
    train_size = len(ds_raw_train) if config['train_size'] == -1 else min(config['train_size'], len(ds_raw_train))
    val_size = len(ds_raw_valid) if config['val_size'] == -1 else min(config['val_size'], len(ds_raw_valid))
    
    train_ds = BilingualDataset(ds_raw_train[:train_size], tokenizer_src, tokenizer_tgt, config['seq_len'])
    val_ds = BilingualDataset(ds_raw_valid[:val_size], tokenizer_src, tokenizer_tgt, config['seq_len'])

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


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total}")
    print(f"Trainable parameters: {trainable}")


def beam_search(model, beam_size, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_id = tokenizer_tgt.token_to_id('[SOS]')
    eos_id = tokenizer_tgt.token_to_id('[EOS]')
    pad_id = tokenizer_tgt.token_to_id('[PAD]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(encoder_input, encoder_mask)
    # Initialize the decoder input with the sos token
    decoder_initial_input = torch.empty(1, 1).fill_(sos_id).type_as(encoder_input).to(device)

    # Create a candidate list
    # (generated text, cumulative log probability score)
    candidates = [(decoder_initial_input, 0.0)]  # Start with log(1) = 0

    while True:

        # If a candidate has reached the maximum length, it means we have run the decoding for at least max_len iterations, so stop the search
        if any([cand.size(1) == max_len for cand, _ in candidates]):
            break
        
        new_candidates = []

        for candidate, score in candidates:

            # If candidate already reached EOS, keep it as-is without expanding
            if candidate[0][-1].item() == eos_id:
                new_candidates.append((candidate, score))
                continue

            # Build the candidate's mask
            candidate_mask = causal_mask(candidate.size(1)).type_as(encoder_mask).to(device) # (1, length of sequence, length of sequence)
            out = model.decode(encoder_output, encoder_mask, candidate, candidate_mask)
            # get next token probabilities (apply log_softmax to get log probabilities)
            prob = model.project(out[:, -1])    # (batch, vocab_size) - raw logits
            log_prob = torch.log_softmax(prob, dim=-1)  # Convert to log probabilities
            # get the top k candidates
            topk_log_prob, topk_idx = torch.topk(log_prob, beam_size, dim=-1)
            for i in range(beam_size):
                # for each of the top k candidates, get the token and its probability
                # hardcode for evaluation with batch size = 1
                token = topk_idx[0][i].unsqueeze(0).unsqueeze(0)    # (1, 1)   
                token_log_prob = topk_log_prob[0][i].item()
                new_candidate = torch.cat([candidate, token], dim=1)     # (1, length of generated sequence + 1)
                # Sum the log probabilities (equivalent to multiplying probabilities)
                new_candidates.append((new_candidate, score + token_log_prob))

        # Sort the new candidates by their score
        candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)
        # Keep only the top k candidates
        candidates = candidates[:beam_size]
        
        # If no candidates left (all reached EOS), break
        if len(candidates) == 0:
            break

         # If all the candidates have reached the eos token, stop
        if all([cand[0][-1].item() == eos_id for cand, _ in candidates]):
            break

    # Return the best candidate with length normalization
    # Normalize scores by length to avoid favoring shorter sequences
    best_candidate = max(candidates, key=lambda x: x[1] / (x[0].size(1) ** 0.6))
    return best_candidate[0].squeeze(0) 


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
        # get next token probabilities
        logits = model.project(decoder_output[:, -1]) # (batch, vocab_size) - raw logits
        # Apply softmax to get probability distribution
        probs = torch.softmax(logits, dim=-1)
        # Select token with highest probability
        _, next_word = torch.max(probs, dim=-1)
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
        # Greedy search metrics
        metric = CharErrorRate()
        cer_greedy = metric(predicted_greedy, expected)
        writer.add_scalar('greedy search - validation cer', cer_greedy, global_step)
        writer.flush()

        metric = WordErrorRate()
        wer_greedy = metric(predicted_greedy, expected)
        writer.add_scalar('greedy search - validation wer', wer_greedy, global_step)
        writer.flush()

        # BLEU requires tokenized text - split into words
        metric = BLEUScore()
        predicted_greedy_tokenized = [pred.split() for pred in predicted_greedy]
        expected_tokenized = [[ref.split()] for ref in expected]  
        bleu_greedy = metric(predicted_greedy_tokenized, expected_tokenized)
        writer.add_scalar('greedy search - validation bleu', bleu_greedy, global_step)
        writer.flush()

        # Beam search metrics
        metric = CharErrorRate()
        cer_beam = metric(predicted_beam, expected)
        writer.add_scalar('beam search - validation cer', cer_beam, global_step)
        writer.flush()

        metric = WordErrorRate()
        wer_beam = metric(predicted_beam, expected)
        writer.add_scalar('beam search - validation wer', wer_beam, global_step)
        writer.flush()

        metric = BLEUScore()
        predicted_beam_tokenized = [pred.split() for pred in predicted_beam]
        bleu_beam = metric(predicted_beam_tokenized, expected_tokenized)
        writer.add_scalar('beam search - validation bleu', bleu_beam, global_step)
        writer.flush()

        print('-----Evaluation-----')
        print('| greedy search - validation cer', cer_greedy)
        print('| greedy search - validation wer', wer_greedy)
        print('| greedy search - validation bleu', bleu_greedy)
        print('| beam search - validation cer', cer_beam)
        print('| beam search - validation wer', wer_beam)
        print('| beam search - validation bleu', bleu_beam)


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
    count_parameters(model)
    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    # AdamW with weight decay instead of Adam
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['lr'], 
        betas=(0.9, 0.98),  
        eps=1e-9,
        weight_decay=config.get('weight_decay', 0.01)
    )

    # Learning rate scheduler with warmup
    def get_lr_scheduler(optimizer, warmup_steps, base_lr):
        """Simple linear warmup scheduler"""
        def lr_lambda(step):
            if step == 0:
                return 0.0
            if step < warmup_steps:
                # Linear warmup
                return float(step) / float(max(1, warmup_steps))
            # Constant LR after warmup
            return 1.0
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Choose scheduler based on config
    if config.get('use_cosine_scheduler', False):
        total_steps = len(train_dataloader) * config['num_epochs']
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    else:
        scheduler = get_lr_scheduler(optimizer, config.get('warmup_steps', 4000), config['lr'])

    # Mixed precision training
    scaler = torch.amp.GradScaler('cuda') if config.get('mixed_precision', True) and device.type == 'cuda' else None

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
        if 'scheduler_state_dict' in state:
            scheduler.load_state_dict(state['scheduler_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_src.token_to_id('[PAD]'), 
        label_smoothing=config.get('label_smoothing', 0.1)
    ).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        if device == "cuda":
            torch.cuda.empty_cache()

        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:04d}")
        losses = []
        gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
        
        for batch_idx, batch in enumerate(batch_iterator):
            encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

            # Mixed precision training
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    # Run the tensors through the encoder, decoder and the projection layer
                    encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
                    decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
                    proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

                    # Compare the output with the label
                    label = batch['label'].to(device) # (B, seq_len)

                    # Compute the loss using cross entropy
                    loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
                    # Scale loss for gradient accumulation
                    loss = loss / gradient_accumulation_steps
                
                # Backpropagate with scaled gradients
                scaler.scale(loss).backward()
            else:
                # Standard training without mixed precision
                encoder_output = model.encode(encoder_input, encoder_mask)
                decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
                proj_output = model.project(decoder_output)
                label = batch['label'].to(device)
                loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
                loss = loss / gradient_accumulation_steps
                loss.backward()
            
            # Update progress bar with actual loss (not scaled)
            actual_loss = loss.item() * gradient_accumulation_steps
            batch_iterator.set_postfix({"loss": f"{actual_loss:6.3f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})
            losses.append(actual_loss)

            # Update weights after accumulating gradients
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if scaler is not None:
                    # Gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('gradient_clip', 1.0))
                    # Update weights
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Gradient clipping for non-mixed precision
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('gradient_clip', 1.0))
                    optimizer.step()
                
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                
                # Log the loss and learning rate
                writer.add_scalar('train loss', actual_loss, global_step)
                writer.add_scalar('learning rate', scheduler.get_last_lr()[0], global_step)
                writer.flush()
                
                global_step += 1

        print('| Average Training-Loss : {:.4f}'.format(torch.mean(torch.tensor(losses))))
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config["seq_len"], device, lambda msg: batch_iterator.write(msg), global_step, writer, config)

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'global_step': global_step
        }, model_filename)
    

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)