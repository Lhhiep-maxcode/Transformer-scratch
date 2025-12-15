import torch
import torch.nn as nn
import warnings
import gc
import torch.nn.functional as F
import wandb
import pandas as pd
import random
import os
import numpy as np
import torch.distributed as dist
from comet import download_model, load_from_checkpoint
from torch.utils.data.distributed import DistributedSampler
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from dataset import BilingualDataset, causal_mask, DistributedLengthBasedCurriculumBatchSampler
from torch.utils.data import Dataset, DataLoader, random_split
from model import build_transformer
from config import get_config, get_weights_file_path, latest_weights_file_path
from tqdm import tqdm
from infer import greedy_search, beam_search
from pathlib import Path


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
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], 
            min_frequency=2, 
            vocab_size=30000,
            show_progress=True
        )
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

def compute_level(train_ds, tokenizer_src, tokenizer_tgt, type_metric="LENGTHS"):
    if type_metric == "LENGTHS":
        lengths = [len(item) for item in train_ds]
        return lengths
    elif type_metric == "TF_IDF":
        res = []
        vocab_src = tokenizer_src.get_vocab()
        vocab_tgt = tokenizer_tgt.get_vocab()
        for i in range(len(train_ds)):
            count = 0
            src_txt = train_ds[i]['src_text']
            tgt_txt = train_ds[i]['tgt_text']
            for c in src_txt:
                if c in vocab_src:
                    count += vocab_src[c]

            for c in tgt_txt:
                if c in vocab_tgt:
                    count += vocab_tgt[c]
            
            res.append(count // (len(src_txt) + len(tgt_txt)))
        print("TF-IDF levels computed", res)
        return res


def get_ds(config, ddp_enabled):
    # Only print on Rank 0
    if not ddp_enabled or (ddp_enabled and int(os.environ.get("LOCAL_RANK", 0)) == 0):
        print('='*10, 'Data Preparation', '='*10)
        
    en_list = []
    vi_list = []
    for path in config['train_path']:
        df = pd.read_csv(path)
        en_list.extend(df['English'].to_list())
        vi_list.extend(df['Vietnamese'].to_list())
    
    en_list, vi_list = shuffle_parallel_lists(en_list, vi_list, seed=config['random_seed'])

    # get tokenizer
    if ddp_enabled:
        local_rank = int(os.environ["LOCAL_RANK"])
        # If Rank 0, build the tokenizer
        if local_rank == 0:
            get_or_build_tokenizer(config, vi_list, 'vi')
            get_or_build_tokenizer(config, en_list, 'en')
        
        # Everyone waits here until Rank 0 is done
        dist.barrier()
        
        # Now everyone loads it safely
        tokenizer_tgt = get_or_build_tokenizer(config, vi_list, 'vi')
        tokenizer_src = get_or_build_tokenizer(config, en_list, 'en')
    else:
        tokenizer_tgt = get_or_build_tokenizer(config, vi_list, 'vi')
        tokenizer_src = get_or_build_tokenizer(config, en_list, 'en')
    
    total_data = len(en_list)

    # filter out sentences' length > config['train_seq_len']
    filtered_en_list = []
    filtered_vi_list = []
    for en, vi in zip(en_list, vi_list):
        enc_input_tokens = tokenizer_src.encode(en).ids
        dec_input_tokens = tokenizer_tgt.encode(vi).ids

        if ((config['train_seq_len'] - len(enc_input_tokens) - 2 < 0) or
            (config['train_seq_len'] - len(dec_input_tokens) - 1 < 0)):
            continue

        filtered_en_list.append(en)
        filtered_vi_list.append(vi)
    
    en_list = filtered_en_list
    vi_list = filtered_vi_list

    if not ddp_enabled or (ddp_enabled and int(os.environ.get("LOCAL_RANK", 0)) == 0):
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

    test_en = []
    test_vi = []
    for path in config['test_path']:
        df = pd.read_csv(path)
        test_en.extend(df['English'].to_list())
        test_vi.extend(df['Vietnamese'].to_list())

    filtered_en_list = []
    filtered_vi_list = []
    for en, vi in zip(test_en, test_vi):
        enc_input_tokens = tokenizer_src.encode(en).ids
        dec_input_tokens = tokenizer_tgt.encode(vi).ids

        if ((config['test_seq_len'] - len(enc_input_tokens) - 2 < 0) or
            (config['test_seq_len'] - len(dec_input_tokens) - 1 < 0)):
            continue

        filtered_en_list.append(en)
        filtered_vi_list.append(vi)
        
    test_en = filtered_en_list
    test_vi = filtered_vi_list

    train_ds = BilingualDataset(train_en, train_vi, tokenizer_src, tokenizer_tgt, config['train_seq_len'])
    val_ds = BilingualDataset(val_en, val_vi, tokenizer_src, tokenizer_tgt, config['train_seq_len'])
    test_ds = BilingualDataset(test_en, test_vi, tokenizer_src, tokenizer_tgt, config['test_seq_len'])

    print(f"Training size: {len(train_ds)} sentences")
    print(f"Validation size: {len(val_ds)} sentences")
    print(f"Test size: {len(test_ds)} sentences")
    print(f"Sample: {train_ds[0]}")

    
    levels = compute_level(train_ds, tokenizer_src, tokenizer_tgt, type_metric="LENGTHS") # Precompute lengths
    
    total_training_steps = config['num_epochs'] * (len(train_ds) // config['batch_size_base'])
    batch_size = config['batch_size_base']
    
    sampler = DistributedLengthBasedCurriculumBatchSampler(
        levels, batch_size, total_steps=total_training_steps, ddp_enabled=ddp_enabled
    )

    train_dataloader = DataLoader(train_ds, batch_sampler=sampler, num_workers=2, pin_memory=True,)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False)
    print('='*30)

    return train_dataloader, val_dataloader, test_dataloader, tokenizer_src, tokenizer_tgt, sampler
    

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(src_vocab_size=vocab_src_len, tgt_vocab_size=vocab_tgt_len,
                              src_seq_len=config['train_seq_len'], tgt_seq_len=config['train_seq_len'],
                              d_model=config['d_model'])
    return model


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total}")
    print(f"Trainable parameters: {trainable}")

def load_comet_model(device):
    model_path = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(model_path)
    comet_model.to(device)
    comet_model.eval()
    return comet_model


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
            f.write("\n")


def run_validation(
    model, validation_ds, 
    tokenizer_src, tokenizer_tgt, 
    max_len, device, epoch, 
    wandb_run, config, comet_model, num_examples=100
):
    model.eval()
    count = 0
    
    source_texts = []
    expected = []
    predicted_greedy = []
    predicted_beam = []

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (batch, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (batch, 1, 1, seq_len) => (batch, heads, seq_len, seq_len)

            assert encoder_input.size(0) == 1, "Batch size must be 1"

            if hasattr(model, 'module'):
                model_out_greedy = greedy_search(model.module, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)
                model_out_beam = beam_search(model.module, config['beam_size'], encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)
            else:
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

            if count == num_examples:
                break

    log_validation_results(
        source_texts, expected, predicted_greedy, predicted_beam, 
        log_path="validation_log.txt", epoch=epoch
    )

    comet_data = [
        {
            "src": s,
            "mt": h,
            "ref": r
        }
        for s, h, r in zip(source_texts, predicted_beam, expected)
    ]

    with torch.no_grad():
        comet_output = comet_model.predict(
            comet_data,
            batch_size=8,
            gpus=1 if device.type == "cuda" else 0
        )

    comet_score = sum(comet_output["scores"]) / len(comet_output["scores"])

    # ---- logging ----
    if wandb_run:
        wandb_run.log({
            "validation/COMET-DA": comet_score,
        })

    print("----- Validation -----")
    print(f"| COMET-DA score: {comet_score:.4f}")


def transformer_lr_lambda(step, d_model, warmup_steps, peak_lr):
    """
    Paper formula:
    lr = d_model^{-0.5} * min(step^{-0.5}, step * warmup^{-1.5})
    """
    step = max(step, 1)
    return peak_lr * (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5)) / ((warmup_steps ** -0.5) * (d_model ** -0.5))

import torch

def switchout(trg, vocab_size, tau=0.1, pad_idx=0, sos_idx=1, eos_idx=2):
    """
    Apply SwitchOut on batch target.
    
    Args:
        trg (Tensor): Batch target [batch_size, seq_len]
        vocab_size (int): size of vocabulary
        tau (float): statistical parameter controlling the noise level
        pad_idx, sos_idx, eos_idx: index of special tokens to be protected
    
    Returns:
        Tensor: target after applying SwitchOut
    """
    
    # 1. Tạo mask xác định vị trí sẽ bị thay thế
    mask = (torch.rand(trg.shape, device=trg.device) < tau)
    
    # 2. BẢO VỆ CÁC TOKEN ĐẶC BIỆT (Rất quan trọng)
    special_tokens_mask = (trg == pad_idx) | (trg == sos_idx) | (trg == eos_idx)
    
    # Loại bỏ các vị trí đặc biệt khỏi mask thay thế
    mask = mask & (~special_tokens_mask)
    
    # 3. Tạo tensor ngẫu nhiên chứa các từ mới
    random_words = torch.randint(0, vocab_size, trg.shape, device=trg.device)
    
    # 4. Tráo đổi
    # Tại đâu mask là True -> lấy random_words
    # Tại đâu mask là False -> giữ nguyên trg gốc
    corrupted_trg = torch.where(mask, random_words, trg)
    
    return corrupted_trg

def get_tau_schedule(current_epoch, total_epochs, min_tau=0.05, max_tau=0.2):
    """
    Tính toán giá trị tau tăng dần tuyến tính theo số epoch.
    
    Args:
        current_epoch (int): Epoch hiện tại (bắt đầu từ 0)
        total_epochs (int): Tổng số epoch định train
        min_tau (float): Giá trị tau khởi điểm (khi epoch = 0)
        max_tau (float): Giá trị tau kết thúc (khi epoch = last)
        
    Returns:
        float: Giá trị tau cho epoch hiện tại
    """
    # Đảm bảo không chia cho 0
    if total_epochs <= 1:
        return max_tau
        
    # Tính tỷ lệ tiến độ (từ 0.0 đến 1.0)
    progress = current_epoch / (total_epochs - 1)
    
    # Công thức nội suy tuyến tính: y = y1 + (y2 - y1) * x
    current_tau = min_tau + (max_tau - min_tau) * progress
    
    return current_tau

def train_model(config):
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        ddp_enabled = True
    else:
        local_rank = 0
        ddp_enabled = False

    set_seed(config['random_seed'])

    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")

    comet_model = None
    if local_rank == 0:
        comet_model = load_comet_model(device)

    if local_rank == 0 and config['wandb_key'] is not None:
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
                "max_seq_len": config['train_seq_len'],
                "batch_size": config['batch_size_max'],
                "hidden_dim": config['d_model'],
                "beam_size": config['beam_size'],      
            },
        )
    else:
        run = None

    Path(f"{config['model_folder']}").mkdir(parents=True, exist_ok=True)
    train_dataloader, val_dataloader, test_dataloader, tokenizer_src, tokenizer_tgt, sampler = get_ds(config, ddp_enabled)

    model = get_model(
        config,
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size()
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1.0,                # <-- dummy base lr, real lr comes from scheduler
        betas=(0.9, 0.98),     # <-- paper values
        eps=1e-9,
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: transformer_lr_lambda(
            step,
            d_model=config['d_model'],
            warmup_steps=config['warmup_steps'],
            peak_lr=config['peak_lr']
        )
    )

    # Load previous training state
    initial_epoch = 0
    global_step = 0
    preload_path = config['preload_path']
    if preload_path:
        if local_rank == 0: print(f'Preloading model {preload_path}')
        state = torch.load(preload_path, map_location=f"cuda:{local_rank}")
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        scheduler.load_state_dict(state['scheduler_state_dict'])
        initial_epoch = state['epoch'] + 1
        global_step = state['global_step']
    else:
        if local_rank == 0: print('No model to preload, starting from scratch')

    if ddp_enabled:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank
        )
    
    if local_rank == 0: count_parameters(model)

    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)
    number_of_iteration = config['batch_size_max'] // config['batch_size_base']

    pad_idx = tokenizer_tgt.token_to_id('[PAD]')     
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')     
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')     
    tgt_vocab_size = tokenizer_tgt.get_vocab_size() 
    if config['test_only'] == False:
        for epoch in range(initial_epoch, config['num_epochs']):
            if hasattr(train_dataloader.batch_sampler, 'set_epoch'):
                train_dataloader.batch_sampler.set_epoch(epoch)
            model.train()
            if local_rank == 0:
                batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:04d}")
            else:
                batch_iterator = train_dataloader

            losses = []
            
            optimizer.zero_grad(set_to_none=True)

            for i, batch in enumerate(batch_iterator):
                encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
                decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
                encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
                decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)
                label = batch['label'].to(device) # (B, seq_len)

                # Apply SwitchOut data augmentation
                get_tau = get_tau_schedule(epoch, config['num_epochs'], min_tau=config['min_tau'], max_tau=config['max_tau'])
                decoder_input = switchout(
                    decoder_input, 
                    vocab_size=tgt_vocab_size, 
                    tau=get_tau,
                    pad_idx=pad_idx, 
                    sos_idx=sos_idx, 
                    eos_idx=eos_idx
                )
                
                # ===== Forward =====
                # Autocast handles the casting (FP32 -> FP16 -> FP32) dynamically
                with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                    proj_output = model(encoder_input, encoder_mask, decoder_input, decoder_mask)

                    # Compute the loss using a simple cross entropy
                    # loss((B * seq_len, vocab_size), (B * seq_len))
                    loss = loss_fn(
                        proj_output.view(-1, tokenizer_tgt.get_vocab_size()), 
                        label.view(-1)
                    )
                # ===== Backprop =====

                scaler.scale(loss).backward()
                if (i + 1) % number_of_iteration == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                    scheduler.step()
                    sampler.step()
                global_step += 1
                losses.append(loss.item())
                
                if local_rank == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    batch_iterator.set_postfix({
                        "loss": f"{loss.item():6.3f}",
                        "lr": f"{current_lr:.2e}"
                    })

                    if global_step % 10 == 0 and config['wandb_key'] is not None:
                        run.log({'Train loss (10 steps)': loss.item(), 'lr(10 steps)': current_lr})

            avg_loss = torch.mean(torch.tensor(losses))
            print('| Average Training-Loss : {:.4f}'.format(avg_loss))            

            if config['wandb_key'] is not None and local_rank == 0:
                run.log({'Train loss (epoch)': avg_loss})
                run_validation(
                    model, val_dataloader, tokenizer_src, 
                    tokenizer_tgt, config["train_seq_len"], device, 
                    epoch, run, config, 
                    comet_model, num_examples=100
                )
            elif local_rank == 0:
                run_validation(
                    model, val_dataloader, tokenizer_src, 
                    tokenizer_tgt, config["train_seq_len"], device, 
                    epoch, None, config, 
                    comet_model, num_examples=100
                )

            # Save the model at the end of every epoch
            if local_rank == 0:
                # Save model.module.state_dict() if DDP is on, else model.state_dict()
                model_to_save = model.module if ddp_enabled else model
                model_filename = get_weights_file_path(config, f"{epoch:02d}")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'global_step': global_step
                }, model_filename)

    if local_rank == 0:
        if config['wandb_key'] is not None:
            run.log({'Train loss (epoch)': avg_loss})
            run_validation(
                model, test_dataloader, tokenizer_src, 
                tokenizer_tgt, config["test_seq_len"], device, 
                "test", run, config, 
                comet_model, num_examples=100
            )
        else:
            run_validation(
                model, test_dataloader, tokenizer_src, 
                tokenizer_tgt, config["test_seq_len"], device, 
                "test", None, config, 
                comet_model, num_examples=100
            )


if __name__ == '__main__':
    # torchrun --nproc_per_node=2 train.py
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)