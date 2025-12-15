import torch
import numpy as np
from torch.utils.data import Dataset, Sampler, BatchSampler

class BilingualDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences, tokenizer_src, tokenizer_tgt, seq_len):
        super().__init__()
        self.seq_len = seq_len

        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        assert len(self.src_sentences) == len(self.tgt_sentences), "Source sentences and Target sentences do not match"

        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, idx):
        src_sentence = self.src_sentences[idx].strip()
        tgt_sentence = self.tgt_sentences[idx].strip()

        # Transform the text into tokens: "I love you" -> [2, 15, 3]
        enc_input_tokens = self.tokenizer_src.encode(src_sentence).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_sentence).ids

        # Add [sos], [eos] and [padding] to each sentence
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # We will add <s> and </s>
        # We will only add <s> on decoder input, and </s> only on the label
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError(f"Sentence is too long. Src input length: {len(enc_input_tokens)}. Tgt input length: {len(dec_input_tokens)}")

        # Add <s> and </s> to encoder input
        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
        ], dim=0)

        # Add only <s> token
        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
        ], dim=0)

        # Add only </s> token
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Double check the size of the tensors to make sure they are all seq_len long
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len),
            "label": label,  # (seq_len)
            "src_text": src_sentence,
            "tgt_text": tgt_sentence,
        }
    
def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0

class DistributedLengthBasedCurriculumBatchSampler(BatchSampler):
    def __init__(
        self,
        data_lengths,
        batch_size,
        start_percentile=0.1,
        end_percentile=1.0,
        total_steps=10000,
        drop_last=True,
        shuffle=True,
        rank=None,
        world_size=None,
        seed=0,
        ddp_enabled=True,  # <--- THAM SỐ MỚI
    ):
        """
        Args:
            ddp_enabled (bool): Nếu False, sẽ chạy chế độ Single GPU (rank=0, world_size=1).
        """
        self.data_lengths = np.asarray(data_lengths)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.base_seed = seed
        self.ddp_enabled = ddp_enabled # Lưu lại trạng thái

        self.total_steps = total_steps
        self.current_step = 0
        self.epoch = 0

        # --- LOGIC XỬ LÝ DDP ---
        # Chỉ lấy thông tin từ dist nếu ddp_enabled=True VÀ dist đã được khởi tạo
        if self.ddp_enabled and dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank() if rank is None else rank
            self.world_size = dist.get_world_size() if world_size is None else world_size
        else:
            # Fallback về chế độ Single Device
            self.rank = 0
            self.world_size = 1

        # Sắp xếp index theo chiều dài
        self.sorted_indices = np.argsort(self.data_lengths)
        self.num_samples = len(self.sorted_indices)
        
        # Tính toán boundary
        self.start_idx = max(1, int(start_percentile * self.num_samples))
        self.final_idx = max(self.start_idx, int(end_percentile * self.num_samples))

    def step(self, n: int = 1):
        self.current_step += n

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _current_max_idx(self):
        progress = min(1.0, self.current_step / self.total_steps)
        max_idx = int(self.start_idx + progress * (self.final_idx - self.start_idx))
        return max(self.start_idx, max_idx)

    def __iter__(self):
        max_idx = self._current_max_idx()
        
        # 1. Lấy dữ liệu global hợp lệ
        eligible_global = self.sorted_indices[:max_idx]
        
        # 2. Xử lý chia hết cho world_size
        # Nếu world_size=1 (tắt DDP), dòng này không thay đổi gì cả (vẫn giữ nguyên len)
        total_eligible = len(eligible_global)
        valid_len = (total_eligible // self.world_size) * self.world_size
        eligible_global = eligible_global[:valid_len]

        # 3. Shard cho từng rank
        # Nếu world_size=1, cú pháp [0 :: 1] sẽ lấy toàn bộ list -> Đúng logic Single GPU
        eligible = eligible_global[self.rank :: self.world_size]

        # 4. Shuffle
        if self.shuffle:
            g = np.random.default_rng(seed=self.base_seed + self.epoch)
            eligible = g.permutation(eligible)

        # 5. Tạo batch
        if self.drop_last:
            num_batches = len(eligible) // self.batch_size
            eligible = eligible[:num_batches * self.batch_size]
        
        batch = []
        for idx in eligible:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        max_idx = self._current_max_idx()
        # Tính toán len tương thích với world_size
        valid_len = (max_idx // self.world_size) * self.world_size
        samples_per_rank = valid_len // self.world_size
        
        if self.drop_last:
            return samples_per_rank // self.batch_size
        else:
            return (samples_per_rank + self.batch_size - 1) // self.batch_size