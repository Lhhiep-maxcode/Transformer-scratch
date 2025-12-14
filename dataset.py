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

class LengthBasedCurriculumBatchSampler(BatchSampler):
    def __init__(
        self,
        data_lengths,
        batch_size,
        start_percentile=0.1,
        end_percentile=1.0,
        total_steps=10000,
        drop_last=True,
        shuffle=True,
    ):
        """
        Length-based curriculum BatchSampler.

        Curriculum progresses based on *training step*,
        NOT epoch or __iter__ calls.

        Args:
            data_lengths (list | np.array): sequence length per sample
            batch_size (int)
            start_percentile (float)
            end_percentile (float)
            total_steps (int): total optimizer steps
            drop_last (bool)
            shuffle (bool)
        """
        self.data_lengths = np.asarray(data_lengths)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

        self.total_steps = total_steps
        self.current_step = 0  # MUST be updated externally

        # sort indices by length (ascending)
        self.sorted_indices = np.argsort(self.data_lengths)

        self.num_samples = len(self.sorted_indices)
        self.start_idx = max(1, int(start_percentile * self.num_samples))
        self.final_idx = max(
            self.start_idx, int(end_percentile * self.num_samples)
        )

    # --------------------------------------------------
    # curriculum logic
    # --------------------------------------------------
    def step(self, n: int = 1):
        """Call this after each optimizer.step()."""
        self.current_step += n

    def _current_max_idx(self):
        progress = min(1.0, self.current_step / self.total_steps)
        max_idx = int(
            self.start_idx
            + progress * (self.final_idx - self.start_idx)
        )
        return max(self.start_idx, max_idx)

    # --------------------------------------------------
    # BatchSampler interface
    # --------------------------------------------------
    def __iter__(self):
        max_idx = self._current_max_idx()
        eligible = self.sorted_indices[:max_idx]

        if len(eligible) < self.batch_size:
            return  # do NOT yield duplicated samples

        if self.shuffle:
            eligible = np.random.permutation(eligible)

        for i in range(0, len(eligible), self.batch_size):
            batch = eligible[i : i + self.batch_size]

            if len(batch) < self.batch_size and self.drop_last:
                continue

            yield batch.tolist()

    def __len__(self):
        max_idx = self._current_max_idx()
        if self.drop_last:
            return max_idx // self.batch_size
        else:
            return (max_idx + self.batch_size - 1) // self.batch_size