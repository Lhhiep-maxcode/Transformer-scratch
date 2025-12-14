import torch
import numpy as np
from torch.utils.data import Dataset, Sampler

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

class LengthBasedCurriculumSampler(Sampler):
    def __init__(self,
                 data_level,
                 batch_size,
                 start_percentile=0.1,
                 end_percentile=1.0,
                 total_steps=10000):
        """
        Samples batches based on increasing sequence length percentile
        over training.

        Args:
            data_level (list or np.array): List of lengths for each
                                             data sample.
            batch_size (int): The size of each batch.
            start_percentile (float): Initial length percentile threshold
                                      (0.0 to 1.0).
            end_percentile (float): Final length percentile threshold
                                    (0.0 to 1.0).
            total_steps (int): Total number of training steps over which
                               the curriculum progresses.
        """
        self.data_level = np.array(data_level)
        self.indices = np.argsort(self.data_level) # Indices sorted by length
        self.sorted_lengths = self.data_level[self.indices]
        self.batch_size = batch_size
        self.start_percentile = start_percentile
        self.end_percentile = end_percentile
        self.total_steps = total_steps
        self.current_step = 0

        self.num_samples = len(data_level)
        # Calculate initial and final indices based on percentiles
        self.start_idx = int(self.start_percentile * self.num_samples)
        self.final_max_idx = int(self.end_percentile * self.num_samples)

    def get_current_max_index(self):
        # Linearly increase the maximum index allowed over total_steps
        progress = min(1.0, self.current_step / self.total_steps)
        increase = progress * (self.final_max_idx - self.start_idx)
        current_max_idx = int(self.start_idx + increase)
        # Ensure we always include at least the starting percentile of data
        return max(self.start_idx, current_max_idx)

    def __iter__(self):
        current_max_idx = self.get_current_max_index()
        # Eligible indices are those up to the current maximum length threshold
        eligible_indices = self.indices[:current_max_idx]

        if len(eligible_indices) < self.batch_size:
            # Handle cases where eligible data is too small (e.g., early steps)
            # Might repeat samples or use a smaller batch
            eligible_indices = np.random.choice(
                eligible_indices, size=self.batch_size, replace=True
            )
        else:
             # Shuffle the eligible indices for the current epoch/step
            np.random.shuffle(eligible_indices)

        # Yield batches (simplified batching logic)
        num_batches = 0
        for i in range(0, len(eligible_indices), self.batch_size):
            batch_indices = eligible_indices[i : i + self.batch_size]
            # Drop last incomplete batch for simplicity
            if len(batch_indices) == self.batch_size:
                yield batch_indices.tolist()
                num_batches += 1

        # Increment step after yielding all batches for this iteration
        # In a real trainer, step update would happen per optimizer step
        # This simplified version increments once per __iter__ call
        # Rough step increment
        self.current_step += num_batches

    def __len__(self):
        # Estimated number of batches per epoch/iteration
        current_max_idx = self.get_current_max_index()
        num_eligible = len(self.indices[:current_max_idx])
        return num_eligible // self.batch_size