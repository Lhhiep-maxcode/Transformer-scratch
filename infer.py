from pathlib import Path
from config import get_config, latest_weights_file_path 
from model import build_transformer
from tokenizers import Tokenizer
from datasets import load_dataset
from dataset import BilingualDataset, causal_mask
import torch
import torch.nn.functional as F

def length_norm(score, length, alpha=0.6):
    # Wu's formula (used in Google's NMT system) for Length Penalization
    return score / (((5 + length)**alpha) / ((5 + 1)**alpha))


def beam_search(model, beam_size, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_id = tokenizer_tgt.token_to_id('[SOS]')
    eos_id = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(encoder_input, encoder_mask)  # (B=1, seq_len, d_model)
    # Initialize the decoder input with the sos token
    decoder_initial_input = torch.empty(1, 1).fill_(sos_id).type_as(encoder_input).to(device)

    # Create a candidate list
    # (generated text, score)
    candidates = [(decoder_initial_input, 0)]
    finished_candidates = []

    for _ in range(max_len):
        new_candidates = []
        if len(candidates) == 0:
            break 

        for candidate, score in candidates:
            if candidate[0][-1].item() == eos_id:
                finished_candidates.append((candidate, score))
                continue

            # Build the candidate's mask
            candidate_mask = causal_mask(candidate.size(1)).type_as(encoder_mask).to(device) # (1, length of sequence, length of sequence)
            out = model.decode(encoder_output, encoder_mask, candidate, candidate_mask)
            # get next token probabilities
            logits = model.project(out[:, -1])    # (batch, d_model)
            prob = F.log_softmax(logits, dim=-1)
            # get the top k candidates
            topk_prob, topk_idx = torch.topk(prob, beam_size, dim=-1)
            for i in range(beam_size):
                # for each of the top k candidates, get the token and its probability
                # hardcode for evaluation with batch size = 1
                token = topk_idx[0][i].unsqueeze(0).unsqueeze(0)    # (1, 1)   
                token_prob = topk_prob[0][i].item()
                new_candidate = torch.cat([candidate, token], dim=1)     # (1, length of generated sequence + 1)
                # We sum the log probabilities because the probabilities are in log space
                new_candidates.append((new_candidate, (score + token_prob)))

        # Sort the new candidates by their normalized score so that it would not be bias to short sentence
        candidates = sorted(new_candidates, key=lambda x: length_norm(x[1], x[0].size(1)), reverse=True)
        # Keep only the top k candidates
        candidates = candidates[:beam_size]


    finished_candidates.extend(candidates)
    finished_candidates = sorted(finished_candidates, key=lambda x: length_norm(x[1], x[0].size(1)), reverse=True)

    # Return the best candidate
    return finished_candidates[0][0].squeeze(0)  # (seq_len)


def greedy_search(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_id = tokenizer_tgt.token_to_id('[SOS]')
    eos_id = tokenizer_tgt.token_to_id('[EOS]')
    pad_id = tokenizer_tgt.token_to_id('[PAD]')

    encoder_output = model.encode(encoder_input, encoder_mask)
    decoder_input = torch.empty((1, 1)).fill_(sos_id).type_as(encoder_input).to(device) # (batch, 1)
    for _ in range(max_len):
        if decoder_input.size(1) == max_len:
            break
        
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(encoder_mask).to(device) # (1, length of sequence, length of sequence)
        decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (batch, length of sequence, d_model)
        # print("Decoder out", decoder_output.shape)
        # get next token
        prob = model.project(decoder_output[:, -1]) # (batch, d_model) -> (batch, vocab_size)
        # print("Probability", prob.shape)
        _, next_word = torch.max(prob, dim=-1)
        # print("Next word", next_word)
        decoder_input = torch.cat(
            [decoder_input, torch.empty((1, 1)).type_as(decoder_input).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_id:
            break

    return decoder_input.squeeze(0)   # (seq_len)


def translate(sentence):
    # Define the device, tokenizers, and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    config = get_config()
    tokenizer_src = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_src']))))
    tokenizer_tgt = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_tgt']))))
    model = build_transformer(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(), config["seq_len"], config['seq_len'], d_model=config['d_model']).to(device)

    # Load the pretrained weights
    model_filename = config['preload_path']
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