import torch
import torch.nn as nn

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * torch.sqrt(torch.tensor(self.d_model))
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = self.pe.unsqueeze(0) # add batch dimension (batch, seq_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.size(1), :]).requires_grad_(False)
        return self.dropout(x)
    

class LayerNormalization(nn.Module):
    def __init__(self, esp: float = 1e-6):
        super().__init__()
        self.esp = esp
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.esp) + self.beta
    

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model) -> (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
        return self.fc2(self.dropout(torch.relu(self.fc1(x))))
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, seq_len: int, n_heads: int, dropout: float = 0.1):
        super().__inti__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.n_heads = n_heads
        assert d_model % n_heads == 0, f'd_model = {self.d_model} should be divisible by n_heads = {self.n_heads}'

        self.d_k = self.d_model // self.n_heads
        self.w_q = nn.Linear(self.d_model, self.d_model)
        self.w_k = nn.Linear(self.d_model, self.d_model)
        self.w_v = nn.Linear(self.d_model, self.d_model)
        self.w_o = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(self, query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        # (batch, n_heads, seq_len, d_k) x (batch, n_heads, d_k, seq_len) -> (batch, n_heads, seq_len, seq_len))
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k))
        if mask is not None:
            scores = scores.masked_fill_(mask == 0, -1e9)
        
        # (batch, n_heads, seq_len, seq_len) --> (batch, n_heads, seq_len, seq_len)
        attention_score = scores.softmax(dim=-1) # softmax along the last dimension (sum of the last dimension = 1)
        if dropout is not None:
            attention_score = dropout(attention_score)
        
        # (batch, n_heads, seq_len, d_k), (batch, n_heads, seq_len, seq_len)
        return torch.matmul(attention_score, value), attention_score

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k)   # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, n_heads, d_k) --> (batch, n_heads, seq_len, d_k)
        query = query.view(-1, self.seq_len, self.n_heads, self.d_k).transpose(1, 2)
        key = key.view(-1, self.seq_len, self.n_heads, self.d_k).transpose(1, 2)
        value = value.view(-1, self.seq_len, self.n_heads, self.d_k).transpose(1, 2)
        x, attention_score = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        # (batch, n_heads, seq_len, d_k) --> (batch, seq_len, n_heads, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(-1, self.seq_len, self.d_model)
        return self.w_o(x)
    

class ResidualConnection(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(self, multihead_attention: MultiHeadAttention, feedforward: FeedForwardBlock, dropout:float = 0.1):
        super().__init__()
        self.multihead_attention = multihead_attention
        self.feedforward = feedforward
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout=dropout) for _ in range(2)])
    
    def forward(self, x, src_mask):
        """
        src_mask is to hide the padding tokens in the input sequence
        """
        x = self.residual_connections[0](x, lambda x: self.multihead_attention(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feedforward)
        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)
    

class DecoderBlock(nn.Module):
    def __init__(self, self_attention: MultiHeadAttention, cross_attention: MultiHeadAttention, feedforward: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feedforward = feedforward
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout=dropout) for _ in range(3)])
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feedforward)
        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    

class ProjectionLayer(nn.Module):
    """
    The final layer of the Transformer model projecting the output of the decoder to the vocabulary size
    """
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        # (Batch, seq_len, d_model) --> (Batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim=-1)
    

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: nn.Module, tgt_embed: nn.Module, 
                 src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
    
    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
