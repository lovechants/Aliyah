import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import math 
from typing import Optional, List

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model should be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        return torch.matmul(attention_weights, V), attention_weights

    def split_heads(self, x):
        batch_size = x.size(0)
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)

    def combine_heads(self, x):
        batch_size = x.size(0)
        return x.transpose(1,2).contiguous().view(batch_size, -1, self.d_model)

    def forward(self, query, key, value, mask=None):
        Q = self.split_heads(self.W_q(query))
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))

        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attention_output))

        return output, attention_weights

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class EncodeLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask = None):
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x 

class DecodeLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model) 
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        attn_output, _ = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        attn_output, attention_weights = self.cross_attention(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x, attention_weights

class CustomTransformer(nn.Module):
    def __init__(self,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 d_model: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 d_ff: int = 2048,
                 max_seq_length: int = 5000,
                 dropout: float = 0.1):
        super().__init__()

        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([
            EncodeLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            DecodeLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.final_layer = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model

    def generate_mask(self, src, tgt=None):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)

        if tgt is not None:
            tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)

            seq_length = tgt.size(1)

            nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
            tgt_mask = tgt_mask & nopeak_mask
            return src_mask, tgt_mask
        
        return src_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)

        src_embedded = self.dropout(self.positional_encoding(
            self.encoder_embedding(src) * math.sqrt(self.d_model)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        tgt_embedded = self.dropout(self.positional_encoding(
            self.decoder_embedding(tgt) * math.sqrt(self.d_model)))

        dec_output = tgt_embedded
        attention_weights = []

        for dec_layer in self.decoder_layers:
            dec_output, attn_weights = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
            attention_weights.append(attn_weights)

        output = self.final_layer(dec_output)
        return output, attention_weights

class ModernizationModel:
    def __init__(self, src_vocab_size, tgt_vocab_size, device):
        self.device = device
        self.model = CustomTransformer(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=512,
            num_heads=8,
            num_layers=6,
            d_ff=2048,
            max_seq_length=5000,
            dropout=0.1
        ).to(device)
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
        
    def train_step(self, src, tgt):
        self.model.train()
        self.optimizer.zero_grad()
        
        output, _ = self.model(src, tgt[:, :-1])
        loss = self.criterion(output.contiguous().view(-1, output.size(-1)), 
                            tgt[:, 1:].contiguous().view(-1))
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def validate(self, val_dataloader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                src, tgt = batch['source'].to(self.device), batch['target'].to(self.device)
                output, _ = self.model(src, tgt[:, :-1])
                loss = self.criterion(output.contiguous().view(-1, output.size(-1)),
                                   tgt[:, 1:].contiguous().view(-1))
                total_loss += loss.item()
                
        return total_loss / len(val_dataloader)

    def modernize_text(self, src_tokens, max_length=100):
        self.model.eval()
        
        with torch.no_grad():
            encoder_output = None
            decoder_input = torch.tensor([[1]], device=self.device)  # Start token
            
            for _ in range(max_length):
                output, _ = self.model(src_tokens, decoder_input)
                pred = output[:, -1:, :]
                pred_token = pred.argmax(dim=-1)
                
                decoder_input = torch.cat([decoder_input, pred_token], dim=-1)
                
                if pred_token.item() == 2:  # End token
                    break
                    
        return decoder_input.squeeze(0)




