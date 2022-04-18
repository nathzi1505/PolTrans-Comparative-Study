import torch
import torch.nn as nn
import numpy as np
import random

import torch.nn.functional as F

import time

SEED = 0

class PositionalEncoding(nn.Module):
    
    """
    Peforms the positional encoding of the words in the sequence.
    
    Parameters
    ---
    d_model : Final output dimension of model
    max_len : Maximum length of the sequence.
    
    Note
    ---
    [1] Sequence here refers to the sentence. Words are the elements/members
        of the sequence.
        
    References
    ---
    [1] Attention Is All You Need - Vaswani et. al. URL https://arxiv.org/abs/1706.03762
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__() 
        
        pos_encoding = torch.zeros(max_len, d_model)
        positions = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-np.log(10000.0) / d_model))
        
        pos_encoding[:, 0::2] = torch.sin(positions * div_term)
        pos_encoding[:, 1::2] = torch.cos(positions * div_term)
        
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        
        # To prevent PyTorch from using it to be trained by the optimizer.
        # Useful Tip:  If you have parameters in your model, which should be saved 
        # and restored in the state_dict, but not trained by the optimizer, 
        # you should register them as buffers.
        
        # Buffers won’t be returned in model.parameters(), so that the optimizer won’t 
        # have a change to update them.
        
        self.register_buffer('pe', pos_encoding)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]
    
class Time2VecEncoding(nn.Module):
    """
    Implements the Time2Vec Encoding
    
    Parameters
    ---
    f                : Periodic activation
    input_timesteps  : Size of input
    encoding_size    : Size of encoding 
    
    References
    ---
    [1] "Time2Vec: Learning a Vector Representation of Time" - https://arxiv.org/pdf/1907.05321.pdf
    
    """
    
    def __init__(self, f, input_timesteps, encoding_size):
        super(Time2VecEncoding, self).__init__()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")       
        
        self.f = f
        self.input_timesteps = input_timesteps
        self.encoding_size = encoding_size 
        self.omega = nn.parameter.Parameter(torch.randn(input_timesteps, 1, encoding_size)).to(device)
        self.psi = nn.parameter.Parameter(torch.randn(encoding_size)).to(device)
    
    def _get_encoding(self, tau):
    
        v1 = torch.matmul(tau, self.omega[:, :, 0:1]) + self.psi[0:1]
        v2 = torch.matmul(tau, self.omega[:, :, 1:] + self.psi[1:]) 
        
        encoded_result = torch.cat([v1, v2], 2)
        
        return encoded_result
        
    def forward(self, tau):
        return self._get_encoding(tau)
    
class TransTS(nn.Module):
    def __init__(self, input_timesteps, d_model=250, num_layers=1, dropout=0.25):
        super(TransTS, self).__init__()
        
        torch.manual_seed(SEED)
        random.seed(SEED)
        np.random.seed(SEED)
        
        self.model_type = 'Transformer'
        
        self.src_mask = None
        
        self.time2vec_encoding = Time2VecEncoding(torch.sin, input_timesteps, d_model)
        self.ts_encoder = PositionalEncoding(d_model=d_model)
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=10, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)        
    
        self.decoder = nn.Linear(d_model, 1)
        
        self.layers = nn.ModuleList([
            self.time2vec_encoding, 
            self.ts_encoder,
            self.transformer_encoder,
            self.decoder,
        ])
        
        self.init_weights()

    def init_weights(self):
        # TODO : Initialization of weights can be improved.
        # Examples : [1] Glorot (or Xavier) initialization
        #            [2] He initialization (Kaiming)
        
        for layer in self.layers:
            for p in layer.parameters():
                if p.dim() > 1:
                    nn.init.kaiming_uniform_(p)
        
#         initrange = 0.1    
#         self.decoder.bias.data.zero_()
#         self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
                
        src = self.time2vec_encoding(src)
        
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
                
        src += self.ts_encoder(src)  
        
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        output = F.leaky_relu(output)
        
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def save(self, PATH):
        torch.save(self.state_dict(), PATH)