'''
VAET MODEL: VAE WITH Transformer context for hyperparameter optimization.

Implementation of the Variational Autoencoder is adapted from [1]. 
Implementation of the encoder layer as in [2], [3], and [4] for sequence to 
sequence modeling.

[1] https://github.com/dungxibo123/vae
[2] https://arxiv.org/pdf/1706.03762.pdf
[3] https://arxiv.org/pdf/2005.12872.pdf
[4] https://github.com/idiap/potr/ 
'''

import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

from vaet_utils import weight_init, normal_init_, xavier_init_, device

class VAET_Model(nn.Module):
    def __init__(self, params):
        """
            @param params: parameters for the model
        """
        super(VAET_Model, self).__init__()
        self.params = params
        self.device = device()
        self.context_size = self.params['trials'] + self.params['obs'] - 1

        # Context Transformer
        self.cntxembedding = self.context_embedding(params)
        self.context = ContextTransformerEncoder(
            num_layers=params['transformer_layers'],
            model_dim=params['transformer_model_dim'],
            num_heads=params['transformer_num_heads'],
            dim_ffn=params['transformer_dim_ffn'],
            dropout=params['dropout'],
            init_fn = normal_init_ \
                if params['init_fn'] == 'normal_init' else xavier_init_,
            pre_normalization=params['transformer_pre_normalization']
        )

        self.en = nn.Sequential(
            nn.Linear(params['input_size'] + 1 + (params['transformer_model_dim'] * self.context_size), 
                      params['enc_hidden_dim_vae']),

            nn.BatchNorm1d(params['enc_hidden_dim_vae']),
            nn.Tanh()
        )

        self.mu = nn.Linear(params['enc_hidden_dim_vae'], params['latent_dim_vae'])
        self.var = nn.Linear(params['enc_hidden_dim_vae'], params['latent_dim_vae'])
        
        self.de = nn.Sequential(
            nn.Linear(params['latent_dim_vae'] + 1 + (params['transformer_model_dim'] * self.context_size),
                      params['dec_hidden_dim_vae']),
            nn.BatchNorm1d(params['dec_hidden_dim_vae']),
            nn.Tanh()
        )

        self.final_layer=nn.Sequential(
            nn.Linear(params['dec_hidden_dim_vae'], params['input_size']),
        )
          
    def context_embedding(self, params):
        init_fn = normal_init_ \
            if params['init_fn'] == 'normal_init' else xavier_init_
        ctx_embedding = nn.Sequential(
            nn.Linear(params['input_size']+1, params['transformer_model_dim']),
            nn.Dropout(0.1)
        )
        weight_init(ctx_embedding, init_fn_=init_fn)
        return ctx_embedding

    def encode(self, x, I, c_latent):
        en_input = torch.cat([x, I.unsqueeze(1), 
                              c_latent.view(-1, self.context_size * self.params['transformer_model_dim'] )], dim=1)
        #x = torch.flatten(x)
        res = self.en(en_input)
        mu = self.mu(res)
        log_var = self.var(res)
        return mu, log_var
            
    def decode(self, x, I, c_latent):
        de_input = torch.cat([x, I.unsqueeze(1), 
                              c_latent.view(-1, self.context_size * self.params['transformer_model_dim'])], dim=1)

        res = self.de(de_input)
        res = self.final_layer(res)
        return res
    
    def reparameterize(self, mu, log_var):
        epsilon = torch.normal(mu,torch.exp(0.5 * log_var))
        return mu + log_var * epsilon
    
    def forward(self, x, I, C, mask):
        
        # Transform the context
        emb = self.cntxembedding(C)
        c_latent = self.context(emb, mask)[0]

        # VAE Encoder Decoder
        mu, log_var = self.encode(x, I, c_latent)
        norm = self.reparameterize(mu, log_var)
        res = self.decode(norm, I, c_latent)
        
        return (res, x, mu, log_var)

    def generate(self, I, C, mask):
        emb = self.cntxembedding(C)
        c_latent = self.context(emb, mask)[0]
        norm = torch.normal(torch.zeros(self.params['latent_dim_vae']).to(self.device), torch.ones(self.params['latent_dim_vae']).to(self.device))
        # add dim 1 to norm to match the batch size
        norm = norm.unsqueeze(0)
        res = self.decode(norm, I, c_latent)
        return res

class ContextTransformerEncoder(nn.Module):
  def __init__(self,
               num_layers=6,
               model_dim=256,
               num_heads=8,
               dim_ffn=2048,
               dropout=0.1,
               init_fn=normal_init_,
               pre_normalization=False):
    super(ContextTransformerEncoder, self).__init__()
    """Transforme encoder initialization."""
    self._num_layers = num_layers
    self._model_dim = model_dim
    self._num_heads = num_heads
    self._dim_ffn = dim_ffn
    self._dropout = dropout
    self._pre_normalization = pre_normalization

    self._encoder_stack = self.init_encoder_stack(init_fn)

  def init_encoder_stack(self, init_fn):
    """Create the stack of encoder layers."""
    stack = nn.ModuleList()
    for s in range(self._num_layers):
      layer = EncoderLayer(
          model_dim=self._model_dim,
          num_heads=self._num_heads,
          dim_ffn=self._dim_ffn,
          dropout=self._dropout,
          init_fn=init_fn,
          pre_normalization=self._pre_normalization
      )
      stack.append(layer)
    return stack

  def forward(self, input_sequence, mask):
    outputs = input_sequence

    for l in range(self._num_layers):
      outputs, attn_weights = self._encoder_stack[l](outputs, mask)

    return outputs, attn_weights
  

class EncoderLayer(nn.Module):
  """Implements the transformer encoder Layer."""
  def __init__(self,
               model_dim=256,
               num_heads=8,
               dim_ffn=2048,
               dropout=0.1,
               init_fn=normal_init_,
               pre_normalization=False):
    super(EncoderLayer, self).__init__()

    self._model_dim = model_dim
    self._num_heads = num_heads
    self._dim_ffn = dim_ffn
    self._dropout = dropout
    self._pre_normalization = pre_normalization

    self._self_attn = nn.MultiheadAttention(model_dim, num_heads,dropout= dropout, batch_first=True)
    self._relu = nn.ReLU()
    self._dropout_layer = nn.Dropout(self._dropout)

    self._linear1 = nn.Linear(model_dim, self._dim_ffn)
    self._linear2 = nn.Linear(self._dim_ffn, self._model_dim)
    self._norm1 = nn.LayerNorm(model_dim, eps=1e-5)
    self._norm2 = nn.LayerNorm(model_dim, eps=1e-5)

    weight_init(self._linear1, init_fn_=init_fn)
    weight_init(self._linear2, init_fn_=init_fn)

  def forward(self, source_seq, mask):
    if self._pre_normalization:
      return self.forward_pre(source_seq, mask)

    return self.forward_post(source_seq, mask)

  def forward_post(self, source_seq, mask):
    query = source_seq 
    key = query
    value = source_seq

    attn_output, attn_weights = self._self_attn(
        query, 
        key, 
        value, 
        key_padding_mask = mask,
        need_weights=True
    )

    norm_attn = self._dropout_layer(attn_output) + source_seq
    norm_attn = self._norm1(norm_attn)

    output = self._linear1(norm_attn)
    output = self._relu(output)
    output = self._dropout_layer(output)
    output = self._linear2(output)
    output = self._dropout_layer(output) + norm_attn
    output = self._norm2(output)

    return output, attn_weights

  def forward_pre(self, source_seq_, mask):
    source_seq = self._norm1(source_seq_)
    query = source_seq 
    key = query
    value = source_seq

    attn_output, attn_weights = self._self_attn(
        query, 
        key, 
        value, 
        key_padding_mask = mask,
        need_weights=True
    )

    norm_attn_ = self._dropout_layer(attn_output) + source_seq_
    norm_attn = self._norm2(norm_attn_)

    output = self._linear1(norm_attn)
    output = self._relu(output)
    output = self._dropout_layer(output)
    output = self._linear2(output)
    output = self._dropout_layer(output) + norm_attn_

    return output, attn_weights



