import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
from layers.StandardNorm import Normalize
from layers.Cross_Modal_Align import CrossModal


class EnhancedSpatialTransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, num_nodes, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_nodes = num_nodes
        
        self.e1 = nn.Parameter(torch.randn(num_nodes, d_model))
        self.e2 = nn.Parameter(torch.randn(num_nodes, d_model))
        
        encoder_layers = [
            EnhancedTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            ) for _ in range(num_layers)
        ]
        
        self.layers = nn.ModuleList(encoder_layers)
        
    def forward(self, src):
        e_ada = torch.matmul(self.e1, self.e2.transpose(0, 1)) 
        
        output = src
        for layer in self.layers:
            output = layer(output, attention_bias=e_ada)
        
        return output

class EnhancedTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiheadAttentionWithBias(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, src, attention_bias=None):
        src2 = self.self_attn(
            src, src, src,
            attn_bias=attention_bias
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class MultiheadAttentionWithBias(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, attn_bias=None):
        batch_size = query.size(0)
        
        q = self.q_proj(query) 
        k = self.k_proj(key)  
        v = self.v_proj(value)  
        
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2) 
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2) 
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  
        
        if attn_bias is not None:
            attn_scores = attn_scores + attn_bias.unsqueeze(0)
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, v) 
        
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim) 
        
        output = self.out_proj(output)
        
        return output

class Dual(nn.Module):
    def __init__(
        self,
        device = "cuda:0",
        channel = 32,
        num_nodes = 7,
        seq_len = 96,
        pred_len = 96,
        dropout_n = 0.1,
        d_llm = 768,
        e_layer = 1,
        d_layer = 1,
        d_ff=32,
        head =8
    ):
        super().__init__()

        self.device = device
        self.channel = channel
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.dropout_n= dropout_n
        self.d_llm = d_llm
        self.e_layer = e_layer
        self.d_layer = d_layer
        self.d_ff = d_ff
        self.head = head

        self.normalize_layers = Normalize(self.num_nodes, affine=False).to(self.device)
        self.length_to_feature = nn.Linear(self.seq_len, self.channel).to(self.device)

        self.ts_encoder = EnhancedSpatialTransformerEncoder(d_model=self.channel, nhead= self.head, num_layers=self.e_layer, num_nodes=self.num_nodes,dropout=self.dropout_n).to(self.device)

        self.prompt_encoder_layer = nn.TransformerEncoderLayer(d_model = self.d_llm, nhead = self.head, batch_first=True, 
                                                               norm_first = True,dropout = self.dropout_n).to(self.device)
        self.prompt_encoder = nn.TransformerEncoder(self.prompt_encoder_layer, num_layers = self.e_layer).to(self.device)

        self.cross = CrossModal(d_model= self.num_nodes, n_heads= 1, d_ff=self.d_ff, norm='LayerNorm', attn_dropout=self.dropout_n, 
                                dropout=self.dropout_n, pre_norm=True, activation="gelu", res_attention=True, n_layers=1, store_attn=False).to(self.device)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model = self.channel, nhead = self.head, batch_first=True, norm_first = True, dropout = self.dropout_n).to(self.device)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers = self.d_layer).to(self.device)

        self.c_to_length = nn.Linear(self.channel, self.pred_len, bias=True).to(self.device)

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_data, input_data_mark, embeddings):
        input_data = input_data.float()
        input_data_mark = input_data_mark.float()
        embeddings = embeddings.float()

        input_data = self.normalize_layers(input_data, 'norm')

        input_data = input_data.permute(0,2,1) 
        input_data = self.length_to_feature(input_data)

        embeddings = embeddings.float()
        embeddings = embeddings.squeeze(-1)
        embeddings = embeddings.permute(0,2,1) 

        fft_result = torch.fft.fft(input_data)
        real_part = fft_result.real
        imag_part = fft_result.imag
        half_len = len(real_part) // 2
        input_data = torch.cat([real_part[:half_len], imag_part[:half_len]], dim=0)

        enc_out = self.ts_encoder(input_data) 
        enc_out = enc_out.permute(0,2,1) 
        embeddings = self.prompt_encoder(embeddings) 
        embeddings = embeddings.permute(0,2,1) 

        cross_out = self.cross(enc_out, embeddings, embeddings)
        cross_out = cross_out.permute(0,2,1)

        dec_out = self.decoder(cross_out, cross_out) 

        dec_out = self.c_to_length(dec_out) 
        dec_out = dec_out.permute(0,2,1) 

        dec_out = self.normalize_layers(dec_out, 'denorm')

        return dec_out