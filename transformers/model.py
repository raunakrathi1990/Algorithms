import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)

    def forward(self, x):
        return self.embedding(x)* math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    def __init__(self,d_model: int, seq_len: int, dropout: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = dropout

        pe = torch.zeros(seq_len,d_model)
        position = torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0))/d_model )

        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe',pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad(False)
        return self.dropout(x)
    

class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.zeros(1)) #Multiplier
        self.bias = nn.Parameter(torch.ones(1)) # added

    def forward(self, x):
        mean = x.mean(dim = -1,keepdim=True)
        std = x.std(dim = -1,keepdim = True)
        return self.alpha * (x-mean)/(std + self.eps) + self.bias 

class FeedForwardBlock(nn.Module):
    def __init__(self,d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_layer_1 = nn.Linear(d_model,d_ff,bias=True) #W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_layer_2 = nn.Linear(d_ff,d_model) #W2 and B2

    def forward(self,x):
        #(batch,seq_len, d_model) --> (Batch, seq_len, d_ff) --> (Batch, seq_len, d_model)
        return self.linear_layer_2(self.dropout(nn.ReLU(self.linear_layer_1(x))))
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, h:int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        
        assert self.d_model % self.h == 0, "input dimension should be divisible by number of attention heads"

        self.d_k = self.d_model //self.h
        self.w_q = nn.Linear(self.d_model, self.d_model)
        self.w_k = nn.Linear(self.d_model, self.d_model)
        self.w_v = nn.Linear(self.d_model, self.d_model)

        self.w_o = nn.Linear(self.d_model,self.d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):

        d_k = query.shape[-1]


        # (Batch, seq_len, h, d_k) --> (Batch, h, seq_len, seq_len)
        attention_score = (query @ key.transpose(-2,-1))/ math.sqrt(d_k)
        if mask is not None:
            attention_score.masked_fill_(mask == 0, -1e9)
        attention_score = attention_score.softmax(dim=-1) #(Batch, h, seq_len, seq_len)
        if dropout is not None:
            attention_score = dropout(attention_score)
        
        return (attention_score @ value), attention_score
        

    def forward(self, q, k, v, mask):
        query = self.w_q(q) #(Batch, seq_len, d_model) --> (Batch, d_model, seq_len)
        key = self.w_k(k) #(Batch, seq_len, d_model) --> (Batch, d_model, seq_len)
        value = self.w_v(v) #(Batch, seq_len, d_model) --> (Batch, d_model, seq_len)

        # (Batch, seq_len, d_model) --> (Batch, seq_len, h, d_k) --> (Batch, h, seq_len, d_k)
        query = query.view(query.shape[0],query.shape[1],self.h, self.d_k).tranpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0],value.shape[1], self.h, self.d_k).transpose(1,2)

        x,attention_score = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        # (Batch, h, seq_len, d_k) --> (Batch, seq_len, h , d_k) --> (Batch, seq_len, d_model) 
        x = x.transpose(1,2).contigous().view(x.shape[0],-1, self.h * self.d_k)

        # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        return self.w_o(x)
    
class ResidulConnection(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    

class EncoderBlock(nn.Module):
    def __init__(self, self_Attention_Block: MultiHeadAttention, feed_Forward_Block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_Attention_Block = self_Attention_Block
        self.feed_Forward_Block = feed_Forward_Block
        self.residual_Connection = nn.ModuleList([ResidulConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_Connection[0](x, lambda x: self.self_Attention_Block(x, x, x, src_mask))
        x = self.residual_Connection[1](x, self.feed_Forward_Block)
        return x

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)
    


