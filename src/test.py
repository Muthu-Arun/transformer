import torch
import torch.nn as nn
import math

input_fh = open("data/input.txt","r")
lookup_table = dict()
reverse_lookup_table = dict()
char_set = set()
lines = input_fh.readlines()
print(len(lines))
# print(lines)
print(lines[:10])
for line in lines:
    for char in line:
        char_set.add(char)
print(char_set)
print(len(char_set))
tkn_id = 0
for i in char_set:
    
    lookup_table[i] = tkn_id
    reverse_lookup_table[tkn_id] = i
    tkn_id += 1
print(lookup_table)
print(reverse_lookup_table)

vocab_size = len(char_set)
dmodel = 512
max_len = 128

class PositionEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        pe = torch.zeros(max_len,dmodel)
        position = torch.arange(0,max_len,dtype=float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dmodel, 2).float() * (-math.log(10000.0) / dmodel))
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    def forward(self,x : torch.Tensor):
        x = x + self.pe[:,:x.size(1),:]
        # print("Shape of X : ",x.size())
        return x
class Embedding(nn.Module):
    def __init__(self):
       super().__init__()
       self.token_embedding = nn.Embedding(vocab_size,dmodel)
       self.pos_embedding = PositionEmbedding()
    
    def forward(self,x : torch.Tensor):
       x = self.token_embedding(x)
    #    print(x)
       x = self.pos_embedding(x)
    #    print(x)
       return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,dmodel = 512, num_heads = 8):
        super().__init__()
        self.num_heads = num_heads
        self.dmodel = 512
        assert dmodel % num_heads == 0, "Embedding Dimension (dmodel) Must be divisible by number of attention heads (num_heads)"
        self.head_dim = int(dmodel / num_heads)
        self.qkv_projection = nn.Linear(dmodel,3 * dmodel) # Create Prjections for Query, Key, Value
        self.out_projection = nn.Linear(dmodel,dmodel)
    def forward(self,x : torch.Tensor):
        B,T,D = x.size()
        qkv = self.qkv_projection(x)
        print(qkv.size())
        qkv = qkv.reshape(B,T,3,self.num_heads,self.head_dim)
        qkv = qkv.permute(2,0,3,1,4)
        q,k,v = qkv[0],qkv[1],qkv[2]
        attention_score = q @ k.transpose(-2,-1) / math.sqrt(self.head_dim)

        # Masks for preventing peeking into future tokens

        mask = torch.tril(torch.ones(T,T)).unsqueeze(0).unsqueeze(0)
        attention_score = attention_score.masked_fill(mask == 0, float('-inf'))

        attention_weights = torch.softmax(attention_score,-1)

        attention_output = attention_weights @ v

        attention_output = attention_output.transpose(1,2).contiguous().reshape(B,T,D)
        return self.out_projection(attention_output)
embeddingLayer = Embedding()
testString = "hello im"
tokens = [lookup_table[c] for c in testString]
inputTensor = torch.tensor(tokens).unsqueeze(0)
output = embeddingLayer(inputTensor)
print(output)
attention = MultiHeadSelfAttention()
after_self_attention = attention(output)
print("After Attention : ",after_self_attention)
