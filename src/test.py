import torch
import torch.nn as nn
import math
import torch.optim as optim

device = torch.device("cuda")
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
        # print(qkv.size())
        qkv = qkv.reshape(B,T,3,self.num_heads,self.head_dim)
        qkv = qkv.permute(2,0,3,1,4)
        q,k,v = qkv[0],qkv[1],qkv[2]
        attention_score = q @ k.transpose(-2,-1) / math.sqrt(self.head_dim)

        # Masks for preventing peeking into future tokens

        mask = torch.tril(torch.ones(T,T)).unsqueeze(0).unsqueeze(0).to(device)
        attention_score = attention_score.masked_fill(mask == 0, float('-inf'))

        attention_weights = torch.softmax(attention_score,-1)

        attention_output = attention_weights @ v

        attention_output = attention_output.transpose(1,2).contiguous().reshape(B,T,D)
        return self.out_projection(attention_output)

class DecoderBlock(nn.Module):
    def __init__(self,dmodel = 512, num_heads = 8, ff_hidden_dim = 2048, dropout = 0.1):
        super().__init__()
        self.self_attention = MultiHeadSelfAttention()
        self.norm1 = nn.LayerNorm(dmodel)
        self.norm2 = nn.LayerNorm(dmodel)
        self.feed_forward = nn.Sequential(
            nn.Linear(dmodel,ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim,dmodel),
            
        )
        self.dropout = nn.Dropout(dropout)
    def forward(self,x : torch.Tensor):
            attention_out = self.self_attention(self.norm1(x))
            x = x + self.dropout(attention_out)

            ff_out = self.feed_forward(self.norm2(x))
            x = x + self.dropout(ff_out)

            return x


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = Embedding()
        self.decoders = nn.Sequential(*[DecoderBlock() for _ in range(6)])
        self.layernorm = nn.LayerNorm(dmodel)
        self.lm_head = nn.Linear(dmodel,vocab_size)
    
    def forward(self,x : torch.Tensor):
        x = self.embedding(x)
        x = self.decoders(x)
        x = self.layernorm(x)
        logits = self.lm_head(x)
        return logits
    
# embeddingLayer = Embedding()
testString = "T"
# tokens = [lookup_table[c] for c in testString]
# inputTensor = torch.tensor(tokens).unsqueeze(0)


# output = embeddingLayer(inputTensor)
# print(output)
# attention = MultiHeadSelfAttention()
# after_self_attention = attention(output)
# print("After Attention : ",after_self_attention)
model = Transformer().to(device)
optimizer = optim.Adam(model.parameters(),lr=1e-4)
loss_fn = nn.CrossEntropyLoss()
text = open("data/input.txt").read()
def get_batch(text, block_size=128, batch_size=32):
    ix = torch.randint(0, len(text) - block_size - 1, (batch_size,))
    x = torch.stack([torch.tensor([lookup_table[c] for c in text[i:i+block_size]]) for i in ix]).to(device)
    y = torch.stack([torch.tensor([lookup_table[c] for c in text[i+1:i+block_size+1]]) for i in ix]).to(device)
    return x, y

for step in range(5000):
    x_batch, y_batch = get_batch(text)
    logits = model(x_batch)
    B, T, V = logits.shape
    loss = loss_fn(logits.view(B*T, V), y_batch.view(B*T))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"Step {step} | Loss: {loss.item():.4f}")
# model.forward(inputTensor)
model.eval()

for i in range(127):
    inputTensor = torch.tensor([lookup_table[c] for c in testString]).unsqueeze(0).to(device)
    logits = model(inputTensor)
    logits = logits[0,-1,:]
    token = torch.argmax(logits)
    print(token)
    char_tkn = reverse_lookup_table[token.item()]
    testString += char_tkn

    # torch.argmax(logits)

print(testString)
torch.save(model.state_dict(),"first_iter.pth")