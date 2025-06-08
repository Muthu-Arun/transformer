import torch
import torch.nn as nn
import math
import pickle

device = torch.device("cuda"if torch.cuda.is_available() else "cpu")
input_fh = open("data/input.txt","r")
with open("data/vocabfile.pkl", "rb") as f:
    lookup_table, reverse_lookup_table = pickle.load(f)

vocab_size = len(lookup_table)
dmodel = 512
max_len = 256

class Embedding(nn.Module):
    def __init__(self):
       super().__init__()
       self.token_embedding = nn.Embedding(vocab_size,dmodel)
       self.pos_embedding = nn.Embedding(max_len,dmodel)
    
    def forward(self,x : torch.Tensor):
       B, T = x.shape
       x = self.token_embedding(x)
    #    print(x)
       x += self.pos_embedding(torch.arange(T,device=device))
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
testString = "Tech"
model = Transformer()
model.load_state_dict(torch.load("first_iter.pth"))
model.to(device)
model = torch.compile(model)
model.eval()
def get_batch(text, block_size=256, batch_size=64):
    ix = torch.randint(0, len(text) - block_size - 1, (batch_size,))
    x = torch.stack([torch.tensor([lookup_table[c] for c in text[i:i+block_size]]) for i in ix]).to(device)
    y = torch.stack([torch.tensor([lookup_table[c] for c in text[i+1:i+block_size+1]]) for i in ix]).to(device)
    return x, y
for i in range(2550):
    context = testString[-max_len:]
    inputTensor = torch.tensor([lookup_table[c] for c in context]).unsqueeze(0).to(device)
    logits = model(inputTensor)
    logits = logits[0,-1,:]
    token = torch.softmax(logits,dim=0)
    token = torch.multinomial(token,num_samples=1)
    # print(token)
    char_tkn = reverse_lookup_table[token.item()]
    testString += char_tkn

    # torch.argmax(logits)
with open("data/generated.txt","w") as fout:
    fout.write(testString)
print(testString)