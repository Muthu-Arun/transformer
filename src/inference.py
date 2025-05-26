import torch

model = torch.load("first_iter.pth")
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

context_length = 128  # must match max_len in your model
testString = "T"

for i in range(3000):
    context = testString[-context_length:]  # slide the window
    inputTensor = torch.tensor([lookup_table[c] for c in context]).unsqueeze(0).to(device)
    logits = model(inputTensor)
    logits = logits[0, -1, :]
    probs = torch.softmax(logits, dim=0)
    token = torch.multinomial(probs, num_samples=1).item()
    testString += reverse_lookup_table[token]

print(testString)
outfile = open("data/output.txt","w")
