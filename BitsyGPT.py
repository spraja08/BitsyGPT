import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import time
from functools import wraps
from ast import literal_eval

configs = {}
#torch.manual_seed(2024)
class Data:
    def __init__(self):
        with open(configs["data_path"], 'r') as f:
            self.text = f.read()
        self.text = self.text.replace('\xad', '')
        self.vocab = sorted(list(set(self.text)))
        self.vocab_size = len(self.vocab)

        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for i, ch in enumerate(self.vocab)}

        self.data = torch.tensor(self.encode(self.text))
        n = int(len(self.text) * 0.9)
        self.train_data = self.data[n:]
        self.val_data = self.data[:n]

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return ''.join(self.itos[i] for i in l)

    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - configs["block_size"], (configs["batch_size"],))
        x = torch.stack([data[i:i+configs["block_size"]] for i in ix])
        y = torch.stack([data[i+1:i+configs["block_size"]+1] for i in ix])
        if configs["device"] == 'cuda':
            x, y = x.pin_memory().to(configs["device"], non_blocking=True), y.pin_memory().to(configs["device"], non_blocking=True)
        x, y = x.to(configs["device"]), y.to(configs["device"])
        return x, y

# Single Head of Self Attention
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(configs["n_embd"], head_size, bias=False)
        self.query = nn.Linear(configs["n_embd"], head_size, bias=False)
        self.value = nn.Linear(configs["n_embd"], head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(configs["block_size"], configs["block_size"])))
        self.dropout = nn.Dropout(configs["dropout"])
        
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out
        
# Multi Head Self Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(configs["n_embd"], configs["n_embd"])
        self.dropout = nn.Dropout(configs["dropout"])

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        #out = self.proj(x) #adding the residual connection. Not converging as expected. Will investigate further.
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    def gelu(self, x): #activation function used by OpenAI
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
    
    def __init__(self):
        super().__init__()
        #self.c_fc = nn.Linear(configs["n_embd"], 4 * configs["n_embd"]) #embeddings mul by 4 as in paper again.
        #self.c_proj = nn.Linear(4 * configs["n_embd"], configs["n_embd"]) #residual connection
        #self.dropout = nn.Dropout(configs["dropout"])
        self.net = nn.Sequential(
            nn.Linear(configs["n_embd"], 4 * configs["n_embd"]), #embeddings mul by 4 as in paper again.
            nn.ReLU(),
            nn.Linear(4 * configs["n_embd"], configs["n_embd"]), #residual connection
            nn.Dropout(configs["dropout"])
        )
    
    def forward(self, x):
        return self.net(x)
        #x = self.c_fc(x)
        #x = self.gelu(x) #not converging as expected. Using ReLU until further investigation
        #x = self.c_proj(x)
        #x = self.dropout(x)
        #return x

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = configs["n_embd"] // configs["num_heads"]
        self.multihead = MultiHeadAttention(configs["num_heads"], head_size)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(configs["n_embd"])
        self.ln2 = nn.LayerNorm(configs["n_embd"])
    
    def forward(self, x):
        #compute phase & communicate phase
        x = x + self.multihead(self.ln1(x)) #added the residual connection as in the paper. Added the LayerNorm
        x = x + self.ffwd(self.ln2(x)) #added the residual connection as in the paper. Added LayerNorm
        return x
        
class BitsyGPT(nn.Module):
    def timeit(func):
        @wraps(func)
        def timeit_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            total_time = end_time - start_time
            total_time_str = f'{total_time:.4f}'
            print(f'Function {func.__name__}{args} {kwargs} Took {total_time_str} seconds')
            return result, total_time_str
        return timeit_wrapper
    
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.token_embedding_table = nn.Embedding(data.vocab_size, configs["n_embd"])
        self.position_embedding_table = nn.Embedding(configs["block_size"], configs["n_embd"])
        #self.sa_head = Head(configs["n_embd"]) #single self attention head
        #self.sa_heads = MultiHeadAttention(configs["num_heads"], configs["n_embd"]//configs["num_heads"])
        #self.ffwd = FeedForward() #feed forward - blue color node in the self attention block in the the paper
        self.blocks = nn.Sequential(*[Block() for _ in range(configs["n_layer"])])
        self.layer_norm = nn.LayerNorm(configs["n_embd"])
        #self.blocks = nn.Sequential(
        #    Block(),
        #    Block(),
        #    Block(),
        #    nn.LayerNorm(configs["n_embd) #Added LayerNorm as in the paper.
        #)
        self.lm_head = nn.Linear(configs["n_embd"], data.vocab_size) #decoder's language modelling head
        
    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) #Batch, Token, Channels
        pos_emb = self.position_embedding_table(torch.arange(T, device=configs["device"]))
        x = tok_emb + pos_emb
        x = self.blocks(x) #feed to self attention head first (single/multi)
        #x = self.ffwd(x) #ffwd is moved into the Block. Don't have to have this here.
        x = self.layer_norm(x)
        logits = self.lm_head(x) #Batch, Token, #Vocab_size
        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    @timeit
    def perform_training(self):
        print(f"Training Started. Max Iterations : {configs['max_iter']}, Number of Parameters : {self.get_num_params()}")
        optimizer = torch.optim.AdamW(self.parameters(), configs["learning_rate"], 
                                      (configs['beta1'], configs['beta2'],))
        for iter in range(configs["max_iter"]):
            xb, yb = self.data.get_batch('train')
            logits, loss = self(xb, yb)
            if iter % configs["eval_interval"] == 0:
                losses = self.estimate_loss()
                print(f"Step {iter} train loss : {losses['train']:.4f}, val loss : {losses['val']:.4f}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        losses = self.estimate_loss()
        print(f"Step {iter} train loss : {losses['train']:.4f}, val loss : {losses['val']:.4f}")
        return loss
    
    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(configs["eval_iter"])
            for k in range(configs["eval_iter"]):
                X, Y = self.data.get_batch(split)
                logits, loss = self(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.train()
        return out

    def generate(self, idx, max_token_limit):
        for _ in range(max_token_limit):
            idx_cropped = idx[:, -configs["block_size"]:]
            logits, loss = self(idx_cropped)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim = -1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx =  torch.cat((idx, idx_next), dim=1)
        return self.data.decode(idx[0].tolist())
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def persistModel(self, outpath):
        path = outpath + '/' + configs["model_file"]
        torch.save(self.state_dict(), path)
        return path
    
    def loadModel(self, outpath):
        path = outpath + '/' + configs["model_file"]
        self.load_state_dict(torch.load(path, map_location=configs["device"]))
          
def read_config(config_path):
    config_map = {}
    with open(config_path, 'r') as f:
        config_entries = f.read().splitlines()
    for config in config_entries:
        name, value = config.split('=')
        config_map[name.strip()] = literal_eval(value.strip())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config_map['device'] = device
    if device == 'cuda':
        torch.cuda.set_device(device)
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
        torch.autocast(device_type=device, dtype=ptdtype)
    return config_map

def main():
    args_map = {}
    for arg in sys.argv[1:]:
       if arg.startswith("--"):
           key, val = arg.split('=')
           args_map[key[2:]] = val
    
    globals()["configs"] = read_config(args_map['config'])
    print(configs)
    
    if args_map['mode'] == 'train':
        data = Data()
        model = BitsyGPT(data)
        m = model.to(configs["device"])
        loss = m.perform_training()
        print(f"Number of Parameters : {m.get_num_params()}")
        path = m.persistModel(args_map['out_dir'])
        print(f"Model Saved at {path} ")
    elif args_map['mode'] == 'generate':
        data = Data()
        model = BitsyGPT(data)
        model.loadModel(args_map['out_dir'])
        context=torch.zeros((1,1), dtype=torch.long) 
        print(model.generate(context, int(args_map['max_tokens'])))
        
if __name__ == "__main__":
    main()