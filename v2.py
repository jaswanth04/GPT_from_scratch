import torch
import torch.nn as nn
from torch.nn import functional as F

device = torch.device("cuda")

class Head(nn.Module):
    def __init__(self, block_size, num_embed, head_size, dropout) -> None:
        super().__init__()
        self.key = nn.Linear(num_embed, head_size, bias=False)
        self.query = nn.Linear(num_embed, head_size, bias=False)
        self.value = nn.Linear(num_embed, head_size, bias=False)
        self.register_buffer('trill', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1) * (C**-0.5)

        wei = wei.masked_fill(self.trill[:T, :T] == 0, float('-inf')) 
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        out = wei @ v

        return out

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size, block_size, num_embed, dropout) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(block_size=block_size,
                                         num_embed=num_embed,
                                         head_size=head_size, 
                                         dropout=dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(num_embed, num_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))

        return out

class FeedForward(nn.Module):
    def __init__(self, num_embed, dropout) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(num_embed, 4 * num_embed),
                                 nn.ReLU(),
                                 nn.Linear(4 * num_embed, num_embed), 
                                 nn.Dropout(dropout))
        
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, num_embed, num_heads, block_size, dropout) -> None:
        super().__init__()
        head_size = num_embed // num_heads

        self.sa = MultiHeadAttention(num_heads=num_heads, 
                                     head_size=head_size, 
                                     block_size=block_size, 
                                     num_embed=num_embed,
                                     dropout=dropout)
        self.ffd = FeedForward(num_embed=num_embed, 
                               dropout=dropout)

        self.ln1 = nn.LayerNorm(num_embed)
        self.ln2 = nn.LayerNorm(num_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffd(self.ln2(x))
        
        return x

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size, num_embed, block_size, num_layers, dropout) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, num_embed)
        self.pos_embedding_table = nn.Embedding(block_size, num_embed)
        self.lm_head = nn.Linear(num_embed, vocab_size)
        self.block_size = block_size
        # self.ffd = FeedForward(num_embed=num_embed)

        # 4 heads of 8 dimensional self-attention
        # self.sa_head = MultiHeadAttention(block_size=block_size, num_embed=num_embed, head_size=num_embed//4, num_heads=4)
        # self.block = Block(num_embed=num_embed, num_heads=4, block_size=block_size)

        self.blocks = nn.Sequential(*[Block(num_embed=num_embed, 
                                            num_heads=4, 
                                            block_size=block_size,
                                            dropout=dropout) for _ in range(num_layers)],
                                            nn.LayerNorm(num_embed))

        # self.blocks = nn.Sequential(
        #     Block(num_embed=num_embed, num_heads=4, block_size=block_size),
        #     Block(num_embed=num_embed, num_heads=4, block_size=block_size),
        #     Block(num_embed=num_embed, num_heads=4, block_size=block_size),
        #     nn.LayerNorm(num_embed)
        # )

    def forward(self, idx, targets=None):

        B, T = idx.shape

        token_embedding = self.token_embedding_table(idx) # (B, T, C)
        pos_embed = self.pos_embedding_table(torch.arange(T, device=device))

        x = token_embedding + pos_embed
        # x = self.sa_head(x)
        # x = self.ffd(x)
        x = self.blocks(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            # Changing the shape so that the cross entropy function can understand
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            # Implement the negative log likelihood
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):

            # crop idx to last block size tokens
            idx_cond = idx[:, -self.block_size:]

            # Predictions for a given index
            logits, _ = self(idx_cond)

            # We only use the last time step
            logits = logits[:, -1, :] # dimension is B, C
            probs = F.softmax(logits, dim=-1)

            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1) # B, 1

            # append sample index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx




def main():

    torch.manual_seed(1337)
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    # Encoder and decoder
    stoi = {ch:i for i, ch in enumerate(chars)}
    itos = {i:ch for i, ch in enumerate(chars)}

    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    data = torch.tensor(encode(text), dtype=torch.long)


    # separate the data to train and val split
    n = int(0.9*len(data))
    train = data[:n]
    val = data[n:]

    def get_batch(split, batch_size=4, block_size=8):
        data = train if split == 'train' else val
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i: i+block_size] for i in ix])
        y = torch.stack([data[i+1: i+block_size+1] for i in ix])
        return x, y


    

    # Hyper parameters
    block_size = 256
    batch_size = 64
    num_epochs = 25000
    learning_rate = 3e-4
    max_new_tokens = 10000
    device = torch.device('cuda')
    eval_iters = 2000
    num_embed = 384
    num_layers = 6
    dropout = 0.2

    m = BigramLanguageModel(vocab_size, num_embed, block_size, num_layers, dropout).to(device)
    optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
    
    @torch.no_grad()
    def estimate_loss():
        out = {}
        m.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                X, Y = X.to(device), Y.to(device)
                logits, loss = m(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        m.train()
        return out

    # Training
    print('Training started')
    for epoch in range(num_epochs):

        if epoch % eval_iters == 0:
            losses = estimate_loss()
            print(f'epoch: {epoch}, train loss: {losses["train"]}, val loss: {losses["val"]}')

        # sample batch 
        xb, yb = get_batch('train', batch_size=batch_size, block_size=block_size)
        xb = xb.to(device)
        yb = yb.to(device)

        # evaluate loss
        logits, loss = m(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # print('Training loss: ', loss.item())

    # Sample Generation
    with torch.no_grad():
        m.eval()
        context = torch.zeros((1, 1), dtype=torch.long).to(device)
        print("Sample generation !")
        print(decode(m.generate(context, max_new_tokens)[0].tolist()))

    
if __name__ == '__main__':
    main()
