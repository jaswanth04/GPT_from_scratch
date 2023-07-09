import torch
import torch.nn as nn
from torch.nn import functional as F

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        logits = self.token_embedding_table(idx)

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
            # Predictions for a given index
            logits, _ = self(idx)

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
    block_size = 8
    batch_size = 32
    num_epochs = 25000
    learning_rate = 1e-3
    max_new_tokes = 10000
    device = torch.device('cuda')
    eval_iters = 2000

    m = BigramLanguageModel(vocab_size).to(device)
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
    context = torch.zeros((1, 1), dtype=torch.long).to(device)
    print("Sample generation !")
    print(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))

    
if __name__ == '__main__':
    main()
