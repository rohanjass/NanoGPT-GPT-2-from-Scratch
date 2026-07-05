import torch
import time
from model import GPT
from config import GPTConfig
from data import prepare, get_batch

# ── config ──────────────────────────────────────────────
device     = "cuda" if torch.cuda.is_available() else "cpu"
dtype      =  torch.float32
batch_size = 8
block_size = 128
max_iters  = 5000
eval_every = 250
lr         = 6e-4
weight_decay = 1e-1

# GPT-2 small — reduce for CPU: n_layer=6, n_head=6, n_embd=384
cfg = GPTConfig()

# ── data ────────────────────────────────────────────────
prepare()

# ── model ───────────────────────────────────────────────
model = GPT(cfg).to(device)
print(f"Parameters: {model.num_params()/1e6:.1f}M")

optimizer = model.configure_optimizer(weight_decay, lr, device) \
    if hasattr(model, "configure_optimizer") else \
    torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))



# ── training loop ───────────────────────────────────────
@torch.no_grad()
def estimate_loss(iters=20):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = []
        for _ in range(iters):
            x, y = get_batch(split, block_size, batch_size, device)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out

t0 = time.time()
for step in range(max_iters):
    if step % eval_every == 0:
        losses = estimate_loss()
        print(f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | "
              f"{(time.time()-t0):.1f}s")
        torch.save(model.state_dict(), "ckpt.pt")

    x, y = get_batch("train", block_size, batch_size, device)
    _, loss = model(x, y)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)