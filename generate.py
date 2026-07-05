import torch, tiktoken
from model import GPT
from config import GPTConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
enc    = tiktoken.get_encoding("gpt2")

model = GPT(GPTConfig())
model.load_state_dict(torch.load("ckpt.pt", map_location=device))
model.eval().to(device)

prompt = "ROMEO:"
ctx    = torch.tensor(enc.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
out    = model.generate(ctx, max_new_tokens=500, temperature=0.8, top_k=200)
print(enc.decode(out[0].tolist()))