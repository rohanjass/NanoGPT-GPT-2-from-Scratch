import torch
import tiktoken

from model import GPT
from config import GPTConfig

# Device

device = "cpu"

# Load configuration and model

cfg = GPTConfig()
model = GPT(cfg)

# Load trained weights

model.load_state_dict(torch.load("ckpt.pt", map_location=device))
model.to(device)
model.eval()

# GPT-2 tokenizer

enc = tiktoken.get_encoding("gpt2")

# Starting prompt

prompt = "ROMEO:"

# Encode prompt

tokens = enc.encode(prompt)
x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

# Generate text

with torch.no_grad():
y = model.generate(
x,
max_new_tokens=300,
temperature=0.8,
top_k=40
)

# Decode and print

generated_text = enc.decode(y[0].tolist())
print("\nGenerated Text:\n")
print(generated_text)
