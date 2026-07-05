import os, requests
import numpy as np
import tiktoken

def prepare(data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "shakespeare.txt")
    if not os.path.exists(path):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with open(path, "w") as f:
            f.write(requests.get(url).text)
        print("Downloaded Shakespeare.")

    text  = open(path).read()
    enc   = tiktoken.get_encoding("gpt2")
    ids   = np.array(enc.encode_ordinary(text), dtype=np.uint16)
    split = int(0.9 * len(ids))
    ids[:split].tofile(os.path.join(data_dir, "train.bin"))
    ids[split:].tofile(os.path.join(data_dir, "val.bin"))
    print(f"Train tokens: {split:,}  Val tokens: {len(ids)-split:,}")

def get_batch(split, block_size, batch_size, device, data_dir="data"):
    data = np.memmap(
        os.path.join(data_dir, f"{split}.bin"), dtype=np.uint16, mode='r'
    )
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x  = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y  = torch.stack([torch.from_numpy(data[i+1:i+block_size+1].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

import torch  # needed by get_batch