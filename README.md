# NanoGPT From Scratch (PyTorch)

A lightweight GPT-style Transformer Language Model built from scratch using **PyTorch**. The model is trained on the Tiny Shakespeare dataset and implements the core components of the GPT architecture, including multi-head self-attention, positional embeddings, causal masking, and autoregressive text generation.

---

## Features

* GPT-style Decoder-only Transformer
* Multi-Head Self-Attention
* Token & Positional Embeddings
* Causal (Masked) Self-Attention
* Feed Forward Network (MLP)
* Residual Connections
* Layer Normalization
* Weight Tying
* Model Checkpointing
* Autoregressive Text Generation

---

## Tech Stack

* Python
* PyTorch
* tiktoken
* NumPy

---

## Project Structure

```text
nanoGPT/
│── data/
│── ckpt.pt
│── config.py
│── data.py
│── model.py
│── train.py
│── generate.py
│── README.md
```

---

## Model Configuration

| Hyperparameter  | Value |
| --------------- | ----: |
| Layers          |     4 |
| Attention Heads |     4 |
| Embedding Size  |   128 |
| Context Length  |   128 |
| Vocabulary Size | 50257 |
| Parameters      | ~7.2M |

---

## Dataset

* Tiny Shakespeare
* Training Tokens: **304,222**
* Validation Tokens: **33,803**

---

## Installation

```bash
git clone https://github.com/rohanjass/nanoGPT.git
cd nanoGPT

pip install torch tiktoken numpy
```

---

## Training

```bash
python train.py
```

The trained model checkpoint is saved as:

```text
ckpt.pt
```

---

## Generate Text

```bash
python generate.py
```

Example prompt:

```text
ROMEO:
```

Example output:

```text
ROMEO:
I shall he shall not, our thanks.

ROMEO:
And, God-for so, madam, I can keep
Sir, my life?

JULIET:
What dost thou swear thou!
```

---

## Architecture

* Token Embeddings
* Positional Embeddings
* Multi-Head Causal Self-Attention
* Feed Forward Network
* Residual Connections
* Layer Normalization
* Language Modeling Head

---

## Future Improvements

* Train larger GPT models
* GPU & Mixed Precision Training
* Flash Attention
* Fine-tuning on custom datasets
* Hugging Face integration
* Web interface for text generation
