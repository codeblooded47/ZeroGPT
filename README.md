# 🧠 LLM-Train — Build Your Own GPT from Scratch

> **What is this?** This project lets you build your very own AI language model — the same kind of technology behind ChatGPT — completely from scratch using Python and PyTorch. No magic, no black boxes — just math and code you can read and understand.

📚 **[Beginner's Guide](GUIDE.md)** — Understand datasets, training steps, context length, and parameters in plain English
🧮 **[Technical Blueprint](BLUEPRINT.md)** — Deep dive into the math, formulas, and design decisions behind the model

---

## 📖 What Do All These Words Mean?

Before we start, let's break down the jargon. If you already know these, skip ahead!

| Term                                         | Plain English                                                                                                                                                                      | Example                                                                                       |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| **LLM** (Large Language Model)               | A computer program that has "read" tons of text and learned to predict what word comes next.                                                                                       | You type "The sky is" → the model predicts "blue"                                             |
| **GPT** (Generative Pre-trained Transformer) | A specific design (architecture) for building an LLM. "Generative" = it generates text. "Pre-trained" = trained on lots of data first. "Transformer" = the math technique it uses. | ChatGPT, GPT-4                                                                                |
| **Token**                                    | A piece of a word. Models don't read words — they read tokens.                                                                                                                     | "unhappiness" → `["un", "happiness"]` (2 tokens)                                              |
| **Tokenizer**                                | The tool that chops text into tokens and gives each one a number.                                                                                                                  | "hello world" → `[15339, 1917]`                                                               |
| **Training**                                 | Showing the model millions of sentences so it learns patterns — like a student reading textbooks.                                                                                  | Feed it Wikipedia → it learns grammar, facts, writing style                                   |
| **Parameters**                               | The "knobs" inside the model that get adjusted during training. More parameters = more capacity to learn (but needs more memory).                                                  | Our small model has 41.6 million knobs!                                                       |
| **Loss**                                     | A number that tells us how wrong the model is. Lower = better.                                                                                                                     | Loss 10.0 = terrible, Loss 2.0 = getting there, Loss 1.0 = good                               |
| **Perplexity**                               | How "confused" the model is. It's like asking: "how many words is the model choosing between?"                                                                                     | Perplexity 20 = choosing between ~20 words (good!). Perplexity 32000 = random guessing (bad!) |
| **Epoch**                                    | One full pass through all the training data.                                                                                                                                       | If you have 1000 sentences, 1 epoch = the model has seen all 1000 once                        |
| **Batch**                                    | A small group of training examples processed together (faster than one at a time).                                                                                                 | Batch size 32 = process 32 sentences at once                                                  |
| **GPU**                                      | A powerful chip (originally for gaming graphics) that's great at the math needed for AI.                                                                                           | NVIDIA RTX 4090, Apple M1/M2/M3 chip                                                          |
| **Inference**                                | Using the trained model to generate text (the fun part!).                                                                                                                          | Give it "Once upon a time" → it writes a story                                                |
| **Checkpoint**                               | A saved snapshot of the model during training, so you can resume later or use the best version.                                                                                    | Like saving your game progress                                                                |
| **Overfitting**                              | When the model memorizes the training data instead of learning general patterns. Like a student who memorizes answers but can't solve new problems.                                | Model produces perfect training text but garbage on new prompts                               |
| **Learning Rate**                            | How big of a step the model takes when adjusting its parameters. Too big = overshoots, too small = never learns.                                                                   | Like adjusting volume — you turn it up slowly, then fine-tune                                 |

---

## 🖥️ What You Need (Prerequisites)

### Hardware

- **Minimum**: Any modern computer with 8GB RAM (training will be slow but works)
- **Recommended**: A computer with an NVIDIA GPU (8GB+ VRAM) or Apple M1/M2/M3 Mac
- **Ideal**: NVIDIA GPU with 24GB+ VRAM (like RTX 3090/4090)

### Software

- **Python 3.9+** — the programming language
- **pip** — Python's package installer (comes with Python)
- That's it! Everything else gets installed automatically.

### How to Check

Open your terminal (on Mac: search for "Terminal" in Spotlight) and type:

```bash
python3 --version
```

You should see something like `Python 3.9.6` or higher. If not, download Python from [python.org](https://python.org).

---

## 🚀 Step-by-Step Setup (Do This First!)

### Step 1: Open Your Terminal

- **Mac**: Press `Cmd + Space`, type "Terminal", press Enter
- **Windows**: Press `Win + R`, type "cmd", press Enter
- **Linux**: Press `Ctrl + Alt + T`

### Step 2: Navigate to the Project

```bash
llm-train
```

### Step 3: Activate the Virtual Environment

A **virtual environment** is like a clean room for your project — it keeps all the special tools (libraries) separate from the rest of your computer so nothing gets messy.

```bash
source venv/bin/activate
```

You'll see `(venv)` appear at the beginning of your terminal line. This means you're "inside" the virtual environment. **Always do this before running any commands.**

> 💡 **To leave the virtual environment later**: just type `deactivate`

If the venv doesn't exist yet (you'll get an error), create it first:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 📥 Step-by-Step: Download the Training Data

The model needs text to learn from — just like a student needs textbooks. We'll download **WikiText-103**, which is ~500MB of clean Wikipedia articles.

### Option A: Automatic Download (Easiest)

First, make sure the `datasets` library is installed:

```bash
pip install datasets
```

Then download WikiText-103:

```bash
python -c "
from data.download import download_wikitext103
train_file = download_wikitext103('data/raw')
print(f'Downloaded to: {train_file}')
"
```

**What just happened?** You downloaded ~500MB of Wikipedia text from HuggingFace. This is what the model will "read" to learn English. Two files are created:

- `data/raw/wiki.train.raw` — training data
- `data/raw/wiki.valid.raw` — validation data (used to check if the model is learning)

### Option B: Manual Download via HuggingFace

1. Go to: `https://huggingface.co/datasets/Salesforce/wikitext`
2. Select the `wikitext-103-raw-v1` config
3. Download the train and validation splits
4. Save them as `data/raw/wiki.train.raw` and `data/raw/wiki.valid.raw`

### How Much Data Do You Need?

| Data Size                  | Good For               | Training Time (1 GPU) |
| -------------------------- | ---------------------- | --------------------- |
| 500MB (WikiText-103)       | Learning & prototyping | 2-6 hours             |
| 2-5GB (OpenWebText subset) | Decent quality model   | 12-48 hours           |
| 10GB+                      | Good quality model     | Days to weeks         |

> 💡 **Rule of thumb**: You need about **20 tokens per parameter**. Our small model (41.6M params) ideally wants ~800M tokens ≈ 3-4 GB of text.

---

## 🔤 Step-by-Step: Train the Tokenizer

Before the model can read text, we need to build a **tokenizer** — a tool that converts text into numbers the model understands.

**Think of it like this**: The model can't read English. It can only understand numbers. So we create a dictionary that maps text chunks to numbers.

```
"Hello world" → [2, 72, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100, 3]
                 ^                    the actual text bytes                  ^
                BOS                                                        EOS
               (Begin)                                                    (End)
```

### Train the Tokenizer

We have two options. **Option A (Recommended)** uses Google's SentencePiece — it's fast (~30 seconds) and battle-tested. **Option B** uses our pure Python BPE implementation — slower but you can read every line of code.

#### Option A: SentencePiece (Recommended — ~30 seconds)

```bash
pip install sentencepiece
python -c "
import sentencepiece as spm
spm.SentencePieceTrainer.train(
    input='data/raw/wiki.train.raw',
    model_prefix='tokenizer/sp_bpe',
    vocab_size=32000,
    model_type='bpe',
    byte_fallback=True,
)
print('Done!')
"
```

This produces `tokenizer/sp_bpe.model` — the file all other scripts will use.

> 💡 This is what LLaMA, Mistral, and most production models use. It's the same BPE algorithm — just 100x faster because it's written in C++.

#### Option B: Pure Python BPE (Educational — ~5-30 minutes)

If you want to see the BPE algorithm working step by step:

```bash
python tokenizer/train_tokenizer.py \
  --input data/raw/wiki.train.raw \
  --vocab-size 32000 \
  --max-chars 50000000 \
  --output tokenizer/bpe_tokenizer.json
```

> 💡 `--max-chars 50000000` uses only the first 50M characters. This is enough data to learn good English subwords and keeps training under 10 minutes.

**What's happening?** The tokenizer reads all the text and learns which letter combinations appear most often. For example, "th" appears very often in English, so it becomes one token instead of two.

> 💡 **What's `--vocab-size 32000`?** This is the size of the tokenizer's dictionary — it will learn 32,000 "word pieces". Bigger vocabulary = shorter sequences but more memory usage. 32K is the sweet spot that most real models use.

### Test the Tokenizer

```bash
python tokenizer/sp_wrapper.py tokenizer/sp_bpe.model
```

You should see `[PASS]` for each test sentence. This verifies that encoding text → numbers → decoding back gives the original text.

---

## 📊 Step-by-Step: Prepare the Data

Now we convert the raw text into a format the model can consume efficiently — a binary file of token numbers.

```bash
python -c "
from tokenizer.sp_wrapper import load_tokenizer
from data.dataset import tokenize_and_save

# Load our trained tokenizer (auto-detects .model or .json)
tok = load_tokenizer('tokenizer/sp_bpe.model')

# Tokenize the training data (converts text → numbers → binary file)
tokenize_and_save(
    'data/raw/wiki.train.raw',
    'data/processed/train.bin',
    tok
)

# Tokenize validation data (used to check if the model is learning)
tokenize_and_save(
    'data/raw/wiki.valid.raw',
    'data/processed/val.bin',
    tok
)
"
```

**What just happened?** We converted the Wikipedia text from human-readable words into a file full of numbers. The model will now read this number file during training. This is much faster than re-tokenizing text every time.

---

## 🏋️ Step-by-Step: Train the Model

This is the big one! We're going to teach the model to predict the next word. It does this millions of times until it gets good at it.

### Quick Test First (5 minutes)

Before spending hours training, let's make sure everything works:

```bash
python training/train.py --config small --sanity
```

**What this does**: Creates a tiny fake dataset and trains for 100 steps. You should see:

- `loss` numbers going **down** (the model is learning!)
- `PASS: Loss is decreasing` at the end

If you see this, everything is working! 🎉

### Real Training

```bash
# Small model (~42M params) — good for learning, trains in 2-6 hours
python training/train.py \
  --config small \
  --data-path data/processed/train.bin \
  --val-path data/processed/val.bin

# Medium model (~110M params) — better quality, needs more GPU memory
python training/train.py \
  --config medium \
  --data-path data/processed/train.bin \
  --val-path data/processed/val.bin
```

### What You'll See During Training

```
step    100 | loss 9.8234 | lr 3.00e-04 | grad_norm=1.24 | 5.2K tok/s
step    200 | loss 8.1456 | lr 6.00e-04 | grad_norm=0.91 | 5.5K tok/s
step    500 | loss 6.3421 | lr 5.97e-04 | grad_norm=0.78 | 5.4K tok/s
  >>> Eval: val_loss=6.5123, ppl=672.34
```

**Reading the output**:

- `step 100` — how many training steps have happened (each step = one batch of data)
- `loss 9.82` — how wrong the model is (starts high ~10, should go down to ~3-5)
- `lr 3.00e-04` — learning rate (how fast the model adjusts, changes automatically)
- `grad_norm=1.24` — gradient size (should stay between 0.1 and 10; if it's NaN or huge, something is wrong)
- `5.2K tok/s` — speed (tokens processed per second)
- `val_loss=6.51` — performance on data the model has **never seen** (this is the true measure of quality)
- `ppl=672.34` — perplexity (should go down; under 100 = decent, under 30 = good)

### How Long Will Training Take?

| Config | GPU         | WikiText-103 | Expected Final PPL |
| ------ | ----------- | ------------ | ------------------ |
| Small  | Apple M1/M2 | ~4-8 hours   | 40-60              |
| Small  | RTX 3090    | ~2-4 hours   | 40-60              |
| Medium | RTX 3090    | ~6-12 hours  | 25-40              |
| Medium | A100        | ~2-4 hours   | 25-40              |

> 💡 **Can I stop and resume?** Yes! The trainer saves checkpoints automatically to `checkpoints/`. To resume training from where you left off:
>
> ```bash
> # Resume from the latest periodic checkpoint
> python training/train.py \
>   --config small \
>   --data-path data/processed/train.bin \
>   --val-path data/processed/val.bin \
>   --resume checkpoints/checkpoint_step5000.pt
>
> # Or resume from the best checkpoint (lowest validation loss)
> python training/train.py \
>   --config small \
>   --data-path data/processed/train.bin \
>   --val-path data/processed/val.bin \
>   --resume checkpoints/best.pt
> ```
>
> This restores the model weights, optimizer state, learning rate schedule, and step counter — training continues exactly where it stopped.

---

## 📝 Step-by-Step: Generate Text (Use Your Model!)

The model is trained — now let's make it write! This is called **inference**.

```bash
python -c "
import torch
from config import small_config
from model.transformer import GPT
from tokenizer.sp_wrapper import load_tokenizer
from evaluation.sampling import generate_text

# 1. Load the tokenizer (auto-detects .model or .json)
tok = load_tokenizer('tokenizer/sp_bpe.model')

# 2. Load the model
model_cfg, _ = small_config()
model = GPT(model_cfg)

# 3. Load the trained weights (the 'brain' of the model)
checkpoint = torch.load('checkpoints/best.pt', map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint['model_state'])
print(f'Model loaded from step {checkpoint[\"step\"]}')

# 4. Generate text!
prompt = 'The history of artificial intelligence'
output = generate_text(
    model=model,
    tokenizer=tok,
    prompt=prompt,
    max_new_tokens=200,     # How many new words to generate
    temperature=0.8,        # Creativity: 0.1=boring/safe, 1.0=creative, 1.5=wild
    top_k=50,               # Only consider top 50 most likely next words
    top_p=0.9,              # Nucleus sampling: consider words until 90% probability covered
)
print(output)
"
```

### Sampling Parameters Explained

Think of the model as a writer choosing the next word. These settings control how it chooses:

| Parameter         | What It Does                                             | Analogy                                           |
| ----------------- | -------------------------------------------------------- | ------------------------------------------------- |
| `temperature=0.1` | Very safe, repetitive, always picks the most likely word | A boring writer who always uses the same phrases  |
| `temperature=0.8` | Good balance of creativity and coherence                 | A skilled writer (recommended!)                   |
| `temperature=1.5` | Wild, unpredictable, might ramble                        | A writer who had too much coffee                  |
| `top_k=50`        | Only consider the 50 most likely next words              | A writer with a vocabulary of 50 options per word |
| `top_p=0.9`       | Keep words until their total probability reaches 90%     | A writer who considers all reasonable options     |
| `top_p=0.5`       | Only consider words in the top 50% of probability        | A writer who plays it very safe                   |

---

## 🌐 Step-by-Step: Run as a Web Server (Optional)

Want to make your model accessible via a web API — like a mini ChatGPT server?

```bash
# Start the server
python inference/server.py
```

Then in another terminal (or browser):

```bash
# Test it
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Once upon a time", "max_new_tokens": 100, "temperature": 0.8}'
```

Or open your browser and go to `http://localhost:8000/docs` — you'll see an interactive API page where you can type prompts and see responses!

### Run with Docker (Optional)

**Docker** is a tool that packages your app into a container — like a shipping container for software. It makes deployment easy.

```bash
docker build -t my-gpt -f inference/Dockerfile .
docker run -p 8000:8000 -v ./checkpoints:/app/checkpoints my-gpt
```

---

## 🧪 Step-by-Step: Run Tests

Tests verify that the model code is correct. Think of them as a "health check" for your code.

```bash
python -m pytest evaluation/tests.py -v
```

You should see:

```
TestOutputShape::test_logits_shape            PASSED ✅
TestOutputShape::test_loss_is_scalar          PASSED ✅
TestCausalMask::test_mask_blocks_future       PASSED ✅
TestOverfit::test_loss_decreases              PASSED ✅
TestGradients::test_gradients_exist           PASSED ✅
TestTokenizer::test_encode_decode_roundtrip   PASSED ✅
... (11 tests total)
```

If any test says `FAILED`, something is wrong — the error message will tell you what.

---

## 📁 Project Map (What's Each File For?)

```
llm-train/
│
├── README.md              ← 📍 You are here! The guide.
├── BLUEPRINT.md           ← 🧮 Deep technical reference (math, theory, formulas)
├── config.py              ← ⚙️  Model settings (size, dimensions, training params)
├── requirements.txt       ← 📦 List of required Python packages
├── venv/                  ← 🏠 Virtual environment (your isolated Python setup)
│
├── tokenizer/             ← 🔤 Text ↔ Numbers conversion
│   ├── bpe.py             ←    Pure Python BPE tokenizer (educational)
│   ├── sp_wrapper.py      ←    SentencePiece wrapper (recommended, fast)
│   └── train_tokenizer.py ←    Script to train the pure Python tokenizer
│
├── data/                  ← 📊 Data handling
│   ├── download.py        ←    Download datasets (WikiText, OpenWebText, etc.)
│   └── dataset.py         ←    Load tokenized data efficiently for training
│
├── model/                 ← 🧠 The transformer model
│   ├── attention.py       ←    Self-attention mechanism (the core of transformers)
│   ├── rope.py            ←    Positional encoding (tells the model word order)
│   └── transformer.py     ←    The complete GPT model (ties everything together)
│
├── training/              ← 🏋️ Training pipeline
│   ├── trainer.py         ←    The training loop (optimizer, scheduling, logging)
│   └── train.py           ←    Run this to start training!
│
├── evaluation/            ← 📊 Testing and text generation
│   ├── metrics.py         ←    Calculate validation loss and perplexity
│   ├── sampling.py        ←    Text generation strategies (greedy, top-k, top-p)
│   └── tests.py           ←    Unit tests to verify correctness
│
├── inference/             ← 🚀 Using the trained model
│   ├── generate.py        ←    Fast text generation with KV caching
│   ├── server.py          ←    Web API server (FastAPI)
│   └── Dockerfile         ←    Package as a Docker container
│
├── scripts/               ← 🔧 Utilities
│   └── count_params.py    ←    Count model parameters
│
├── checkpoints/           ← 💾 Saved model weights (created during training)
└── logs/                  ← 📋 Training logs (created during training)
```

---

## ❓ Troubleshooting

### "No module named torch" or similar errors

You forgot to activate the virtual environment:

```bash
source venv/bin/activate
```

### "CUDA out of memory"

Your GPU doesn't have enough memory. Try:

1. Reduce batch size in `config.py` (e.g., `batch_size=8`)
2. Use the `small` config instead of `medium`
3. Enable gradient checkpointing (trades speed for memory)

### Loss is NaN (Not a Number)

Something went wrong numerically. Try:

1. Reduce learning rate (e.g., `learning_rate=1e-4`)
2. Make sure your data doesn't contain garbage characters
3. Check that gradient clipping is enabled (`max_grad_norm=1.0`)

### Model generates garbage text

This is normal at the start! The model needs thousands of training steps to produce coherent text. Train longer and make sure `val_loss` is decreasing.

### `killed` or process crashes

Your computer ran out of RAM. Try:

1. Use smaller batch size
2. Use shorter sequence length (`max_seq_len=128`)
3. Use the `small` config

---

## 🎓 Want to Learn More?

- **BLUEPRINT.md** — The deep technical reference in this project. Has all the math, formulas, and design decisions explained.
- **Attention Is All You Need** (Vaswani et al., 2017) — The original transformer paper
- **Language Models are Unsupervised Multitask Learners** (GPT-2 paper) — How GPT-2 works
- **Andrej Karpathy's "Let's build GPT"** — Excellent YouTube walkthrough

---

## 🔑 The Complete Flow (TL;DR)

```
1. Download text data (Wikipedia)
         ↓
2. Train tokenizer (learn to chop text into tokens)
         ↓
3. Tokenize data (convert text files → number files)
         ↓
4. Train model (show the model millions of examples)
         ↓
5. Generate text (give it a prompt, get a response!)
         ↓
6. (Optional) Serve via API (make it accessible over the web)
```

**That's it!** You've built a language model from scratch. Welcome to the world of AI! 🚀
