# 📘 Beginner's Guide: Understanding LLM Training

> This guide explains key concepts you'll encounter when training your own language model.
> Written for people who are completely new to AI/ML.

---

## Table of Contents

1. [How to Create Your Own Dataset](#-how-to-create-your-own-dataset)
2. [Training Steps: How Long to Train](#-training-steps-how-long-to-train)
3. [Context Length: The Model's Memory](#-context-length-the-models-memory)
4. [Understanding Model Output Quality](#-understanding-model-output-quality)
5. [Parameters: What Are They?](#-parameters-what-are-they)

---

## 📊 How to Create Your Own Dataset

### What is a "dataset"?

A dataset is simply **a big text file** that the model reads to learn. Think of it like a textbook for a student:

- If you give it Wikipedia → it learns to write encyclopedia articles
- If you give it novels → it learns to write stories
- If you give it Python code → it learns to write code
- If you give it chat logs → it learns to converse

**The model becomes what it reads.** This is the most important thing to understand.

### What format does the text need to be?

Just **plain text** in a `.txt` file. No special format. No JSON. No CSV. Just text:

```
This is the first document. It could be an article about dogs.
Dogs are wonderful companions that have been domesticated for
thousands of years. There are hundreds of breeds, from tiny
Chihuahuas to massive Great Danes.

This is the second document. Maybe it's about cooking.
The key to a great pasta sauce is using fresh tomatoes and
letting it simmer for at least 30 minutes. Add basil at the
very end to preserve its flavor.

And so on...
```

Documents are separated by blank lines. That's it!

### Where to get text

Here are some sources, from easiest to most advanced:

#### 1. Books (Public Domain)

```bash
# Download "Pride and Prejudice" from Project Gutenberg
curl https://www.gutenberg.org/files/1342/1342-0.txt > data/raw/my_data.txt

# Download more books and combine them
curl https://www.gutenberg.org/files/84/84-0.txt >> data/raw/my_data.txt   # Frankenstein
curl https://www.gutenberg.org/files/11/11-0.txt >> data/raw/my_data.txt   # Alice in Wonderland
```

> 💡 `>` creates a new file. `>>` appends to an existing file.

#### 2. Your own writing

If you have blogs, essays, notes, or any text files:

```bash
# Combine all your .txt files into one
cat ~/Documents/*.txt > data/raw/my_data.txt

# Copy content from a Word doc (save as .txt first)
# Copy from Google Docs (File → Download → Plain Text)
```

#### 3. Code repositories

```bash
# Combine all Python files from a project
find ~/my-project -name "*.py" -exec cat {} + > data/raw/code_data.txt
```

#### 4. Web scraping

```bash
# Simple example: extract text from a webpage
curl https://en.wikipedia.org/wiki/Machine_learning | \
  python -c "import sys; from html.parser import HTMLParser; \
  # (use a real tool like trafilatura for proper web scraping)
"

# Better: use the 'trafilatura' library
pip install trafilatura
python -c "
import trafilatura
url = 'https://en.wikipedia.org/wiki/Machine_learning'
downloaded = trafilatura.fetch_url(url)
text = trafilatura.extract(downloaded)
with open('data/raw/my_data.txt', 'w') as f:
    f.write(text)
print(f'Saved {len(text)} characters')
"
```

#### 5. Existing datasets (HuggingFace)

```bash
pip install datasets
python -c "
from datasets import load_dataset

# Load any text dataset from HuggingFace
# Browse: https://huggingface.co/datasets?task_categories=text-generation
ds = load_dataset('roneneldan/TinyStories')  # Simple children's stories

with open('data/raw/my_data.txt', 'w') as f:
    for example in ds['train']:
        f.write(example['text'] + '\n\n')
print('Done!')
"
```

### How much text do you need?

| Dataset Size              | Characters  | Tokens (~4 chars/token) | Quality You'll Get                                 |
| ------------------------- | ----------- | ----------------------- | -------------------------------------------------- |
| **1 MB**                  | 1,000,000   | ~250,000                | Very basic. Model babbles. Good for testing only.  |
| **10 MB**                 | 10,000,000  | ~2,500,000              | Learns basic grammar. Short coherent phrases.      |
| **50 MB**                 | 50,000,000  | ~12,500,000             | Decent sentence structure. Starting to make sense. |
| **100 MB**                | 100,000,000 | ~25,000,000             | Good grammar. Paragraphs start to flow.            |
| **500 MB** (WikiText-103) | 500,000,000 | ~125,000,000            | Coherent multi-sentence output.                    |
| **1-5 GB**                | 1-5 billion | ~250M-1.25B             | Near-optimal for a 41M param model.                |

> 💡 **Rule of thumb**: You want at least **20× more tokens than parameters**.
> Your model has 41.6M parameters → ideally wants ~832M tokens → ~3.3 GB of text.
> But you can get decent results with much less (WikiText-103 at 500MB works fine).

### Data quality matters more than quantity

**Bad data** = model learns bad habits:

- ❌ HTML tags: `<div class="nav">Click here</div>`
- ❌ JavaScript: `function(){var x=document.getElementById...`
- ❌ Repeated headers: `Home | About | Contact | Privacy Policy`
- ❌ Garbled text: `â€™ Ã© Ã¼` (encoding errors)

**Good data** = clean, natural text:

- ✅ Complete sentences and paragraphs
- ✅ Proper grammar and punctuation
- ✅ Diverse topics and writing styles
- ✅ UTF-8 encoded

### The complete workflow with your own data

```bash
# 1. Put your text file in the data folder
cp ~/my_text.txt data/raw/my_data.txt

# 2. Split into train (90%) and validation (10%)
total_lines=$(wc -l < data/raw/my_data.txt)
train_lines=$((total_lines * 9 / 10))
head -n $train_lines data/raw/my_data.txt > data/raw/train.txt
tail -n +$((train_lines + 1)) data/raw/my_data.txt > data/raw/val.txt

# 3. Train a tokenizer on your data
python -c "
import sentencepiece as spm
spm.SentencePieceTrainer.train(
    input='data/raw/train.txt',
    model_prefix='tokenizer/sp_bpe',
    vocab_size=32000,       # Use 8000-16000 for small datasets
    model_type='bpe',
    byte_fallback=True,
)
"

# 4. Tokenize your data
python -c "
from tokenizer.sp_wrapper import load_tokenizer
from data.dataset import tokenize_and_save
tok = load_tokenizer('tokenizer/sp_bpe.model')
tokenize_and_save('data/raw/train.txt', 'data/processed/train.bin', tok)
tokenize_and_save('data/raw/val.txt', 'data/processed/val.bin', tok)
"

# 5. Train!
python training/train.py \
  --config small \
  --data-path data/processed/train.bin \
  --val-path data/processed/val.bin
```

---

## ⏱ Training Steps: How Long to Train

### What is a "step"?

One training step = the model reads **one batch** of text and adjusts its parameters to get a little better at predicting the next word.

```
Step 1:  Model reads 16,384 tokens → adjusts parameters slightly
Step 2:  Model reads 16,384 tokens → adjusts parameters slightly
Step 3:  Model reads 16,384 tokens → adjusts parameters slightly
...
Step 20,000: Model has seen ~328 million tokens total
```

### How many tokens per step?

```
Tokens per step = batch_size × gradient_accumulation × sequence_length
```

For your current config:

```
16 × 4 × 256 = 16,384 tokens per step
```

### What happens at each stage of training?

| Steps               | Tokens Seen | What the Model Can Do                                    | Loss (approx) |
| ------------------- | ----------- | -------------------------------------------------------- | ------------- |
| 0                   | 0           | Random garbage: `Xq7 ##!@ vmk 4Tz`                       | ~10.4         |
| 100                 | ~1.6M       | Still garbage but with real letters: `the the the is is` | ~10.0         |
| 500                 | ~8M         | Repeating common words: `the world is the world`         | ~9.0          |
| 1,000               | ~16M        | Basic word patterns: `the government of the country`     | ~8.0          |
| 2,000 (you're here) | ~33M        | Simple sentences but repetitive                          | ~7.0          |
| 5,000               | ~82M        | Real sentences, some coherence                           | ~6.0          |
| 10,000              | ~164M       | Decent paragraphs, occasional facts                      | ~5.5          |
| 20,000              | ~328M       | Good grammar, multi-sentence coherence                   | ~5.0          |
| 50,000              | ~820M       | Best quality for this model size                         | ~4.5          |

### What production models train with

| Model             | Parameters | Training Steps | Total Tokens | Training Time  | Cost                |
| ----------------- | ---------- | -------------- | ------------ | -------------- | ------------------- |
| **Your model**    | **41.6M**  | **20,000**     | **~328M**    | **~2-3 hours** | **Free (your Mac)** |
| GPT-2 Small       | 117M       | ~300,000       | ~40B         | ~1 week        | ~$5,000             |
| GPT-2 Large       | 774M       | ~300,000       | ~40B         | ~2 weeks       | ~$25,000            |
| LLaMA-2 7B        | 7B         | ~1,000,000     | 2T           | ~3 months      | ~$500,000           |
| LLaMA-3 8B        | 8B         | ~5,000,000     | 15T          | ~6 months      | ~$2,000,000         |
| GPT-4 (estimated) | ~1.8T      | Unknown        | ~13T         | ~3-4 months    | ~$100,000,000       |

> 💡 **Your 41.6M model is to GPT-4 what a bicycle is to a rocket ship.**
> But the fundamental principle is identical — the same algorithm, just scaled up massively.

### The Chinchilla Rule: How to Know When to Stop

Researchers at DeepMind discovered an optimal ratio:

```
Optimal training tokens ≈ 20 × number of parameters
```

For your model:

```
41.6M parameters × 20 = ~832M tokens
832M tokens ÷ 16,384 tokens/step ≈ 50,000 steps
```

Training **less** than this → model hasn't learned everything it could from the data.
Training **more** than this → diminishing returns (better to make the model bigger instead).

### How to tell if training is going well

Watch these numbers in the training output:

```
step  5000 | loss 6.21 | lr 5.40e-04 | grad_norm=0.53 | 5.4K tok/s
  >>> Eval: val_loss=6.35, ppl=573.12
```

**Good signs** ✅:

- `loss` (training loss) is going **down** over time
- `val_loss` (validation loss) is **also** going down
- `val_loss` is close to `loss` (no overfitting)
- `grad_norm` is between 0.1 and 10.0

**Bad signs** ❌:

- `loss` goes **up** → learning rate might be too high
- `val_loss` goes up while `loss` goes down → **overfitting** (model is memorizing, not learning)
- `loss` becomes `NaN` → **training collapsed** (reduce learning rate)
- `grad_norm` is very large (>100) → **gradient explosion** (reduce learning rate)

---

## 🧠 Context Length: The Model's Memory

### What is context length?

Context length is **how many tokens the model can look at simultaneously** when generating the next word. Think of it as the model's **short-term memory** or **window of attention**.

### A visual example

Imagine the model is writing: _"The cat sat on the mat because it was tired after chasing the mouse around the house all day long."_

With **context length = 10 tokens**, the model can only see:

```
[ chasing | the | mouse | around | the | house | all | day | long | . ]
  ←────────────── model can see this ──────────────→

It has NO MEMORY of "The cat sat on the mat because it was tired"!
```

With **context length = 256 tokens** (~200 words), it can see the entire paragraph — and generate text that's consistent with everything before it.

### Why context matters

**Short context (256 tokens — your model)**:

- Can write a coherent sentence ✅
- Can write a coherent paragraph ✅
- Cannot maintain a story across multiple paragraphs ❌
- Cannot remember something mentioned 300 words ago ❌
- Cannot follow complex multi-step instructions ❌

**Long context (4096+ tokens — GPT-3.5)**:

- Can write coherent multi-page documents ✅
- Can reference things mentioned pages ago ✅
- Can follow complex, detailed prompts ✅
- Can summarize long articles ✅

**Very long context (128K+ tokens — GPT-4 Turbo)**:

- Can read and analyze entire books ✅
- Can compare multiple documents ✅
- Can maintain context across very long conversations ✅

### Context length comparison

| Model          | Context Length | In Words       | In Pages         | Real-World Analogy |
| -------------- | -------------- | -------------- | ---------------- | ------------------ |
| **Your model** | **256 tokens** | **~200 words** | **~half a page** | A sticky note      |
| GPT-2          | 1,024 tokens   | ~800 words     | ~1.5 pages       | A short email      |
| GPT-3          | 2,048 tokens   | ~1,500 words   | ~3 pages         | A short essay      |
| GPT-3.5        | 4,096 tokens   | ~3,000 words   | ~6 pages         | A long article     |
| GPT-4          | 8,192 tokens   | ~6,000 words   | ~12 pages        | A research paper   |
| GPT-4 Turbo    | 128,000 tokens | ~96,000 words  | ~200 pages       | A novel            |
| Claude 3.5     | 200,000 tokens | ~150,000 words | ~300 pages       | Multiple novels    |

### Why can't we just make context length infinite?

The attention mechanism (the core of transformers) compares **every token with every other token**:

```
Context 256:    256 × 256     =         65,536 comparisons
Context 512:    512 × 512     =        262,144 comparisons    (4× more)
Context 1024:   1,024 × 1,024 =      1,048,576 comparisons    (16× more)
Context 4096:   4,096 × 4,096 =     16,777,216 comparisons    (256× more!)
Context 128K:   128K × 128K   = 16,384,000,000 comparisons    (250,000× more!!!)
```

Each of those comparisons takes memory (RAM/VRAM) and compute time.

**Memory formula**:

```
Attention memory ≈ batch_size × num_heads × context_length² × 2 bytes

Your model: 16 × 8 × 256² =          8 MB    ← fits easily
  at 512:   16 × 8 × 512² =         32 MB    ← still fine
  at 2048:  16 × 8 × 2048² =       512 MB    ← getting big
  at 8192:  16 × 8 × 8192² =     8,192 MB    ← 8 GB just for attention!
```

That's why GPT-4 needs clusters of hundreds of GPUs — not just for the parameters, but for the enormous attention matrices.

> 💡 **Techniques like FlashAttention** (explained in BLUEPRINT.md) reduce the memory from O(n²) to O(n) by computing attention in tiles. This is how models like GPT-4 Turbo achieve 128K context without needing impossible amounts of memory.

### Can you change context length?

Yes, but with trade-offs:

| Context Length    | Memory Usage | Training Speed | What Changes                                      |
| ----------------- | ------------ | -------------- | ------------------------------------------------- |
| 128               | Very low     | Very fast      | Short memory, can't track multi-sentence patterns |
| **256 (current)** | **Low**      | **Fast**       | **Good balance for 16GB Mac**                     |
| 512               | Medium       | Slower         | Better coherence, double the memory               |
| 1024              | High         | Much slower    | Great coherence, may OOM on 16GB                  |

To change: edit `max_seq_len` in `config.py`. **But you must retrain from scratch** — the model learns position-specific patterns that don't transfer.

---

## 📈 Understanding Model Output Quality

### Why does my model repeat itself?

At step 2,000 with 256 context, you're seeing output like:

```
"The history of the world is the most important of the history
of the world . The history of the world is the most important..."
```

This is **completely normal**! Here's why:

1. **The model has barely started learning** — 2,000 steps out of 20,000 means it's 10% through training
2. **It has learned that common phrases score well** — "the world", "most important" are very frequent in Wikipedia
3. **It hasn't yet learned diversity** — that comes later with more training

### What to expect at each stage

**Step 2,000** — Repetitive loops of common phrases

```
"the world is the most important the world is the most important"
```

**Step 5,000** — Real words in semi-coherent order

```
"The city was founded in the early 20th century and was the
first of its kind in the region."
```

**Step 10,000** — Coherent sentences with some Wikipedia-like facts

```
"The Battle of Gettysburg was fought in July 1863 during the
American Civil War. It was one of the bloodiest battles of the war."
```

**Step 20,000** — Multi-sentence coherent text

```
"The Amazon rainforest is the largest tropical rainforest in
the world, covering an area of approximately 5.5 million square
kilometers. It is home to an estimated 10% of all species on Earth."
```

> ⚠️ **Important**: Even at 20,000 steps, the model will sometimes "hallucinate" (make up fake facts). This is inherent to how language models work — they learn statistical patterns, not actual knowledge. A 41.6M parameter model will never be as accurate as GPT-4 with 1.8 trillion parameters.

---

## 🔢 Parameters: What Are They?

### The simplest explanation

A **parameter** is a number inside the model that gets adjusted during training. The model is essentially a huge mathematical function with millions of adjustable numbers (knobs).

Before training: all these numbers are random → model outputs garbage.
During training: the numbers get gradually adjusted → model gets better.
After training: the numbers are saved as a "checkpoint" → you can use the model.

### A tiny analogy

Imagine you're trying to learn to predict tomorrow's temperature:

```
temperature = (weight_1 × today's_temp) + (weight_2 × humidity) + bias

At first:    weight_1 = random, weight_2 = random, bias = random
             → prediction: 847°F (terrible!)

After training: weight_1 = 0.9, weight_2 = -0.05, bias = 3.2
             → prediction: 72°F (reasonable!)
```

Those three numbers (weight_1, weight_2, bias) are the **parameters** of this tiny model. Your GPT has **41.6 million** of them, arranged in a much more complex structure.

### Where do the 41.6M parameters live?

| Component                 | What It Does                                      | Parameters    | Analogy                      |
| ------------------------- | ------------------------------------------------- | ------------- | ---------------------------- |
| **Token Embedding**       | Converts each token to a 512-dimensional vector   | 16.4M (39%)   | A dictionary                 |
| **Attention Layers** (×8) | Figures out which words are related to each other | 8.4M (20%)    | Reading comprehension        |
| **FFN Layers** (×8)       | Processes and transforms the information          | 16.8M (40%)   | Thinking about what was read |
| **Layer Norms**           | Keeps numbers in a stable range                   | 0.02M (0.05%) | Calibration                  |

### More parameters = smarter?

Generally yes, but with diminishing returns:

```
41.6M params  → can write basic coherent English
120M params   → noticeably better, more nuanced
1B params     → can follow simple instructions
7B params     → quite capable, can reason somewhat
70B params    → very capable, strong reasoning
1.8T params   → GPT-4 level (near-human in many tasks)
```

But more parameters need **more data and more compute**:

```
41.6M  → 832M tokens optimal  → ~3 hours on your Mac
120M   → 2.4B tokens optimal  → ~12 hours on a GPU
7B     → 140B tokens optimal  → ~3 months on a GPU cluster
```

---

## 🎯 Quick Reference Card

```
YOUR MODEL AT A GLANCE
═════════════════════════════════════════
Parameters:     41.6 million
Context:        256 tokens (~200 words)
Vocabulary:     32,000 tokens
Training data:  WikiText-103 (113M tokens)
Training steps: 20,000 (current config)
Device:         Apple M4 (MPS)
Expected time:  ~2-3 hours
Expected PPL:   40-60 (good for this size)
═════════════════════════════════════════
```
