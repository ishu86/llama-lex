# llama-lex

Fine-tunes **Llama-3.2-3B** to write social media posts in **Lex Fridman's** distinctive voice — philosophical, reflective, optimistic about technology and humanity.

## Pipeline

1. **Raw posts** — `data/lex_posts.json` (~7,625 entries)
2. **Curation** — `data/curator.py` uses GPT-4o-mini to score each post 1–10 on authenticity
3. **Filtering** — `train/filter_dataset.py` keeps 5+ score (~340 entries)
4. **Fine-tuning** — Unsloth + LoRA on Llama-3.2-3B (4-bit, r=64, α=16)
5. **Inference** — `train/inference.py` generates posts from a topic

## Data

The scraped posts and the curated/scored derivatives (`data/*.json`) are **not included** in this repo. Bring your own `data/lex_posts.json` (a list of `{"date_posted", "content"}` objects), then run `data/curator.py` and `train/filter_dataset.py` to build the training set.

## Layout

```
data/                      # scoring utility (datasets gitignored)
train/
  training_5plus_curated.py  # SFT on curated 5+ subset
  train_lex_style.py         # SFT on full dataset
  train_wieighted_sft.py     # quality-weighted SFT
  filter_dataset.py          # extract high-score subset
  inference.py               # LexStyleGenerator
run_training.py            # interactive launcher
models/                    # trained artifacts (gitignored — see HF Hub)
```

## Models

Trained artifacts are hosted on Hugging Face: **`ishu86/llama-lex`**

| Subfolder | Format | Use |
|---|---|---|
| `lex-5plus-model/` | LoRA adapter | layer onto base Llama-3.2-3B |
| `lex-5plus-model_merged/` | Merged HF safetensors | `transformers.from_pretrained` / vLLM |
| `lexF-llama.gguf` + `Modelfile` | GGUF | llama.cpp / Ollama |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install unsloth transformers trl datasets openai python-dotenv
echo "OPENAI_API_KEY=sk-..." > .env   # only needed for curation
```

## Usage

```bash
python run_training.py          # interactive training menu
python train/inference.py       # generate Lex-style posts
```

## Stack

Unsloth · Hugging Face Transformers · TRL · Llama-3.2-3B · OpenAI API (curation only)
