# Akkadian Transliteration → English (ByT5) — Kaggle-ready codebase

This is a **fully offline** training + inference codebase intended for Kaggle notebooks/competitions.
It is designed for the "Deep Past Challenge – Translate Akkadian to English" setup (transliteration → English),
and it also supports **word-level** training using a dictionary dataset.

## Why ByT5?
Akkadian transliteration contains lots of diacritics and sign-like tokens (e.g., š, ḫ, ṭ, {KUR}, etc.).
**ByT5** operates on bytes, so it does not need a custom tokenizer and is robust to unusual characters.

## Datasets (expected as Kaggle Inputs)
Add these as notebook inputs:

1) **Competition dataset** (contains train.csv, test.csv)
   - Provides the *main* transliteration→translation pairs.

2) **MTM24 transliteration dataset** (contains `mtm24_transliterated.csv`)
   - A large parallel corpus (transliteration + English target) for sentence-level training.

3) **Akkadian Dictionary** (contains `unified_akkadian_dict.json`)
   - Used to create *word-level* pairs: (form → gloss).

The code auto-discovers files by scanning the input root for:
- `train.csv`, `test.csv` (competition)
- `mtm24_transliterated.csv`
- `unified_akkadian_dict.json`

## Setup
```bash
pip install -r requirements.txt
```

## Download Pre-trained Model (Required)
The training script uses a pre-trained ByT5 model optimized for Akkadian. Download it from Kaggle before training:

```bash
# Install Kaggle CLI
pip install kaggle

# Download and unzip the model (~2GB)
kaggle datasets download assiaben/final-byt5 -p ./final-byt5 --unzip
```

> **Note:** You need a Kaggle API key (`kaggle.json`) in `~/.kaggle/`. Get it from [Kaggle Settings → API → Create New Token](https://www.kaggle.com/settings).

## Running Locally

### Step 1: Prepare Training Data (JSONL)

Place `train.csv`, `test.csv`, `mtm24_transliterated.csv`, and `unified_akkadian_dict.json` in the project root (or inside a folder and use `--input_root`).

```bash
# Using project root as input source
python train.py --prepare_data --input_root . --out_dir ./data_out
```

This creates:
- `data_out/train.jsonl` — combined sentence-level (MTM24 + competition) and word-level (dictionary) training data
- `data_out/valid.jsonl` — validation split

### Step 2: Train the Model

```bash
# GPU (recommended)
python train.py --data_dir ./data_out --output_dir ./ckpt_byt5

# CPU (add --no_fp16, significantly slower)
python train.py --data_dir ./data_out --output_dir ./ckpt_byt5 --no_fp16
```

#### Optional Training Arguments

| Argument              | Default       | Description                            |
|-----------------------|---------------|----------------------------------------|
| `--model_name`        | `google/byt5-small` | HuggingFace model name            |
| `--num_train_epochs`  | `1.0`         | Number of training epochs              |
| `--lr`                | `3e-4`        | Learning rate                          |
| `--train_bs`          | `16`          | Per-device train batch size            |
| `--eval_bs`           | `32`          | Per-device eval batch size             |
| `--grad_accum`        | `1`           | Gradient accumulation steps            |
| `--max_mtm24_rows`    | `400000`      | Max MTM24 rows to use                  |
| `--max_dict_entries`  | `250000`      | Max dictionary entries for word-level  |
| `--max_comp_rows`     | `200000`      | Max competition train rows             |
| `--w_sentence`        | `1.0`         | Sentence-level upsampling weight       |
| `--w_word`            | `0.25`        | Word-level upsampling weight           |
| `--fp16` / `--no_fp16`| `fp16=True`   | Mixed precision training               |

### Step 3: Inference (Generate submission.csv)

```bash
# Using --test_csv for direct file path
python infer.py --model_dir ./ckpt_byt5/best --test_csv ./test.csv --out_csv submission.csv

# Or using auto-discover with --input_root
python infer.py --model_dir ./ckpt_byt5/best --input_root . --out_csv submission.csv
```

## Quickstart (Kaggle Notebook)

```bash
# Step 1: Build training data
python train.py --prepare_data --out_dir /kaggle/working/data_out

# Step 2: Train
python train.py --data_dir /kaggle/working/data_out --output_dir /kaggle/working/ckpt_byt5

# Step 3: Inference
python infer.py --model_dir /kaggle/working/ckpt_byt5/best --out_csv /kaggle/working/submission.csv
```

## Notes on speed / avoiding timeouts
- Inference is batched and uses fp16 autocast when CUDA is available.
- Default decoding is **beam=2** (good quality/latency tradeoff). Use `--num_beams 1` for fastest.
- **GPU is strongly recommended** for training. CPU training can take days.

## File structure
- `train.py`: data prep + training entrypoint
- `infer.py`: competition inference entrypoint
- `src/akkadian_mt/`: library code (cleaning, dataset building, metrics)

---
If you want to add your own cleaning rules, edit:
`src/akkadian_mt/text_cleaning.py`
