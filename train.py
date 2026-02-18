import argparse
import os
import random

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

from src.akkadian_mt.config import TrainConfig
from src.akkadian_mt.data_sources import auto_discover, assert_required
from src.akkadian_mt.build_dataset import build_jsonl
from src.akkadian_mt.metrics import compute_bleu_chrf_geomean


def parse_args():
    p = argparse.ArgumentParser()

    # Local runs: point this at the folder that contains the downloaded Kaggle inputs.
    # On Kaggle notebooks, this is typically /kaggle/input.
    p.add_argument(
        "--input_root",
        type=str,
        default=os.environ.get("KAGGLE_INPUT_ROOT", "./kaggle_input"),
        help="Root folder containing Kaggle inputs (competition data + optional extra datasets).",
    )
    p.add_argument("--prepare_data", action="store_true", help="Build train/valid JSONL from Kaggle inputs.")
    p.add_argument("--out_dir", default="./data_out", help="Where to write JSONL if --prepare_data.")
    p.add_argument("--data_dir", default="./data_out", help="Directory containing train.jsonl and valid.jsonl.")
    p.add_argument("--model_name", default=None)
    p.add_argument("--output_dir", default="./ckpt_byt5")
    p.add_argument("--max_source_len", type=int, default=None)
    p.add_argument("--max_target_len", type=int, default=None)
    p.add_argument("--num_train_epochs", type=float, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--train_bs", type=int, default=None)
    p.add_argument("--eval_bs", type=int, default=None)
    p.add_argument("--grad_accum", type=int, default=None)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--no_fp16", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    # limits / mixture
    p.add_argument("--max_mtm24_rows", type=int, default=None)
    p.add_argument("--max_dict_entries", type=int, default=None)
    p.add_argument("--max_comp_rows", type=int, default=None)
    p.add_argument("--w_sentence", type=float, default=None)
    p.add_argument("--w_word", type=float, default=None)

    return p.parse_args()


def main():
    args = parse_args()
    cfg = TrainConfig(seed=args.seed)
    if args.model_name:
        cfg.model_name = args.model_name
    if args.max_source_len is not None:
        cfg.max_source_len = args.max_source_len
    if args.max_target_len is not None:
        cfg.max_target_len = args.max_target_len
    if args.num_train_epochs is not None:
        cfg.num_train_epochs = args.num_train_epochs
    if args.lr is not None:
        cfg.lr = args.lr
    if args.train_bs is not None:
        cfg.per_device_train_batch_size = args.train_bs
    if args.eval_bs is not None:
        cfg.per_device_eval_batch_size = args.eval_bs
    if args.grad_accum is not None:
        cfg.gradient_accumulation_steps = args.grad_accum
    if args.max_mtm24_rows is not None:
        cfg.max_mtm24_rows = args.max_mtm24_rows
    if args.max_dict_entries is not None:
        cfg.max_dict_entries = args.max_dict_entries
    if args.max_comp_rows is not None:
        cfg.max_comp_rows = args.max_comp_rows
    if args.w_sentence is not None:
        cfg.w_sentence = args.w_sentence
    if args.w_word is not None:
        cfg.w_word = args.w_word
    if args.fp16:
        cfg.fp16 = True
    if args.no_fp16:
        cfg.fp16 = False

    set_seed(cfg.seed)

    if args.prepare_data:
        os.makedirs(args.out_dir, exist_ok=True)
        paths = auto_discover(args.input_root)
        # require MTM24 + dict by default for best performance; turn off if you want
        assert_required(paths, require_mtm24=False, require_dict=False)

        stats = build_jsonl(
            out_train_jsonl=os.path.join(args.out_dir, "train.jsonl"),
            out_valid_jsonl=os.path.join(args.out_dir, "valid.jsonl"),
            competition_train_csv=paths.competition_train_csv,
            mtm24_csv=paths.mtm24_csv,
            dict_json=paths.akkadian_dict_json,
            seed=cfg.seed,
            max_comp_rows=cfg.max_comp_rows,
            max_mtm24_rows=cfg.max_mtm24_rows,
            max_dict_entries=cfg.max_dict_entries,
            w_sentence=cfg.w_sentence,
            w_word=cfg.w_word,
        )
        print("Data prep done:", stats)
        return

    # load jsonl datasets
    train_path = os.path.join(args.data_dir, "train.jsonl")
    valid_path = os.path.join(args.data_dir, "valid.jsonl")
    if not os.path.exists(train_path) or not os.path.exists(valid_path):
        raise FileNotFoundError(
            f"Missing {train_path} or {valid_path}. Run with --prepare_data first."
        )

    raw = load_dataset("json", data_files={"train": train_path, "validation": valid_path})

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name)

    # m2m100-specific: set target language for tokenizer
    forced_bos_token_id = None
    if hasattr(tokenizer, "lang_code_to_id"):
        # M2M100 / NLLB style tokenizer
        if cfg.tgt_lang in tokenizer.lang_code_to_id:
            tokenizer.tgt_lang = cfg.tgt_lang
            forced_bos_token_id = tokenizer.lang_code_to_id[cfg.tgt_lang]
        elif "en" in tokenizer.lang_code_to_id:
            tokenizer.tgt_lang = "en"
            forced_bos_token_id = tokenizer.lang_code_to_id["en"]

    def preprocess(batch):
        model_inputs = tokenizer(
            batch["source"],
            max_length=cfg.max_source_len,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["target"],
            max_length=cfg.max_target_len,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = raw.map(preprocess, batched=True, remove_columns=raw["train"].column_names)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Metrics: decode generated tokens at eval time
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        m = compute_bleu_chrf_geomean(decoded_preds, decoded_labels)
        return {
            "bleu": m["bleu"],
            "chrfpp": m["chrfpp"],
            "geomean": m["geomean"],
        }

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        logging_steps=cfg.logging_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        predict_with_generate=True,
        generation_max_length=cfg.max_new_tokens,
        generation_num_beams=cfg.num_beams,
        fp16=cfg.fp16 and torch.cuda.is_available(),
        gradient_checkpointing=cfg.gradient_checkpointing,
        dataloader_num_workers=cfg.dataloader_num_workers,
        report_to="none",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="geomean",
        greater_is_better=True,
    )

    # Set forced_bos_token_id for m2m100 generation (tells model to output English)
    if forced_bos_token_id is not None:
        model.config.forced_bos_token_id = forced_bos_token_id
        training_args.generation_config = model.generation_config
        training_args.generation_config.forced_bos_token_id = forced_bos_token_id

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Save best model in a stable location
    best_dir = os.path.join(args.output_dir, "best")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    print("Saved best model to:", best_dir)


if __name__ == "__main__":
    main()
