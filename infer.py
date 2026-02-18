import argparse
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.akkadian_mt.text_cleaning import clean_akkadian_transliteration


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--test_csv", type=str, default=None,
                   help="Direct path to test.csv. If not given, uses auto_discover.")
    p.add_argument(
        "--input_root",
        type=str,
        default=os.environ.get("KAGGLE_INPUT_ROOT", "./kaggle_input"),
        help="Root folder containing Kaggle inputs (competition data + optional extra datasets).",
    )
    p.add_argument("--model_dir", required=True, help="Path to trained model directory (e.g., .../best).")
    p.add_argument("--out_csv", default="submission.csv", help="Where to write Kaggle submission.")
    p.add_argument("--num_beams", type=int, default=2)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=32)
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve test CSV path
    if args.test_csv and os.path.exists(args.test_csv):
        test_csv_path = args.test_csv
    else:
        from src.akkadian_mt.data_sources import auto_discover, assert_required
        paths = auto_discover(args.input_root)
        assert_required(paths, require_mtm24=False, require_dict=False)
        test_csv_path = paths.competition_test_csv

    test_df = pd.read_csv(test_csv_path)

    # infer source column
    cols = {c.lower(): c for c in test_df.columns}
    tcol = cols.get("transliteration") or cols.get("source") or cols.get("input") or None
    if tcol is None:
        # take first object column
        text_cols = [c for c in test_df.columns if test_df[c].dtype == object]
        if not text_cols:
            raise ValueError("Could not infer source column in test.csv")
        tcol = text_cols[0]

    # infer ID column
    id_col = cols.get("id") or cols.get("text_id") or cols.get("index") or None
    if id_col is None:
        id_col = test_df.columns[0]

    sources = [clean_akkadian_transliteration(x) for x in test_df[tcol].astype(str).tolist()]
    # prefix consistent with training
    sources = [f"sent: {s}" for s in sources]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir).to(device).eval()

    # m2m100-specific: get forced_bos_token_id for English generation
    gen_kwargs = {}
    if hasattr(model.config, "forced_bos_token_id") and model.config.forced_bos_token_id is not None:
        gen_kwargs["forced_bos_token_id"] = model.config.forced_bos_token_id

    preds = []
    bs = args.batch_size

    for i in range(0, len(sources), bs):
        batch = sources[i:i+bs]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                out = model.generate(
                    **enc,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    **gen_kwargs,
                )
        dec = tokenizer.batch_decode(out, skip_special_tokens=True)
        preds.extend(dec)

    # Build submission DataFrame directly (no sample_submission.csv needed)
    out_df = pd.DataFrame({
        id_col: test_df[id_col].values[:len(preds)],
        "translation": preds[:len(test_df)],
    })
    out_df.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(out_df)} predictions to: {args.out_csv}")


if __name__ == "__main__":
    main()

