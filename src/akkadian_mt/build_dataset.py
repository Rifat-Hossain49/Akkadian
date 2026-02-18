from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Optional

import pandas as pd

from .text_cleaning import (
    clean_akkadian_transliteration,
    clean_english_translation,
    split_pseudo_sentences,
)

def _safe_read_csv(path: str, usecols: Optional[List[str]] = None) -> pd.DataFrame:
    # pandas is OK for typical competition train sizes.
    return pd.read_csv(path, usecols=usecols)

def iter_mtm24_rows(csv_path: str, max_rows: int, seed: int) -> Iterable[Tuple[str, str]]:
    """Stream-ish reading with chunking, then random sub-sample up to max_rows."""
    rnd = random.Random(seed)
    chunksize = 200_000
    kept = 0

    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        # Try to infer columns (case-insensitive)
        cols = {c.lower(): c for c in chunk.columns}
        tcol = cols.get("transliteration") or cols.get("source") or cols.get("src") or None
        ecol = cols.get("target") or cols.get("translation") or cols.get("tgt") or None
        if tcol is None or ecol is None:
            raise ValueError(
                f"MTM24 file columns not recognized. Found columns: {list(chunk.columns)[:20]}..."
            )

        for t, e in zip(chunk[tcol].astype(str), chunk[ecol].astype(str)):
            if kept >= max_rows:
                return
            t = clean_akkadian_transliteration(t)
            e = clean_english_translation(e)
            if not t or not e:
                continue
            # tiny filtering to avoid extremely long entries
            if len(t) > 2000 or len(e) > 2000:
                continue
            # light random downsampling inside chunks for speed
            if rnd.random() < 0.80:
                yield (t, e)
                kept += 1

def iter_competition_pairs(train_csv: str, max_rows: int, seed: int) -> Iterable[Tuple[str, str]]:
    df = pd.read_csv(train_csv)
    # heuristically guess columns
    cols = {c.lower(): c for c in df.columns}
    tcol = cols.get("transliteration") or cols.get("source") or cols.get("input") or None
    ecol = cols.get("translation") or cols.get("target") or cols.get("output") or None
    if tcol is None or ecol is None:
        # fallback: take first two text columns
        text_cols = [c for c in df.columns if df[c].dtype == object]
        if len(text_cols) >= 2:
            tcol, ecol = text_cols[0], text_cols[1]
        else:
            raise ValueError(f"Could not infer train columns from: {df.columns.tolist()}")
    df = df[[tcol, ecol]].dropna()
    df = df.sample(frac=1.0, random_state=seed).head(max_rows)

    for t, e in zip(df[tcol].astype(str), df[ecol].astype(str)):
        t = clean_akkadian_transliteration(t)
        e = clean_english_translation(e)
        if not t or not e:
            continue
        yield (t, e)

        # also produce pseudo-sentence pairs if alignable
        for st, se in split_pseudo_sentences(t, e):
            st = clean_akkadian_transliteration(st)
            se = clean_english_translation(se)
            if st and se:
                yield (st, se)

def iter_dictionary_word_pairs(dict_json_path: str, max_entries: int, seed: int) -> Iterable[Tuple[str, str]]:
    """Extract (form -> gloss) pairs from unified_akkadian_dict.json.

    This file structure can vary. We implement best-effort extraction:
    - root['form_to_entries'] mapping of form -> [entry, entry, ...]
    - each entry may include gloss/meaning/translation fields (string or list)
    """
    rnd = random.Random(seed)
    with open(dict_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    root = data.get("root", data)
    form_to_entries = root.get("form_to_entries") or root.get("form2entries") or {}
    if not isinstance(form_to_entries, dict):
        return

    def pick_gloss(entry: dict) -> Optional[str]:
        # Most common keys across glossaries
        for k in ("gloss", "meaning", "translation", "english", "en"):
            v = entry.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
            if isinstance(v, list) and v:
                # join short list
                vv = "; ".join([str(x).strip() for x in v if str(x).strip()])
                if vv.strip():
                    return vv.strip()
        # sometimes nested
        v = entry.get("senses") or entry.get("sense")
        if isinstance(v, list) and v:
            first = v[0]
            if isinstance(first, dict):
                return pick_gloss(first)
        return None

    count = 0
    items = list(form_to_entries.items())
    rnd.shuffle(items)
    for form, entries in items:
        if count >= max_entries:
            break
        if not isinstance(form, str):
            continue
        if not isinstance(entries, list):
            continue
        form_clean = clean_akkadian_transliteration(form)
        if not form_clean:
            continue

        # pick up to 2 glosses per form to limit noise
        glosses = []
        for ent in entries:
            if isinstance(ent, dict):
                g = pick_gloss(ent)
                if g:
                    g = clean_english_translation(g)
                    if g and g not in glosses:
                        glosses.append(g)
            if len(glosses) >= 2:
                break

        for g in glosses:
            # prefix makes multitask explicit
            yield (f"word: {form_clean}", g)
            count += 1
            if count >= max_entries:
                break

def build_jsonl(
    out_train_jsonl: str,
    out_valid_jsonl: str,
    competition_train_csv: str,
    mtm24_csv: Optional[str],
    dict_json: Optional[str],
    seed: int,
    max_comp_rows: int,
    max_mtm24_rows: int,
    max_dict_entries: int,
    w_sentence: float,
    w_word: float,
    valid_ratio: float = 0.02,
) -> Dict[str, int]:
    rnd = random.Random(seed)
    examples: List[Dict[str, str]] = []

    # sentence/doc pairs
    for t, e in iter_competition_pairs(competition_train_csv, max_rows=max_comp_rows, seed=seed):
        examples.append({"source": f"sent: {t}", "target": e})

    if mtm24_csv:
        for t, e in iter_mtm24_rows(mtm24_csv, max_rows=max_mtm24_rows, seed=seed):
            examples.append({"source": f"sent: {t}", "target": e})

    # word pairs
    if dict_json:
        for t, e in iter_dictionary_word_pairs(dict_json, max_entries=max_dict_entries, seed=seed):
            # already prefixed with word:
            examples.append({"source": t, "target": e})

    # reweight by simple upsampling (cheap and works well enough)
    sent = [ex for ex in examples if ex["source"].startswith("sent:")]
    word = [ex for ex in examples if ex["source"].startswith("word:")]
    examples = []
    if sent:
        examples += sent * max(1, int(round(w_sentence)))
    if word:
        examples += word * max(1, int(round(w_word)))

    # shuffle + split
    rnd.shuffle(examples)
    n_valid = max(1, int(len(examples) * valid_ratio))
    n_valid = min(n_valid, 2000)  # cap validation for fast eval
    valid = examples[:n_valid]
    train = examples[n_valid:]

    def dump(path: str, rows: List[Dict[str, str]]):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                # minimal sanity filters
                if not r["source"] or not r["target"]:
                    continue
                if len(r["source"]) > 3000 or len(r["target"]) > 3000:
                    continue
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dump(out_train_jsonl, train)
    dump(out_valid_jsonl, valid)

    return {"train": len(train), "valid": len(valid), "total": len(examples)}
