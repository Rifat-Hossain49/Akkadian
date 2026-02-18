from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional, Dict, List

@dataclass
class DataPaths:
    competition_train_csv: Optional[str] = None
    competition_test_csv: Optional[str] = None
    mtm24_csv: Optional[str] = None
    akkadian_dict_json: Optional[str] = None

def _find_first(root: str, target_filename: str) -> Optional[str]:
    for dirpath, _, filenames in os.walk(root):
        if target_filename in filenames:
            return os.path.join(dirpath, target_filename)
    return None

def auto_discover(kaggle_input_root: str = "/kaggle/input") -> DataPaths:
    if not os.path.exists(kaggle_input_root):
        return DataPaths()

    return DataPaths(
        competition_train_csv=_find_first(kaggle_input_root, "train.csv"),
        competition_test_csv=_find_first(kaggle_input_root, "test.csv"),
        mtm24_csv=_find_first(kaggle_input_root, "mtm24_transliterated.csv"),
        akkadian_dict_json=_find_first(kaggle_input_root, "unified_akkadian_dict.json"),
    )

def assert_required(paths: DataPaths, require_mtm24: bool = True, require_dict: bool = True) -> None:
    missing = []
    if paths.competition_train_csv is None:
        missing.append("train.csv (competition)")
    if paths.competition_test_csv is None:
        missing.append("test.csv (competition)")
    if require_mtm24 and paths.mtm24_csv is None:
        missing.append("mtm24_transliterated.csv (MTM24 dataset)")
    if require_dict and paths.akkadian_dict_json is None:
        missing.append("unified_akkadian_dict.json (Akkadian Dictionary)")
    if missing:
        raise FileNotFoundError(
            "Missing required inputs:\n- " + "\n- ".join(missing) +
            "\n\nAdd the corresponding Kaggle datasets as Notebook Inputs, "
            "then re-run."
        )
