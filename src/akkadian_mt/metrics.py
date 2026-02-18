from __future__ import annotations
import math
from typing import List, Dict

import sacrebleu

def compute_bleu_chrf_geomean(preds: List[str], refs: List[str]) -> Dict[str, float]:
    # sacrebleu expects list of hypotheses and list-of-list references
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    # chrF++: word_order=2
    chrf = sacrebleu.corpus_chrf(preds, [refs], word_order=2).score
    geo = math.sqrt(max(0.0, bleu) * max(0.0, chrf))
    return {"bleu": bleu, "chrfpp": chrf, "geomean": geo}
