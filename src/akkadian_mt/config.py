from dataclasses import dataclass

@dataclass
class TrainConfig:
    model_name: str = "google/byt5-small"
    seed: int = 42

    # sequence lengths
    max_source_len: int = 256
    max_target_len: int = 128

    # training
    lr: float = 3e-4
    weight_decay: float = 0.01
    num_train_epochs: float = 1.0
    warmup_ratio: float = 0.03

    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    gradient_accumulation_steps: int = 1

    fp16: bool = True
    eval_steps: int = 5000
    save_steps: int = 5000
    logging_steps: int = 200

    # decoding (for validation + inference)
    num_beams: int = 2
    max_new_tokens: int = 128

    # mixture weights
    # sentence-level: MTM24 + competition sentence/doc
    # word-level: dictionary (form->gloss)
    w_sentence: float = 1.0
    w_word: float = 0.25

    # limits (to keep runtime stable)
    max_mtm24_rows: int = 400_000
    max_dict_entries: int = 250_000
    max_comp_rows: int = 200_000
