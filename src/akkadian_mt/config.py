from dataclasses import dataclass

@dataclass
class TrainConfig:
    model_name: str = "facebook/m2m100_418M"
    seed: int = 42

    # m2m100 target language (for forced_bos_token_id during generation)
    tgt_lang: str = "en"
    src_lang: str = "akk"  # closest proxy; m2m100 ignores unknown codes gracefully

    # sequence lengths (subword tokenizer = shorter than ByT5 byte-level)
    max_source_len: int = 256
    max_target_len: int = 128

    # training
    lr: float = 3e-4
    weight_decay: float = 0.01
    num_train_epochs: float = 3.0
    warmup_ratio: float = 0.06

    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 64
    gradient_accumulation_steps: int = 2   # effective batch = 64

    fp16: bool = True
    gradient_checkpointing: bool = True    # saves VRAM, allows bigger batch
    dataloader_num_workers: int = 4        # parallel data loading

    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 50

    # decoding (for validation + inference)
    num_beams: int = 2
    max_new_tokens: int = 128

    # mixture weights
    # sentence-level: MTM24 + competition sentence/doc
    # word-level: dictionary (form->gloss)
    w_sentence: float = 1.0
    w_word: float = 0.5

    # limits — balanced for practical training time
    max_mtm24_rows: int = 80_000
    max_dict_entries: int = 50_000
    max_comp_rows: int = 200_000
