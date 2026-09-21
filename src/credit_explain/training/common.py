from pathlib import Path
import random
import yaml
import numpy as np
import torch
from peft import LoraConfig
from transformers import BitsAndBytesConfig


def load_config(path: str = "configs/train.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def quantization_config(cfg: dict) -> BitsAndBytesConfig:
    q = cfg["qlora"]
    dtype = torch.bfloat16 if q["compute_dtype"] == "bfloat16" else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"],
        bnb_4bit_quant_type=q["quant_type"],
        bnb_4bit_use_double_quant=q["double_quant"],
        bnb_4bit_compute_dtype=dtype,
    )


def lora_config(cfg: dict, task_type: str = "CAUSAL_LM") -> LoraConfig:
    q = cfg["qlora"]
    return LoraConfig(
        r=q["lora_r"],
        lora_alpha=q["lora_alpha"],
        lora_dropout=q["lora_dropout"],
        target_modules=q["target_modules"],
        bias="none",
        task_type=task_type,
    )
