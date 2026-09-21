from pathlib import Path

from datasets import concatenate_datasets, load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from .common import load_config, lora_config, quantization_config, seed_everything

COMMON_COLUMNS = ["prompt", "chosen", "rejected"]


def main():
    cfg = load_config()
    seed_everything(cfg["seed"])

    general = load_dataset("csv", data_files="data/processed/general_dpo.csv.gz")["train"]
    domain = load_dataset("csv", data_files="data/processed/credit_dpo.csv.gz")["train"]
    general = general.shuffle(seed=cfg["seed"]).select(range(min(len(general), 40000)))
    domain = domain.shuffle(seed=cfg["seed"])
    general = general.select_columns(COMMON_COLUMNS)
    domain = domain.select_columns(COMMON_COLUMNS)
    dataset = concatenate_datasets([general, domain]).shuffle(seed=cfg["seed"])
    split = dataset.train_test_split(test_size=0.05, seed=cfg["seed"])

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        quantization_config=quantization_config(cfg),
        device_map="auto",
    )

    sft_adapter = Path("artifacts/credit_sft_adapter")
    if sft_adapter.exists() and (sft_adapter / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(base, sft_adapter, is_trainable=True)
        peft_cfg = None
        init_stage = "sft_adapter"
    else:
        model = base
        peft_cfg = lora_config(cfg)
        init_stage = "base_model"

    print(f"DPO initialization: {init_stage}")
    args = DPOConfig(
        output_dir="artifacts/credit_dpo_adapter",
        num_train_epochs=cfg["dpo"]["epochs"],
        learning_rate=cfg["dpo"]["learning_rate"],
        beta=cfg["dpo"]["beta"],
        per_device_train_batch_size=cfg["dpo"]["batch_size"],
        gradient_accumulation_steps=cfg["dpo"]["gradient_accumulation_steps"],
        warmup_ratio=cfg["dpo"]["warmup_ratio"],
        bf16=True,
        gradient_checkpointing=True,
        max_length=cfg["max_length"],
        report_to="none",
    )
    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=tokenizer,
        peft_config=peft_cfg,
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
