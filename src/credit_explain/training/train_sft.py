from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from .common import load_config, lora_config, quantization_config, seed_everything


def main():
    cfg = load_config()
    seed_everything(cfg["seed"])
    dataset = load_dataset("csv", data_files="data/processed/credit_sft.csv.gz")["train"]
    dataset = dataset.train_test_split(test_size=0.1, seed=cfg["seed"])

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        quantization_config=quantization_config(cfg),
        device_map="auto",
    )

    def formatting(example):
        messages = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["response"]},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False)

    args = SFTConfig(
        output_dir="artifacts/credit_sft_adapter",
        num_train_epochs=cfg["sft"]["epochs"],
        learning_rate=cfg["sft"]["learning_rate"],
        per_device_train_batch_size=cfg["sft"]["batch_size"],
        gradient_accumulation_steps=cfg["sft"]["gradient_accumulation_steps"],
        warmup_ratio=cfg["sft"]["warmup_ratio"],
        logging_steps=cfg["sft"]["logging_steps"],
        eval_strategy="steps",
        eval_steps=cfg["sft"]["eval_steps"],
        save_steps=cfg["sft"]["save_steps"],
        bf16=True,
        gradient_checkpointing=True,
        max_length=cfg["max_length"],
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=lora_config(cfg),
        formatting_func=formatting,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
