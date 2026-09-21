import json
import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, log_loss
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from .common import load_config, lora_config, quantization_config, seed_everything

LABEL_NAMES = ["A", "B", "TIE"]


def render(row):
    return (
        "Judge which assistant response is better for the user request. "
        "Return a latent representation for classification.\n\n"
        f"PROMPT\n{row['prompt']}\n\nRESPONSE A\n{row['response_a']}\n\nRESPONSE B\n{row['response_b']}"
    )


def main():
    cfg = load_config()
    seed_everything(cfg["seed"])
    ds = load_dataset("csv", data_files="data/processed/judge.csv.gz")["train"]
    train = ds.filter(lambda x: x["split"] == "train")
    val = ds.filter(lambda x: x["split"] == "validation")

    tok = AutoTokenizer.from_pretrained(cfg["base_model"], use_fast=True)
    def tokenize(batch):
        texts = [render({k: batch[k][i] for k in batch}) for i in range(len(batch["label"]))]
        return tok(texts, truncation=True, max_length=cfg["max_length"])
    train = train.map(tokenize, batched=True)
    val = val.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["base_model"],
        num_labels=3,
        quantization_config=quantization_config(cfg),
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config(cfg, task_type="SEQ_CLS"))
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir="artifacts/preference_judge_adapter",
        num_train_epochs=cfg["judge"]["epochs"],
        learning_rate=cfg["judge"]["learning_rate"],
        per_device_train_batch_size=cfg["judge"]["batch_size"],
        gradient_accumulation_steps=cfg["judge"]["gradient_accumulation_steps"],
        label_smoothing_factor=cfg["judge"]["label_smoothing"],
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
    )

    def metrics(pred):
        logits, labels = pred
        logits = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs = probs / probs.sum(axis=1, keepdims=True)
        yhat = probs.argmax(axis=1)
        return {
            "accuracy": accuracy_score(labels, yhat),
            "macro_f1": f1_score(labels, yhat, average="macro"),
            "log_loss": log_loss(labels, probs, labels=[0, 1, 2]),
        }

    trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=val, compute_metrics=metrics)
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
