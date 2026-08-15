#!/usr/bin/env python3
"""Clean SFT entry point for a verified manifest-based fundus dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mg_rl.prompt import SYSTEM_PROMPT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if cfg.get("seed") is None:
        raise ValueError(
            "The submitted Methods do not state the split seed. "
            "Set a verified seed before training."
        )
    manifest = Path(cfg["dataset_manifest"])
    if not manifest.exists():
        raise FileNotFoundError(
            f"Missing {manifest}. Populate it from the verified historical split; "
            "do not create a new random split."
        )

    import torch
    from datasets import DatasetDict, Image, load_dataset
    from peft import LoraConfig
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        BitsAndBytesConfig,
    )
    from trl import SFTConfig, SFTTrainer

    dataset = load_dataset("csv", data_files=str(manifest), split="train")
    dataset = dataset.rename_column("image_path", "image").cast_column(
        "image", Image()
    )
    data = DatasetDict(
        train=dataset.filter(lambda row: row["split"] == "train"),
        validation=dataset.filter(lambda row: row["split"] == "validation"),
    )
    processor = AutoProcessor.from_pretrained(
        cfg["base_model"], trust_remote_code=True
    )
    model = AutoModelForImageTextToText.from_pretrained(
        cfg["base_model"],
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        trust_remote_code=True,
    )
    peft_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_targets"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    def user_messages() -> list[dict]:
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {"role": "user", "content": [{"type": "image"}]},
        ]

    def collate(rows: list[dict]) -> dict:
        prompts = []
        full_texts = []
        images = []
        for row in rows:
            user = user_messages()
            full = user + [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": row["solution"]}],
                }
            ]
            prompts.append(
                processor.apply_chat_template(
                    user, tokenize=False, add_generation_prompt=True
                )
            )
            full_texts.append(
                processor.apply_chat_template(
                    full, tokenize=False, add_generation_prompt=False
                )
            )
            images.append([row["image"].convert("RGB")])

        prompt_batch = processor(
            text=prompts,
            images=images,
            padding=True,
            truncation=True,
            max_length=cfg["max_length"],
            return_tensors="pt",
        )
        batch = processor(
            text=full_texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=cfg["max_length"],
            return_tensors="pt",
        )
        pad_id = processor.tokenizer.pad_token_id
        labels = batch["input_ids"].clone()
        labels[labels == pad_id] = -100
        for index in range(len(rows)):
            prompt_length = (
                prompt_batch["input_ids"][index] != pad_id
            ).sum().item()
            labels[index, :prompt_length] = -100
        batch["labels"] = labels
        return batch

    training_args = SFTConfig(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["epochs"],
        learning_rate=cfg["learning_rate"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        bf16=cfg["bf16"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=cfg["seed"],
        report_to="none",
        remove_unused_columns=False,
        max_length=cfg["max_length"],
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        processing_class=processor,
        peft_config=peft_config,
        data_collator=collate,
    )
    trainer.train()
    trainer.save_model(cfg["output_dir"])
    processor.save_pretrained(cfg["output_dir"])


if __name__ == "__main__":
    main()

