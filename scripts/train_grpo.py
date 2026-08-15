#!/usr/bin/env python3
"""Clean GRPO entry point based on the recovered source archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mg_rl.prompt import SYSTEM_PROMPT
from mg_rl.rewards_recovered import (
    criteria_reward,
    diagnosis_reward,
    format_reward,
)


def load_config(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if cfg.get("reward_profile") != "recovered_source_unweighted":
        raise ValueError(
            "The exact manuscript-weighted reward implementation was not recovered. "
            "Use configs/recovered_grpo_launch.json or supply a verified reward module."
        )

    import torch
    from datasets import Image, load_dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from trl import GRPOConfig, GRPOTrainer

    processor = AutoProcessor.from_pretrained(
        cfg["base_model"], trust_remote_code=True
    )
    model = AutoModelForImageTextToText.from_pretrained(
        cfg["base_model"],
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    adapter_path = cfg.get("sft_adapter_path")
    if adapter_path:
        model = PeftModel.from_pretrained(
            model, adapter_path, is_trainable=True
        )
        peft_config = None
    else:
        peft_config = LoraConfig(
            r=cfg["lora_r"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],
            target_modules=cfg["lora_targets"],
            bias="none",
            task_type="CAUSAL_LM",
        )

    dataset = load_dataset(
        cfg["dataset_name"], split=cfg.get("train_split", "train")
    )
    if not isinstance(dataset.features.get("image"), Image):
        dataset = dataset.cast_column("image", Image())

    def prepare(example: dict) -> dict:
        prompt = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [{"type": "image", "image": example["image"]}],
            },
        ]
        return {
            "prompt": prompt,
            "image_list": [example["image"]],
            "_solution": example["solution"],
        }

    dataset = dataset.map(prepare, remove_columns=dataset.column_names)
    training_args = GRPOConfig(
        output_dir=cfg["output_dir"],
        max_steps=cfg["max_steps"],
        learning_rate=cfg["learning_rate"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_steps=cfg.get("warmup_steps", 0),
        weight_decay=cfg["weight_decay"],
        bf16=True,
        num_generations=cfg["num_generations"],
        num_iterations=cfg["num_iterations"],
        temperature=cfg["temperature"],
        max_prompt_length=cfg["max_prompt_length"],
        max_completion_length=cfg["max_completion_length"],
        beta=cfg["beta"],
        epsilon=cfg["epsilon"],
        epsilon_high=cfg["epsilon_high"],
        importance_sampling_level=cfg["importance_sampling_level"],
        report_to="none",
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=[diagnosis_reward, format_reward, criteria_reward],
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(cfg["output_dir"])
    processor.save_pretrained(cfg["output_dir"])


if __name__ == "__main__":
    main()

