#!/usr/bin/env python3
"""Deterministic MG_RL inference from a JSONL manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mg_rl.parsing import extract_grade, extract_reasoning
from mg_rl.prompt import SYSTEM_PROMPT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--base-model", default="google/medgemma-4b-it")
    parser.add_argument("--max-new-tokens", type=int, default=320)
    args = parser.parse_args()

    import torch
    from PIL import Image
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        args.base_model, trust_remote_code=True
    )
    base = AutoModelForImageTextToText.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter_path).eval()
    rows = [
        json.loads(line)
        for line in Path(args.input_jsonl).read_text().splitlines()
        if line.strip()
    ]
    results = []
    for row in rows:
        image = Image.open(row["image_path"]).convert("RGB")
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [{"type": "image", "image": image}],
            },
        ]
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[prompt], images=[[image]], return_tensors="pt"
        ).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
            )
        continuation = generated[:, inputs["input_ids"].shape[1] :]
        text = processor.batch_decode(
            continuation, skip_special_tokens=True
        )[0]
        results.append(
            {
                "sample_id": row["sample_id"],
                "model_output": text,
                "predicted_grade": extract_grade(text),
                "reasoning": extract_reasoning(text),
            }
        )
    Path(args.output_json).write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
