# MG\_RL reproducibility package

This repository contains cleaned code and documentation for the MG\_RL.

## Scope

The package contains:

- cleaned SFT, GRPO, and inference entry points;
- the structured ICDR grading prompt;
- reward functions and output parsers;
- configuration records;
- split-manifest and evaluation-input schemas;
- code and tabular inputs for the decision-curve figure
- pure-function tests that run without the MedGemma weights or clinical images.

## Repository layout

```
configs/              recovered configuration records
data/                 public schemas; no image data
figures/              journal-ready decision-curve outputs
scripts/              SFT, GRPO, inference, and figure entry points
src/mg_rl/            prompt, parsing, reward, and DCA functions
tests/                pure-function tests
weights/              adapter metadata and release status
```

## Installation

Python 3.10 and CUDA-capable hardware are expected for training.

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

MedGemma access requires acceptance of the Health AI Developer Foundations terms. Set credentials only in the shell or a secret manager; do not write them into scripts.

## Examples

```
PYTHONPATH=src python scripts/train_sft.py --config configs/manuscript_sft.json
PYTHONPATH=src python scripts/train_grpo.py --config configs/recovered_grpo_launch.json
PYTHONPATH=src python scripts/infer.py --input-jsonl data/evaluation_manifest.example.jsonl --adapter-path /path/to/adapter --output-json results.json
PYTHONPATH=src python scripts/redraw_decision_curve.py
python -m unittest discover -s tests -v
```

## Intended use

This is research software for retrospective model evaluation. It is not a medical device and must not be used for autonomous clinical diagnosis or treatment decisions.

## Model terms

MedGemma and any model derivatives are subject to the [Health AI Developer Foundations Terms of Use](https://developers.google.com/health-ai-developer-foundations/terms) and [Prohibited Use Policy](https://developers.google.com/health-ai-developer-foundations/prohibited-use-policy). Source code in this repository is provided for research review; no license is granted for third-party material or model weights.
