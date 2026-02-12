# Instruction Fine-Tuning Sarvam AI 2B using LoRA

This project explores instruction fine-tuning of the Sarvam AI 2B
language model using LoRA (Low-Rank Adaptation). The goal is to improve
response efficiency and cleanliness without retraining the full model.

Only \~0.13% of model parameters are fine-tuned using a small synthetic
instruction dataset. The fine-tuned model is evaluated against the base
model using a custom, reproducible evaluation pipeline.

## Key Result

-   \~21% reduction in average response length
-   Maintained high instruction adherence across mixed-domain and
    constraint-heavy prompts

## Tech Stack

-   PyTorch
-   HuggingFace Transformers
-   PEFT (LoRA)
-   Custom evaluation scripts

## Highlights

-   End-to-end pipeline: data → fine-tuning → evaluation
-   Lightweight LoRA adaptation (no full model retraining)
-   Honest, interpretable evaluation metrics
-   Reproducible experimental setup

This project demonstrates practical instruction tuning, behavior-level
optimization, and clean ML engineering practices.
