# Vedaz Qwen2.5 LoRA Fine-Tuning

Fine-tunes `Qwen2.5-0.5B-Instruct` with LoRA on a bilingual (English/Hindi)
Vedic-astrology assistant chat dataset, on CPU only. Includes a full
deployment write-up for serving the result via vLLM on a VPS.

See [`docs/approach.md`](docs/approach.md) for the full engineering
write-up (dataset analysis, model/method justification, and results —
including an honest discussion of where the fine-tune fell short).

See [`docs/vps_vllm_deployment.md`](docs/vps_vllm_deployment.md) for the
deployment write-up.

## Project Structure

```
qwen-astro-finetune/
├── data/
│   ├── raw/chat_data.json        # original dataset, untouched
│   └── processed/                # train.jsonl, eval.jsonl (generated)
├── src/
│   ├── preprocess.py             # fixes malformed JSON, tokenizes, masks, splits
│   ├── train.py                  # LoRA SFT training
│   ├── inference.py              # base vs fine-tuned comparison + safety guardrail
│   ├── safety_filter.py          # rule-based crisis-language guardrail
│   └── utils.py                  # config loading, logging
├── config/train_config.yaml      # all hyperparameters — no hardcoding in scripts
├── outputs/adapter/final/        # trained LoRA adapter (generated)
├── docs/
│   ├── approach.md               # engineering write-up + results
│   └── vps_vllm_deployment.md    # deployment write-up
├── examples/sample_outputs.md    # before/after generation comparison (generated)
├── models/                       # HF model cache (generated, gitignored)
└── requirements.txt
```

## Setup

```bash
python3.11 -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
```

## Running the Pipeline

Run from the project root, in order:

```bash
python src/preprocess.py   # ~2 min — downloads tokenizer, fixes raw JSON, tokenizes + masks
python src/train.py        # ~35 min on CPU — downloads base model, runs LoRA SFT
python src/inference.py    # ~8 min — generates before/after comparison
```

Each step logs progress to stdout. `train.py` prints a
`trainable params: X || all params: Y || trainable%: Z` line early on —
confirm this is under ~1% to verify LoRA (not full fine-tuning) is active.

## Model & Method

- **Base model**: `Qwen/Qwen2.5-0.5B-Instruct` — smallest Qwen2.5-Instruct
  checkpoint, chosen for CPU/storage constraints
- **Method**: LoRA (rank 16, alpha 32, targeting attention projections) —
  full fine-tuning is inappropriate for a 55-example dataset
- **Trainable parameters**: 0.44% (2.16M / 496M)

## Results Summary

| Metric | Value |
|---|---|
| Training time (CPU) | ~36 min |
| Epochs | 4 |
| Eval loss | 1.947 → 1.918 (steady, no overfitting collapse) |

Qualitative results (`examples/sample_outputs.md`) show clear stylistic/
persona adaptation on English-language cases. Testing surfaced a genuine
safety gap: crisis-language Hindi input did not reliably trigger the
required safety-redirect behavior from the fine-tuned model alone. This
was fixed with a **rule-based safety guardrail** (`src/safety_filter.py`)
that intercepts crisis/self-harm language (English, romanized Hindi, and
Devanagari) *before* it reaches the model, returning a fixed response with
two verified Indian national helplines — Tele MANAS (14416) and KIRAN
(1800-599-0019) — regardless of model output quality. A residual,
lower-stakes limitation remains: non-crisis Hindi questions sometimes
produce incoherent output due to the base model's limited Hindi fluency at
0.5B scale combined with sampling variance. This is diagnosed in
`docs/approach.md` rather than hidden, along with why oversampling the
training set was considered and deliberately not pursued (the
safety-critical instance of the problem no longer depends on model
quality once the guardrail is in place).

## Known Non-Blocking Warnings

- `huggingface_hub` symlink warning on Windows — cosmetic, caused by
  Windows requiring admin/Developer Mode for symlinks; caching still
  works, just uses slightly more disk space
- `attention_mask`/`pad_token` warning during inference — cosmetic; Qwen's
  pad token defaults to its eos token, generation is unaffected. To
  silence it, pass an explicit `attention_mask` from
  `tokenizer(..., return_tensors="pt")` instead of only `input_ids` in
  `inference.py`'s `generate()` call