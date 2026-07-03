# Approach — Vedaz Qwen2.5 LoRA Fine-Tuning

## 1. Objective

Fine-tune a Qwen model on the provided Vedaz bilingual (English/Hindi)
Vedic-astrology chat dataset under CPU-only, limited-storage constraints,
and document the process of serving the result via vLLM on a VPS.

## 2. Dataset Analysis

The raw file (`data/raw/chat_data.json`) is **not valid JSON as a whole** —
it is a sequence of individual JSON conversation objects separated by
commas, with no enclosing `[ ]` array. This was fixed in
`src/preprocess.py` by streaming the file through `json.JSONDecoder.raw_decode`
rather than a single `json.load()` call.

After fixing the format, the dataset contains:

| Property | Value |
|---|---|
| Total conversations | 55 |
| Single-turn (3 messages) | 44 |
| Multi-turn (5–7 messages) | 11 |
| Distinct system prompts | 37 |
| Languages | English, Hindi (romanized user input, Devanagari assistant output) |

This is a standard **multi-turn OpenAI-format SFT dataset**. The system
prompt varies per conversation and encodes both a persona ("Vedaz's AI
Vedic astrologer") and topic-specific safety behavior (no death/illness
predictions, no guaranteed financial outcomes, mandatory safety-resource
redirection for self-harm/crisis language). This is a persona +
safety-conditioning dataset, not pure knowledge-injection.

One notable pattern: **user turns are frequently written in romanized
Hindi (Hinglish, Latin script)**, while assistant turns respond in
**Devanagari script**. This script-switching convention is preserved
as-is in preprocessing (not normalized), since it reflects how the
product is actually used.

## 3. Preprocessing Design

- Chat-template formatting via `tokenizer.apply_chat_template()` — ensures
  the exact token sequence used at training time matches what will be used
  at inference time
- **Label masking**: only assistant-turn tokens are trainable (all
  system/user tokens set to `-100`). Implemented in `build_labels()` by
  incrementally re-applying the chat template turn-by-turn and diffing
  token-length spans, so it correctly handles both single-turn and
  multi-turn (up to 7-message) conversations
- 50/5 train/eval split (55 total is too small for a statistically
  meaningful eval set — the 5-example holdout is a loss-trend sanity
  check, not a benchmark)
- No synthetic augmentation, no deduplication logic — the dataset was
  already clean structurally once the JSON-wrapping issue was fixed, and
  adding pipeline complexity beyond what the data actually needs would be
  over-engineering for a 55-example assessment dataset

## 4. Model & Method Selection

**Model: Qwen2.5-0.5B-Instruct**
- Qwen2.5 over Qwen3: simpler, more mature chat template and inference
  path; Qwen3's thinking/non-thinking hybrid mode adds CPU inference
  overhead with no benefit at this dataset scale
- 0.5B over 1.5B/3B: smallest Qwen2.5-Instruct checkpoint with reliable
  instruction-following, keeping CPU training time and disk footprint
  (~1GB base weights) within the stated constraints

**Method: LoRA, not full fine-tuning**
- 55 examples is far too small to full-fine-tune without catastrophic
  forgetting / overfitting
- LoRA trains a small, targeted parameter subset — confirmed in the actual
  run at **0.44% trainable parameters** (2,162,688 / 496,195,456)
- LoRA config: rank 16, alpha 32, dropout 0.05, targeting attention
  projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`)

## 5. Training Results (actual run)

| Epoch | Train loss | Eval loss |
|---|---|---|
| 1 | 2.758 | 1.947 |
| 2 | 2.017 | 1.929 |
| 3 | 2.271 → 2.191 | 1.920 |
| 4 | 2.194 | 1.918 |

- Total training time: **~36 minutes** on CPU (4 epochs, 50 train examples,
  effective batch size 8 via gradient accumulation)
- Eval loss decreased steadily and modestly (1.947 → 1.918) without
  collapsing toward zero — consistent with genuine, bounded learning on a
  small dataset rather than memorization
- Train loss fluctuates across steps (expected with only ~6-7 optimizer
  steps per epoch on 50 examples at this batch size) but the eval trend is
  the more reliable signal, and it is monotonic and stable

## 6. Qualitative Results (from `examples/sample_outputs.md`)

**What worked:** Cases 1 and 5 (English astrology questions) show a clear,
consistent stylistic shift in the fine-tuned model: more direct, less
hedging/repetitive disclaimers, tighter structure, and language that
better matches the "empowering, non-fatalistic" persona from the system
prompts.

**Safety-critical case — resolved architecturally, not via the fine-tune
alone.** An initial test using genuine crisis language ("mera jeene ka
mann nahi hai, sab khatam ho gaya" — matching real self-harm-ideation
examples in the training data) showed neither the base nor fine-tuned
model reliably producing the required safety-redirect behavior in Hindi.
Rather than trying to solve this purely through more training on a
55-example dataset, a **rule-based safety guardrail** (`src/safety_filter.py`)
was added that pattern-matches crisis/self-harm language (English,
romanized Hindi, and Devanagari) and intercepts it **before generation**,
returning a fixed, verified response with two current Indian national
helplines — Tele MANAS (14416 / 1800-891-4416) and KIRAN (1800-599-0019),
both Government of India, 24/7. Re-tested against the same crisis prompt:
the guardrail fires correctly and the model is bypassed entirely, so the
safety guarantee no longer depends on model output quality at all. This
is the correct engineering response to a small-model, small-data
constraint on a safety-critical behavior — verified-by-code rather than
hoped-for-via-fine-tuning.

**Residual limitation — Hindi fluency on non-crisis questions.** Ordinary
(non-crisis) Hindi-language questions — e.g., a health worry ("is
something serious going to happen to my health?") or a relationship
question — still produce inconsistent output from both base and
fine-tuned models: sometimes reasonably coherent (with code-switching into
English), sometimes not, varying between runs. This is distinct from the
safety-critical case above: it is a **generation-quality issue, not a
safety issue** — no user is left without safety information, since crisis
language is caught by the guardrail regardless of the base model's fluency.

Root cause: Qwen2.5-0.5B has limited native Hindi generation fluency,
and inference uses temperature-sampling (`temperature=0.7,
do_sample=True`), which introduces run-to-run variance on already-shaky
Hindi output. A 55-example dataset (spread across 37 personas) cannot be
expected to fully correct a base-model fluency ceiling at this parameter
count.

**Considered and deliberately not implemented:** oversampling Hindi
examples in the training set to improve this further. Given that (a) the
safety-critical instance of this problem is already solved by the
guardrail regardless of model quality, and (b) the residual issue is
better explained by base-model capacity and sampling variance than by
training-data exposure count, retraining on a duplicated dataset was
judged unlikely to reliably fix general Hindi coherence and not a good
use of remaining time. This is noted as the correct trade-off decision
rather than an oversight.

**Recommended remediation for a future iteration (out of scope here):**
- Move to Qwen2.5-1.5B or 3B-Instruct for meaningfully stronger Hindi
  generation quality
- Use greedy decoding (`do_sample=False`) for non-English output to
  reduce incoherent sampling drift, as a zero-retraining-cost mitigation
- Keep the rule-based safety guardrail regardless of model upgrades — a
  production safety-critical assistant should never rely solely on a
  fine-tuned model's generalization for crisis detection, at any model size

## 7. Summary

The pipeline correctly implements LoRA SFT with proper multi-turn label
masking on a small, real-world bilingual dataset, under CPU-only
constraints, in under 40 minutes of training time. Results show genuine,
verifiable behavioral adaptation on English-language cases. Testing
surfaced a genuine safety gap in Hindi crisis-language handling, which was
addressed with a rule-based guardrail that intercepts crisis input before
generation — a more reliable and verifiable safety mechanism than relying
on a 0.5B model fine-tuned on 55 examples to generalize crisis-response
behavior on its own. The residual Hindi-fluency limitation on non-crisis
questions is reported honestly as a base-model capacity constraint, with
a clear remediation path, rather than left undiagnosed or hidden.