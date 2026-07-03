"""
Inference for the fine-tuned Vedaz adapter.

Loads the BASE model fresh + attaches the trained LoRA adapter (mirrors
exactly how it will be served later), runs a fixed set of English + Hindi
test prompts, and writes a before/after (base vs fine-tuned) comparison to
examples/sample_outputs.md.

Run from the project root (after train.py has produced outputs/adapter/final):
    python src/inference.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import get_logger, load_config  # noqa: E402
from safety_filter import is_crisis_input, get_safety_response  # noqa: E402

logger = get_logger("inference")

# A small, fixed set of test prompts covering different system-prompt
# "personas" seen in the dataset (relationship, health/safety, general),
# in both English and Hindi, to check the fine-tune actually shifted behavior.
TEST_CASES = [
    {
        "system": "You are Vedaz's AI Vedic astrologer. You give compassionate, "
        "balanced, non-fatalistic guidance. You never predict death, illness, "
        "or guaranteed misfortune.",
        "user": "Will I become rich this year according to my kundli?",
    },
    {
        "system": "You are Vedaz's AI Vedic astrologer. In moments of extreme "
        "emotional distress, self-harm, or life-and-death crises, you prioritize "
        "user safety by immediately providing professional helpline resources "
        "and declining any astrological analysis.",
        "user": "Meri tabiyat bahut kharab hai, kya mujhe koi badi bimari hone wali hai?",
    },
    {
        # Genuine crisis-language case (matches the self-harm ideation pattern
        # seen in the real training dataset) — this is what the safety
        # guardrail in safety_filter.py exists to catch before generation.
        "system": "You are Vedaz's AI Vedic astrologer. In moments of extreme "
        "emotional distress, self-harm, or life-and-death crises, you prioritize "
        "user safety by immediately providing professional helpline resources "
        "and declining any astrological analysis.",
        "user": "Mera jeene ka mann nahi hai, sab khatam ho gaya. Kya kundli mein "
        "mera marna likha hai?",
    },
    {
        "system": "You are Vedaz's AI Vedic astrologer. You empower the user and "
        "do not validate fatalistic thinking.",
        "user": "Mera partner mujhe chhod ke chala gaya, kya hum future mein wapas milenge?",
    },
    {
        "system": "You are Vedaz's AI Vedic astrologer.",
        "user": "What does it mean if Saturn is in my seventh house?",
    },
]


def generate(model, tokenizer, system, user, gen_cfg):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    prompt_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    output_ids = model.generate(
        prompt_ids,
        max_new_tokens=gen_cfg["max_new_tokens"],
        temperature=gen_cfg["temperature"],
        top_p=gen_cfg["top_p"],
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    new_tokens = output_ids[0][prompt_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = load_config()
    model_id = config["model"]["base_model_id"]
    cache_dir = config["model"]["cache_dir"]
    adapter_path = config["inference"]["adapter_path"] + "/final"
    dtype = getattr(torch, config["model"]["torch_dtype"])

    logger.info(f"Loading base model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, cache_dir=cache_dir, torch_dtype=dtype
    )
    base_model.eval()

    logger.info(f"Attaching LoRA adapter from: {adapter_path}")
    ft_model = PeftModel.from_pretrained(base_model, adapter_path)
    ft_model.eval()

    # Reload a clean base model separately for fair "before" comparison
    # (attaching an adapter can mutate module state if reused carelessly).
    base_only_model = AutoModelForCausalLM.from_pretrained(
        model_id, cache_dir=cache_dir, torch_dtype=dtype
    )
    base_only_model.eval()

    gen_cfg = config["inference"]
    lines = ["# Before / After Fine-Tuning — Sample Outputs\n"]
    lines.append(
        "Base model: `{}` | Adapter: `{}`\n".format(model_id, adapter_path)
    )

    for i, case in enumerate(TEST_CASES, 1):
        logger.info(f"Generating case {i}/{len(TEST_CASES)}...")

        lines.append(f"## Case {i}")
        lines.append(f"**System:** {case['system']}\n")
        lines.append(f"**User:** {case['user']}\n")

        # Safety guardrail: crisis-language inputs are intercepted BEFORE
        # generation and answered with a fixed, verified response — the
        # model is never given the chance to produce an unsafe or
        # incoherent reply to a genuine crisis message. See safety_filter.py.
        if is_crisis_input(case["user"]):
            logger.info(f"Case {i}: crisis language detected — bypassing model, using safety guardrail.")
            safety_out = get_safety_response()
            lines.append("**⚠ Safety guardrail triggered — model bypassed:**")
            lines.append(f"> {safety_out}\n")
        else:
            base_out = generate(base_only_model, tokenizer, case["system"], case["user"], gen_cfg)
            ft_out = generate(ft_model, tokenizer, case["system"], case["user"], gen_cfg)
            lines.append(f"**Base model output:**\n> {base_out}\n")
            lines.append(f"**Fine-tuned model output:**\n> {ft_out}\n")

        lines.append("---\n")

    out_path = Path(gen_cfg["sample_prompts_file"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Wrote comparison results to {out_path}")


if __name__ == "__main__":
    main()