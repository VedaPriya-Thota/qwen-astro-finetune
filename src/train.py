"""
LoRA SFT training for Qwen2.5-0.5B-Instruct on the Vedaz astrology dataset.

Loads the already-tokenized train/eval jsonl files produced by
src/preprocess.py (each record already has input_ids/labels/attention_mask
with non-assistant tokens masked to -100), wraps the base model with a
LoRA adapter, and trains with Hugging Face's Trainer.

Run from the project root (after preprocess.py has been run once):
    python src/train.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import get_logger, load_config, read_jsonl, set_seed  # noqa: E402

logger = get_logger("train")


def main():
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    config = load_config()
    set_seed(config["training"]["seed"])

    model_id = config["model"]["base_model_id"]
    cache_dir = config["model"]["cache_dir"]

    # ---- Load tokenizer + base model ----
    logger.info(f"Loading base model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = getattr(torch, config["model"]["torch_dtype"])
    model = AutoModelForCausalLM.from_pretrained(
        model_id, cache_dir=cache_dir, torch_dtype=dtype
    )
    model.config.use_cache = False  # required for gradient checkpointing / training stability

    # ---- Wrap with LoRA ----
    lora_cfg = config["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        target_modules=lora_cfg["target_modules"],
        task_type=lora_cfg["task_type"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()  # sanity-check: should be ~0.5-2% of total params

    # ---- Load preprocessed data ----
    processed_dir = Path(config["data"]["processed_dir"])
    train_records = read_jsonl(str(processed_dir / config["data"]["train_file"]))
    eval_records = read_jsonl(str(processed_dir / config["data"]["eval_file"]))
    logger.info(f"Loaded {len(train_records)} train / {len(eval_records)} eval examples.")

    train_dataset = Dataset.from_list(train_records)
    eval_dataset = Dataset.from_list(eval_records)

    # Pads input_ids/attention_mask/labels to the same length per batch;
    # label padding uses -100 so padded positions don't contribute to loss.
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, padding=True, label_pad_token_id=-100
    )

    # ---- Training arguments ----
    t_cfg = config["training"]
    training_args = TrainingArguments(
        output_dir=t_cfg["output_dir"],
        logging_dir=t_cfg["logging_dir"],
        num_train_epochs=t_cfg["num_train_epochs"],
        per_device_train_batch_size=t_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=t_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=t_cfg["gradient_accumulation_steps"],
        learning_rate=t_cfg["learning_rate"],
        lr_scheduler_type=t_cfg["lr_scheduler_type"],
        warmup_ratio=t_cfg["warmup_ratio"],
        weight_decay=t_cfg["weight_decay"],
        logging_steps=t_cfg["logging_steps"],
        save_strategy=t_cfg["save_strategy"],
        eval_strategy=t_cfg["eval_strategy"],
        save_total_limit=t_cfg["save_total_limit"],
        report_to=t_cfg["report_to"],
        seed=t_cfg["seed"],
        use_cpu=True,
        fp16=False,   # no GPU — fp16 autocast isn't beneficial/stable on CPU
        bf16=False,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training complete.")

    # ---- Save adapter only (small, few MB — not the merged full model) ----
    final_path = Path(t_cfg["output_dir"]) / "final"
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    logger.info(f"Saved LoRA adapter + tokenizer to {final_path}")


if __name__ == "__main__":
    main()