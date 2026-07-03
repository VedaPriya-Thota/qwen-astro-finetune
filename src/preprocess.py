"""
Preprocessing for the Vedaz astrology chat dataset.

What this script does, and why:

1. Parses data/raw/chat_data.json.
   NOTE: the raw file is NOT valid JSON as a whole — it is a sequence of
   individual JSON conversation objects separated by commas, with no
   enclosing `[ ... ]` array. json.load() on the whole file fails. We
   parse it by streaming through the file with json.JSONDecoder.raw_decode,
   which reads one JSON object at a time regardless of the missing wrapper.
   This is a data-quality fix, not a design choice — the file is fixed
   once here rather than working around it in every downstream script.

2. Applies the Qwen2.5 chat template to each conversation via
   tokenizer.apply_chat_template(), so the exact token sequence used at
   training time matches the exact token sequence used at inference time.

3. Builds `labels` aligned to `input_ids`, masking every token that is NOT
   part of an assistant turn (system + user tokens + special tokens outside
   assistant spans -> -100, ignored by the loss). This is what makes it a
   proper SFT run rather than accidental next-token prediction over the
   whole conversation (including the user's own questions).

4. Splits into train/eval and writes data/processed/{train,eval}.jsonl.

Run from the project root:
    python src/preprocess.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import get_logger, load_config, set_seed, write_jsonl  # noqa: E402

logger = get_logger("preprocess")


def parse_malformed_json_stream(raw_text: str) -> list:
    """
    Parse a file containing multiple JSON objects concatenated with commas
    but no enclosing array. Skips whitespace and stray commas between
    objects. Raises with a clear message if a chunk truly isn't valid JSON.
    """
    decoder = json.JSONDecoder()
    idx = 0
    n = len(raw_text)
    records = []

    while idx < n:
        # skip whitespace and separator commas between objects
        while idx < n and raw_text[idx] in " \n\t\r,":
            idx += 1
        if idx >= n:
            break
        try:
            obj, end = decoder.raw_decode(raw_text, idx)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse a JSON object starting near character {idx}: {e}"
            ) from e
        records.append(obj)
        idx = end

    return records


def validate_conversation(conv: dict, idx: int) -> bool:
    """Basic structural sanity checks. Logs and skips malformed conversations
    rather than crashing the whole pipeline on one bad record."""
    if "messages" not in conv or not isinstance(conv["messages"], list):
        logger.warning(f"Record {idx}: missing/invalid 'messages' field — skipping.")
        return False
    roles = [m.get("role") for m in conv["messages"]]
    if "assistant" not in roles:
        logger.warning(f"Record {idx}: no assistant turn — skipping.")
        return False
    for m in conv["messages"]:
        if "role" not in m or "content" not in m:
            logger.warning(f"Record {idx}: message missing role/content — skipping.")
            return False
    return True


def build_labels(tokenizer, messages: list, max_length: int) -> dict:
    """
    Tokenize a conversation with the model's chat template and build labels
    where only assistant-turn tokens are trainable (-100 elsewhere).

    Strategy: incrementally apply the chat template turn-by-turn. After each
    assistant turn is added, we tokenize the "prefix so far" and the "prefix
    including this assistant turn." The token span between the two is the
    assistant's own tokens (plus its closing special tokens) -> unmasked.
    Everything else stays masked. This correctly handles multi-turn examples
    where several assistant turns exist in one conversation.
    """
    input_ids_full = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )

    labels = [-100] * len(input_ids_full)

    # Walk through messages, tracking cumulative tokenized prefix length
    running_messages = []
    prev_len = 0
    for msg in messages:
        running_messages.append(msg)
        prefix_ids = tokenizer.apply_chat_template(
            running_messages, tokenize=True, add_generation_prompt=False
        )
        cur_len = len(prefix_ids)

        if msg["role"] == "assistant":
            # Unmask the token span belonging to this assistant turn
            for i in range(prev_len, cur_len):
                if i < len(labels):
                    labels[i] = input_ids_full[i]

        prev_len = cur_len

    if len(input_ids_full) > max_length:
        input_ids_full = input_ids_full[:max_length]
        labels = labels[:max_length]

    return {
        "input_ids": input_ids_full,
        "labels": labels,
        "attention_mask": [1] * len(input_ids_full),
    }


def main():
    config = load_config()
    set_seed(config["data"]["seed"])

    raw_path = Path(config["data"]["raw_path"])
    processed_dir = Path(config["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Reading raw dataset from {raw_path}")
    raw_text = raw_path.read_text(encoding="utf-8")
    conversations = parse_malformed_json_stream(raw_text)
    logger.info(f"Parsed {len(conversations)} conversation objects from raw file.")

    valid_conversations = [
        c for i, c in enumerate(conversations) if validate_conversation(c, i)
    ]
    logger.info(
        f"{len(valid_conversations)}/{len(conversations)} conversations passed validation."
    )

    # --- Tokenizer-dependent step ---
    # Requires downloading the Qwen2.5 tokenizer from Hugging Face Hub the
    # first time it's run (needs internet access on your machine).
    from transformers import AutoTokenizer

    model_id = config["model"]["base_model_id"]
    cache_dir = config["model"]["cache_dir"]
    logger.info(f"Loading tokenizer for {model_id} (cache_dir={cache_dir}) ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, cache_dir=cache_dir, trust_remote_code=config["model"]["trust_remote_code"]
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_length = config["data"]["max_seq_length"]
    processed = []
    for i, conv in enumerate(valid_conversations):
        try:
            example = build_labels(tokenizer, conv["messages"], max_length)
            n_trainable = sum(1 for l in example["labels"] if l != -100)
            if n_trainable == 0:
                logger.warning(f"Record {i}: zero trainable tokens after masking — skipping.")
                continue
            processed.append(example)
        except Exception as e:
            logger.warning(f"Record {i}: failed during tokenization/masking ({e}) — skipping.")

    logger.info(f"Successfully tokenized {len(processed)} conversations.")

    # --- Train/eval split ---
    eval_size = min(config["data"]["eval_split_size"], max(1, len(processed) // 10))
    eval_set = processed[:eval_size]
    train_set = processed[eval_size:]

    train_path = processed_dir / config["data"]["train_file"]
    eval_path = processed_dir / config["data"]["eval_file"]
    write_jsonl(train_set, str(train_path))
    write_jsonl(eval_set, str(eval_path))

    logger.info(f"Wrote {len(train_set)} training examples -> {train_path}")
    logger.info(f"Wrote {len(eval_set)} eval examples -> {eval_path}")
    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()