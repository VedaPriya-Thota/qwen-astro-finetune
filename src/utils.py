"""
Shared utilities for the Vedaz Qwen fine-tuning pipeline.

Kept intentionally small: config loading, logging setup, and seeding.
Nothing here is specific to preprocessing/training/inference — that logic
stays in its own script.
"""

import json
import logging
import random
import sys
from pathlib import Path

import yaml


def load_config(config_path: str = "config/train_config.yaml") -> dict:
    """Load the YAML config used by every script in this pipeline."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found at '{path}'. "
            f"Run scripts from the project root (qwen-astro-finetune/)."
        )
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_logger(name: str) -> logging.Logger:
    """Consistent logger across all scripts — prints to stdout with timestamps."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def set_seed(seed: int = 42) -> None:
    """Seed Python/NumPy/Torch RNGs for reproducibility."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def read_jsonl(path: str) -> list:
    """Read a .jsonl file into a list of dicts."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: list, path: str) -> None:
    """Write a list of dicts to a .jsonl file, one JSON object per line."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")