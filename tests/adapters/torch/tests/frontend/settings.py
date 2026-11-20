"""The frontend tests' settings registry (`tests/lint/test_repo_env_reads.py`): each variable a test here reads, declared once
with its default and what it is, and read only through `get` (or `offline`, for the three a real-`transformers` test sets
when unset). Standard library only: the opt-in real-`transformers` runs start these tests under another tree's Python."""

from __future__ import annotations

import os

__all__ = ["DECLARED", "get", "offline"]

#: name -> (the value an unset or empty variable takes, what it is)
DECLARED: dict[str, tuple[str, str]] = {
    "HF_HUB_OFFLINE": ("1", "Hugging Face's hub stays offline: the real-`transformers` tests build tiny random configs"),
    "TRANSFORMERS_OFFLINE": ("1", "`transformers` stays offline, as HF_HUB_OFFLINE"),
    "TOKENIZERS_PARALLELISM": ("false", "the tokenizers' thread pool stays off"),
    "HARDEN_OUT": ("", "test_harden_moe: the directory its moe_results.json is written to; empty, none is"),
}


def get(name: str) -> str:
    """The declared variable `name`'s value in the environment, or its default where it is unset or empty."""
    default, _ = DECLARED[name]
    return os.environ.get(name) or default


def offline() -> None:
    """Set HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE and TOKENIZERS_PARALLELISM to their defaults where they are unset."""
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "TOKENIZERS_PARALLELISM"):
        os.environ.setdefault(name, DECLARED[name][0])
