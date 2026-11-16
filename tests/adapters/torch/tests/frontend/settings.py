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
    "REALHF_STEPS": ("4", "test_derive_realhf: greedy decode steps"),
    "REALHF_EVAL": ("1", "test_derive_realhf: 0 derives only, with no evaluation against torch"),
    "REALHF_ONLY": ("", "test_derive_realhf: run only the cases whose id contains it"),
    "REALHF_OUT": ("/tmp/realhf/realhf.jsonl", "test_derive_realhf: the JSONL file its rows are appended to"),
    "REALHF_POS": ("", "test_derive_realhf: `static` passes position_ids=arange(T)[None] explicitly"),
    "REALHF_HEAD": ("", "test_derive_realhf: the commit its rows name, in place of `git rev-parse`"),
    "REALHF_PROBE": ("0", "test_derive_realhf: how many of its PROBES' transformers patches apply"),
    "REALHF_GROUP": ("", "test_derive_hf5b_realhf: the case group it runs; empty, every group"),
    "XFER_OUT": ("/tmp/xfer/transfer-derive.jsonl", "test_derive_transfer: the JSONL file its rows are appended to"),
    "XFER_HEAD": ("", "test_derive_transfer: the commit its rows name, in place of `git rev-parse`"),
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
