"""The perplexity bench's environment variables, each declared once in `DECLARED` with its default and what it is, and
read only through `get`. Standard library only: `job.sh` starts `run.py eval` with the GPU node's own Python."""
from __future__ import annotations

import os

__all__ = ["DECLARED", "get"]

#: name -> (the value an unset variable takes, what it is)
DECLARED: dict[str, tuple[str, str]] = {
    "PPL_DATA": ("/workspace/ppl-data", "where the staged datasets (datasets.json) are: `--data`'s default"),
    "HF_HOME": ("/workspace/hf", "the Hugging Face cache a model's snapshot is read from"),
    "CUDA_VISIBLE_DEVICES": ("", "the GPUs an eval may use, comma-separated (gpu-lease sets it); empty, every GPU torch sees"),
    "RESEARCH_RUN_ID": ("", "the research run's id (research's runner sets it): a result's `env.run`"),
    "GPU_LEASE_WHO": ("ppl-bench", "who a submitted run's GPU lease names"),
}


def get(name: str) -> str:
    """The value of the declared variable `name`: the environment's, or its default."""
    default, _ = DECLARED[name]
    return os.environ.get(name, default)
