"""The perplexity bench's environment variables, each declared once in `DECLARED` with its default and what it is, and
read only through `get`. Standard library only: `job.sh` starts `run.py eval` with the GPU node's own Python."""
from __future__ import annotations

import os

__all__ = ["DECLARED", "get", "given"]

#: name -> (the value an unset variable takes, what it is)
DECLARED: dict[str, tuple[str, str]] = {
    "PPL_DATA": ("/workspace/ppl-data", "where the staged datasets (datasets.json) are: `--data`'s default"),
    "HF_HOME": ("/workspace/hf", "the Hugging Face cache a model's snapshot is read from"),
    "CUDA_VISIBLE_DEVICES": ("", "the GPUs an eval may use, comma-separated (gpu-lease sets it); empty, every GPU torch sees"),
    "RESEARCH_RUN_ID": ("", "the research run's id (research's runner sets it): a result's `env.run`"),
    "RESEARCH_SOURCE_SHA": ("", "the commit `research run` shipped, which results.json records"),
    "GPU_LEASE_WHO": ("ppl-bench", "who a submitted run's GPU lease names"),
    "PEARLC_KERNEL": ("", "the `pearl_c` step's kernel directory (benchmarks/harness/setup/pearl_c_sm90.sh exports it); "
                          "empty, this tree's pearl_c_sm90"),
    "PEARLC_CUBIN": ("", "the `pearl_c` step's sm_90a cubin, which benchmarks/harness/setup/pearl_c_sm90.sh builds and exports"),
}


def get(name: str) -> str:
    """The value of the declared variable `name`: the environment's, or its default."""
    default, _ = DECLARED[name]
    return os.environ.get(name, default)


def given(name: str) -> str | None:
    """The environment's value of the declared variable `name`, or None where it is unset: for a flag whose unset
    default is another source's, not the declared one."""
    DECLARED[name]
    return os.environ.get(name)
