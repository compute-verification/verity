"""verity_vllm: Verity for vLLM serving.

:class:`LLM` (``from verity_vllm import LLM``) is a pinned vLLM engine that is constructed and called like ``vllm.LLM``;
the ``verity-vllm`` command (:mod:`verity_vllm.pipeline.cli`) runs the Build, Match and Commit stages of a row and every
other tool.  Importing the package imports neither torch nor vLLM; ``LLM`` loads them when it is first used.
"""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0"

CAPTURE_FORMAT_VERSION = 1

__all__ = ["LLM"]


def __getattr__(name: str) -> Any:
    if name == "LLM":
        from verity_vllm.pipeline.llm import LLM
        return LLM
    raise AttributeError(f"module 'verity_vllm' has no attribute {name!r}")
