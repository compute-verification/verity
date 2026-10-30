"""The environment variables the store's Tool declarations read, declared once.

``PY`` names the interpreter a vLLM stage's workload runs under (the pods' job environment sets it), whose torch and vLLM
versions ``vllm_tools`` probes as the stage's conditions.
"""
from __future__ import annotations

import os


def workload_python(default: str) -> str:
    """``$PY``, else `default`."""
    return os.environ.get("PY") or default
