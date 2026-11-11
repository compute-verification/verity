"""A stock vLLM engine at the pin, with no protocol on: what code outside this distribution builds beside, or instead of,
:class:`verity_vllm.LLM`.

`engine` is an in-process V1 ``LLMEngine`` built from ``EngineArgs``: its caller adds requests and steps it, and with
``VLLM_ENABLE_V1_MULTIPROCESSING=0`` (`engine.env.ENGINE`) `engine.handle` reaches its scheduler and model runner.
`sampling` is vLLM's ``SamplingParams`` and `version` vLLM's version.  Each imports vLLM when called, so this module
imports without it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm import SamplingParams
    from vllm.v1.engine.llm_engine import LLMEngine

__all__ = ["engine", "sampling", "version"]


def engine(model: str, **args: Any) -> LLMEngine:
    """An in-process V1 ``LLMEngine`` of ``model`` (a hub id or a local directory), built from
    ``EngineArgs(model=model, **args)``."""
    from vllm import EngineArgs
    from vllm.v1.engine.llm_engine import LLMEngine

    return LLMEngine.from_engine_args(EngineArgs(model=model, **args))


def sampling(**params: Any) -> SamplingParams:
    """vLLM's ``SamplingParams(**params)``."""
    from vllm import SamplingParams

    return SamplingParams(**params)


def version() -> str:
    """The installed vLLM's ``__version__``, which carries ``+g<commit>`` (`config.VLLM`)."""
    import vllm

    return vllm.__version__
