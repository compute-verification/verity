"""A stock vLLM engine at the pin, with no protocol on: what code outside this distribution builds beside, or instead of,
:class:`verity_vllm.LLM`.

`engine` is an in-process V1 ``LLMEngine`` built from ``EngineArgs``: its caller adds requests and steps it, and with
``VLLM_ENABLE_V1_MULTIPROCESSING=0`` (`engine.env.ENGINE`) `engine.handle` reaches its scheduler and model runner.
`sampling` is vLLM's ``SamplingParams``, `version` vLLM's version and `build` its commit, one of `BUILDS`.  Each imports
vLLM when called, so this module imports without it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from verity_vllm.config import VLLM

if TYPE_CHECKING:
    from vllm import SamplingParams
    from vllm.v1.engine.llm_engine import LLMEngine

__all__ = ["BUILDS", "build", "engine", "sampling", "version"]

#: The vLLM builds this tree installs, by short commit: the pin (`config.VLLM`), which every row, profile and Definition
#: is derived at and served on, and the build GLM-5.3's serving benchmark may serve stock on instead
#: (``ops/vllm_env.sh``, ``SERVE_VLLM`` in ``benchmarks/serve_tax/run.sh``).
BUILDS = (VLLM, "52aeb5806")


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
    """The installed vLLM's ``__version__``, which carries ``+g<commit>`` at the pin (`config.VLLM`) and no commit at a
    tagged build."""
    import vllm

    return vllm.__version__


def build() -> str:
    """The installed vLLM's short commit, as `BUILDS` names it."""
    from vllm._version import __commit_id__

    return __commit_id__.removeprefix("g")
