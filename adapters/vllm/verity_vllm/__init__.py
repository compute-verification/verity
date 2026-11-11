"""verity_vllm: Verity for vLLM serving.

:class:`LLM` (``from verity_vllm import LLM``) is a pinned vLLM engine that is constructed and called like ``vllm.LLM``;
:class:`EnginePin` is the checkpoint pin it runs, :class:`EngineSpec` the document that describes one (`LLM.from_spec`),
and :class:`TargetProfile` the declared target it is built for.  The ``verity-vllm`` command
(:mod:`verity_vllm.pipeline.cli`) runs the Build, Match and Commit stages of a row and every other tool.  Importing the
package imports neither torch nor vLLM; each name loads its module when it is first used.

Outside this distribution, code imports only the public modules ``tests/test_vllm_public_api.py`` lists, and from each
only the names its ``__all__`` lists.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from verity_vllm.config import EnginePin
    from verity_vllm.engine.spec import EngineSpec
    from verity_vllm.llm import LLM
    from verity_vllm.program.frontend.target_profile import TargetProfile

__version__ = "0.1.0"

__all__ = ["EnginePin", "EngineSpec", "LLM", "TargetProfile"]

_HOME = {"EnginePin": "verity_vllm.config", "EngineSpec": "verity_vllm.engine.spec", "LLM": "verity_vllm.llm",
         "TargetProfile": "verity_vllm.program.frontend.target_profile"}


def __getattr__(name: str) -> Any:
    if name in _HOME:
        return getattr(import_module(_HOME[name]), name)
    raise AttributeError(f"module 'verity_vllm' has no attribute {name!r}")
