"""The POUS challenger inside the serving process, installed through vLLM's `worker_extension_cls` seam and driven with
`collective_rpc`: `pous_info`, `pous_start_responder`, `pous_stop_responder`, `pous_leaves`.

The responder is native (`pous_native.Responder`): a C++ thread that answers each challenge by copying the block from `C` in device
memory on its own highest-priority, non-blocking stream into pinned host memory and sending it, so no Python, GIL or serving-stream
ordering sits on the answer's path; it polls its socket and the copy's completion instead of sleeping, on its own core (`cpu`) when
given.
"""
from __future__ import annotations

from typing import Any

EXTENSION = "verity_vllm.engine.pous.worker.PousWorkerExtension"


class PousWorkerExtension:
    """Mixed into vLLM's Worker; `self` is the Worker of the one rank."""

    def pous_info(self) -> dict[str, Any]:
        from verity_vllm.engine.pous import quant

        return quant.current_store().info()

    def pous_start_responder(self, host: str = "127.0.0.1", port: int = 0, cpu: int = -1) -> int:
        import torch

        from verity_vllm.engine.pous import native, quant

        store = quant.current_store()
        if getattr(self, "_pous_responder", None) is not None:
            return self._pous_responder.port()
        responder = native.ext().Responder(store.c.data_ptr(), store.blocks, store.c.shape[-1], torch.cuda.current_device(),
                                            host, int(port), int(cpu))
        responder.start()
        self._pous_responder = responder
        return responder.port()

    def pous_stop_responder(self) -> dict[str, int]:
        responder = getattr(self, "_pous_responder", None)
        if responder is None:
            return {"answered": 0, "pings": 0}
        responder.stop()
        self._pous_responder = None
        return {"answered": int(responder.answered()), "pings": int(responder.pings())}

    def pous_leaves(self) -> bytes:
        """The reference leaf digest of every block of `C`, concatenated in block order: the server's root is their
        `verity_pous.verifier.MerkleTree` root, which the verifier checks against its own `vk` at setup."""
        from verity_vllm.engine.pous import quant

        return b"".join(quant.current_store().leaves())
