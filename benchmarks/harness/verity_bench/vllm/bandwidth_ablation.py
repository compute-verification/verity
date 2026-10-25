"""Ablations of PoUS's bandwidth-audit arms (memory accounting's ``benchmarks.vllm_encoded.bench_arm``), so one spec times
where the audit's time over ``bandwidth-plain`` goes (``llama31-8b-fp8-1gpu-bandwidth-ablation``). Each is ``AUDIT`` or
``PLAIN`` with one part changed, through a knob the arm reads when it installs (its environment variables, set for the
install only) or with its fill-sk16 filler paused once the engine is ready (``tu_pause``: no chunk launched after it):

~~~text
bandwidth-audit-no-kv-receipts   AUDIT, VERITY_VLLM_BENCH_KV_RECEIPTS=off: the step's commit takes the weights' receipts only
bandwidth-audit-k1               AUDIT, VERITY_VLLM_BENCH_CODEC_ROUNDS=1: one codec round where the audit runs three
bandwidth-audit-no-filler        AUDIT with its filler paused
bandwidth-plain-no-filler        PLAIN with its filler paused
~~~

None of these parts changes a served value, so every ablation is token-controlled by ``bandwidth-plain``. They measure
cost only: an audit without its filler or its KV receipts isn't the protocol.
"""

from __future__ import annotations

import os
from typing import Any, Mapping

CONTROL = "bandwidth-plain"


class _NoFiller:
    """The arm's installed state with its filler paused as soon as it starts."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def ready(self, target: Any) -> None:
        self.inner.ready(target)
        self.inner.topup.so.tu_pause(1)

    def close(self) -> None:
        self.inner.close()


class Ablation:
    precision = "fp8"
    protocols: tuple[str, ...] = ()
    resident = False
    control = CONTROL

    def __init__(self, id: str, *, encoded: bool, env: Mapping[str, str] | None = None, filler: bool = True) -> None:
        self.id, self.encoded, self.env, self.filler = id, encoded, dict(env or {}), filler

    def base(self) -> Any:
        from benchmarks.vllm_encoded import bench_arm

        return bench_arm.AUDIT if self.encoded else bench_arm.PLAIN

    def install(self, target: Any) -> Any:
        saved = {name: os.environ.get(name) for name in self.env}
        os.environ.update(self.env)
        try:
            state = self.base().install(target)
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        return state if self.filler else _NoFiller(state)

    def gate(self, target: Any) -> dict[str, Any]:
        return {**self.base().gate(target), "ablation": {"env": dict(self.env), "filler": self.filler}}

    def spans(self, target: Any) -> dict[str, float]:
        return self.base().spans(target)

    def extras(self, target: Any) -> dict[str, float]:
        return self.base().extras(target)

    def kernel_digest(self, target: Any) -> str:
        return self.base().kernel_digest(target)


AUDIT_NO_KV_RECEIPTS = Ablation("bandwidth-audit-no-kv-receipts", encoded=True,
                                env={"VERITY_VLLM_BENCH_KV_RECEIPTS": "off"})
AUDIT_K1 = Ablation("bandwidth-audit-k1", encoded=True, env={"VERITY_VLLM_BENCH_CODEC_ROUNDS": "1"})
AUDIT_NO_FILLER = Ablation("bandwidth-audit-no-filler", encoded=True, filler=False)
PLAIN_NO_FILLER = Ablation("bandwidth-plain-no-filler", encoded=False, filler=False)

__all__ = ["AUDIT_K1", "AUDIT_NO_FILLER", "AUDIT_NO_KV_RECEIPTS", "PLAIN_NO_FILLER", "Ablation"]
