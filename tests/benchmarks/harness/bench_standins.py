"""The stand-ins the bench tests share: a fake engine and its factory, and a spec's preset, point and arms."""
from __future__ import annotations

from verity_bench.vllm.arms import (
    Arm,
)
from verity_bench.vllm.presets import (
    SweepPoint,
    load_spec,
)
from verity_bench.vllm.runner import (
    PER_PASS_TOKENS,
    ResidentCapacityError,
    _token_digest,
)


class FakeEngine:
    def __init__(self, arm, events, *, gate_ok=True, bad_tokens=False, kv_tokens=458_752, timed_digest=None,
                 drift=0.0, timed_ok=True, per_pass_tokens=False):
        self.arm, self.events, self.gate_ok, self.bad_tokens = arm, events, gate_ok, bad_tokens
        self.timed_digest, self.drift, self.timed_ok, self.per_pass_tokens = timed_digest, drift, timed_ok, per_pass_tokens
        self.identity = {
            "vllm_version": "0.fake+gd9105ea80",
            "kernel_digest": f"digest-{arm.id}",
            "engine_overrides": {"quantization": "fp8"} if arm.id == "stock-fp8" else {},
            **({"kv_cache_tokens": kv_tokens} if kv_tokens is not None else {}),
        }

    def gate(self, point):
        self.events.append(("gate", self.arm.id, point.id))
        token = 8 if self.bad_tokens else 7
        return {f"{point.id}-0": (token, 9)}

    def custom_gate(self):
        self.events.append(("custom-gate", self.arm.id))
        timed = any(e[:2] == ("measure", self.arm.id) for e in self.events)
        return {"passed": self.gate_ok and (self.timed_ok or not timed), "provided": True,
                **({PER_PASS_TOKENS: True} if self.per_pass_tokens else {})}

    def warmup(self, point):
        self.events.append(("warmup", self.arm.id, point.id))
        return {"wall_s": 0.5}

    def measure(self, point):
        rep = sum(e[:2] == ("measure", self.arm.id) for e in self.events)
        # GPU-wide drift: each timed run, of any arm, is `drift` slower than the one before it.
        slower = 1 + self.drift * sum(e[0] == "measure" for e in self.events)
        self.events.append(("measure", self.arm.id, point.id))
        scale = 1.0 if self.arm.id == "stock-fp8" else 2.0
        return {
            "token_digest": self.timed_digest or _token_digest({f"{point.id}-0": (8 if self.bad_tokens else 7, 9)}),
            "metrics": {
                "time_per_output_token_s": scale * slower if self.drift else scale * (rep + 1),
                "time_to_first_token_s": scale * 2,
                "tokens_per_s": 100 / scale,
                "peak_gpu_memory_bytes": 1000 * scale,
            },
            "spans": {"hashing": scale},
            "extras": {"share": scale / 10},
        }

    def close(self):
        self.events.append(("close", self.arm.id))


class FakeFactory:
    def __init__(self, *, resident_failure=None, gate_ok=True, bad_tokens=None, kv_tokens=None, timed_digest=None,
                 drift=0.0, kv_after_gate=None, timed_ok=True, per_pass_tokens=()):
        self.events = []
        self.per_pass_tokens = set(per_pass_tokens)
        self.resident_failure = resident_failure
        self.failed = False
        self.gate_ok = gate_ok
        self.bad_tokens = bad_tokens
        self.kv_tokens = kv_tokens or {}
        self.timed_digest = timed_digest or {}
        self.drift = drift
        self.kv_after_gate = kv_after_gate or {}
        self.timed_ok = timed_ok

    def create(self, arm, preset, *, resident):
        self.events.append(("create", arm.id, resident))
        if resident and arm.id == self.resident_failure and not self.failed:
            self.failed = True
            raise ResidentCapacityError(f"{arm.id} does not fit")
        gated = any(e[:2] == ("gate", arm.id) for e in self.events)
        kv = self.kv_after_gate.get(arm.id) if gated and arm.id in self.kv_after_gate else self.kv_tokens.get(arm.id, 458_752)
        return FakeEngine(arm, self.events, gate_ok=self.gate_ok,
                          bad_tokens=arm.id == self.bad_tokens,
                          kv_tokens=kv, timed_digest=self.timed_digest.get(arm.id), drift=self.drift,
                          timed_ok=self.timed_ok, per_pass_tokens=arm.id in self.per_pass_tokens)


def inputs():
    preset = load_spec("llama31-8b-fp8-1gpu").preset
    point = SweepPoint(32, 512, "mixed", 64)
    arms = (Arm("stock-fp8", precision="fp8"), Arm("candidate", control="stock-fp8"))
    return preset, (point,), arms
