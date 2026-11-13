"""Counterexamples to update hardness (T1): a family shows that some steps' updated weights rebuild from θ in few bits.

A family is a directory a run writes as it trains (`Family`, through `Harness.family`):

- `family.json`: its name, its decoder (a `codecs.Plan` spec from the shared library), `uniform` (one rule for every
  step it declares) and the declared steps;
- `<t>.enc`: E_t, everything the decoder reads beyond θ to rebuild step t's updated weights, declared imported computed
  values included (they travel in E_t and count in its bits).

The checker (`check`) samples a sample of the declared steps from a live coin (`n_sample` of them), rebuilds each
sampled step's updated weights and compares each, bit for bit, with the digest the harness logged when the step read it.
A rebuild must cost at most c MACs: the larger of the plan's declared cost and the MACs the harness's counter sees the
decode compute. Each declared step gets β_t = bits(E_t) / (updated MACs per token block). The family covers the steps
it verified, and every step it declares if it is uniform and every sampled step verified.
"""
from __future__ import annotations

import json
import os
import random
import struct
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from .annotate import annotate
from .codecs import Plan
from .codecs.wire import DTYPES
from .flops import Counter, KeyedDigest, digest

SAMPLE_SEARCH = 5
SAMPLE_FINAL = 20


class Family:
    """A family's writer: `put(step, E_t)` for each step it declares."""

    def __init__(self, root: Path, name: str, plan: Plan, uniform: bool):
        self.dir = Path(root) / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.name, self.plan, self.uniform, self.steps = name, plan, uniform, []
        self._write()

    def put(self, step: int, enc: bytes) -> None:
        (self.dir / f"{step}.enc").write_bytes(enc)
        self.steps.append(int(step))
        self._write()

    def _write(self) -> None:
        meta = {"name": self.name, "decoder": self.plan.spec(), "uniform": self.uniform, "steps": sorted(self.steps)}
        tmp = self.dir / "family.json.tmp"
        tmp.write_text(json.dumps(meta) + "\n")
        tmp.rename(self.dir / "family.json")


def weight_names(weights: dict[str, list[str]]) -> dict[str, tuple[str, str]]:
    """The names a decoder produces for a step: a weight read at one value is its key; one read at several (a MeZO step's
    perturbations) is `key@i` for its i-th value."""
    out = {}
    for key, ds in weights.items():
        for i, d in enumerate(ds):
            out[key if len(ds) == 1 else f"{key}@{i}"] = (key, d)
    return out


@dataclass
class Verified:
    family: str
    uniform: bool
    declared: list[int]
    coin: str
    sampled: list[int]
    passed: list[int] = field(default_factory=list)
    failed: dict = field(default_factory=dict)
    cost: dict = field(default_factory=dict)
    beta: dict = field(default_factory=dict)
    covers: list[int] = field(default_factory=list)

    def coverage(self) -> dict[int, float]:
        return {t: self.beta[t] for t in self.covers if t in self.beta}

    def to_json(self) -> dict:
        d = asdict(self)
        d["beta"] = {str(k): v for k, v in self.beta.items()}
        d["cost"] = {str(k): v for k, v in self.cost.items()}
        d["failed"] = {str(k): v for k, v in self.failed.items()}
        return d


def _bases(names: dict, shapes: dict, theta: dict[str, torch.Tensor], device) -> dict[str, torch.Tensor]:
    out = {}
    for name, (key, _) in names.items():
        dt, shape = shapes[key]
        dtype = DTYPES[dt]
        if key in theta:
            out[name] = theta[key].to(device=device, dtype=dtype)
        else:
            out[name] = torch.zeros(shape, dtype=dtype, device=device)
    return out


def rebuild(plan: Plan, enc: bytes, step, theta: dict, c: float, device="cpu", wdigest=digest) -> tuple[bool, str, int]:
    """Rebuild one step's updated weights from θ and E_t; (verified, reason, cost in MACs). The decoder must produce every
    weight the step read (any others it produces cost their MACs and are not compared); `wdigest` is the run's value
    digest."""
    names = weight_names(step.weights)
    if not names:
        return False, "the step has no updated weights", 0
    bases = _bases(names, step.shapes, theta, device)
    try:
        missing = sorted(set(names) - set(plan.keys(enc)))
        if missing:
            return False, f"weights: missing {missing[:4]}", 0
        counter = Counter(digests=False)
        with counter:
            out = plan.decode(enc, bases)
        cost = max(counter.setup.macs, plan.cost_macs(enc, bases))
    except (ValueError, KeyError, IndexError, TypeError, RuntimeError, struct.error) as e:
        return False, f"decode: {type(e).__name__}: {e}", 0
    if cost > c:
        return False, f"rebuild costs {cost} MACs > c = {c:.0f}", cost
    for name, (_, want) in names.items():
        if wdigest(out[name]) != want:
            return False, f"{name}: rebuilt value differs from the one the step read", cost
    return True, "", cost


def check(family_dir: Path, record: dict, theta: dict[str, torch.Tensor], c: float, n_sample: int = SAMPLE_SEARCH,
          coin: bytes | None = None, device="cpu", plan=None) -> Verified:
    """Verify a family against a run's count. `coin` is the verifier's live coin (default: fresh from the OS); `plan` is
    the family's decoder when the run holds it (a plan outside the codec library), else its spec's."""
    family_dir = Path(family_dir)
    meta = json.loads((family_dir / "family.json").read_text())
    plan = plan or Plan.from_spec(meta["decoder"])
    wdigest = KeyedDigest(bytes.fromhex(record["digest_key"])) if record.get("digest_key") else digest
    ann = annotate(record)
    by_step = {s.step: s for s in ann.steps}
    declared = sorted(t for t in set(meta["steps"]) if t in by_step and by_step[t].updated_macs)
    coin = coin if coin is not None else os.urandom(16)
    rng = random.Random(coin)
    sampled = sorted(rng.sample(declared, min(n_sample, len(declared))))
    v = Verified(meta["name"], bool(meta["uniform"]), declared, coin.hex(), sampled)
    for t in declared:
        st = by_step[t]
        enc = (family_dir / f"{t}.enc").read_bytes()
        if st.updated_per_token > 0:
            v.beta[t] = 8 * len(enc) / st.updated_per_token
    for t in sampled:
        enc = (family_dir / f"{t}.enc").read_bytes()
        ok, reason, cost = rebuild(plan, enc, by_step[t], theta, c, device, wdigest)
        v.cost[t] = cost
        if ok:
            v.passed.append(t)
        else:
            v.failed[t] = reason
    v.covers = list(declared) if v.uniform and sampled and not v.failed else list(v.passed)
    return v


def coverage(verified: list[Verified]) -> dict[int, float]:
    """Each covered step's least β_t over the families that cover it."""
    out: dict[int, float] = {}
    for v in verified:
        for t, b in v.coverage().items():
            out[t] = min(b, out.get(t, b))
    return out
