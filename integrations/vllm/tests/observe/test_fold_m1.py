"""Fold of the M1 capture (`data/logs/m1.jsonl.gz`) under its profile.  Skipped unless both exist,
the memory model is available and `CAPTURE_BIG=1` (full-size builds are pod jobs).  Asserts the expected
instance counts, the return type and the absence of Unsupported / Unresolved; prints the Program size and build
time (run with `-s` to see them)."""
from __future__ import annotations

import importlib.util
import os
from collections import Counter
from pathlib import Path

import pytest

from verity.ml.definitions.registry.prims import I32
from verity.ml.circuits.types import Array

REPO = Path(__file__).resolve().parents[2]
LOG = REPO / "data" / "logs" / "m1.jsonl.gz"
PROFILE = "vllm_d9105ea80_sm89_eager"


def _available() -> str | None:
    if os.environ.get("CAPTURE_BIG") != "1":
        return "full-size build: set CAPTURE_BIG=1"
    if not LOG.exists():
        return f"{LOG} missing"
    if importlib.util.find_spec(f"verity_vllm.observe.fold.profiles.{PROFILE}") is None:
        return f"profile {PROFILE} missing"
    try:
        from verity_vllm.observe.fold.memory import MemoryModel
        MemoryModel()
    except NotImplementedError:
        return "memory.py is a skeleton"
    return None


pytestmark = pytest.mark.skipif(_available() is not None, reason=str(_available()))


@pytest.fixture(scope="module")
def m1():
    from verity_vllm.check.fold_compare import fold_log, load_profile
    r = fold_log(LOG, load_profile(PROFILE), strict=None, with_program=True)
    print(f"\nM1 fold: {len(r.body.nodes)} instances, program gates={r.program.gates if r.program else None}, "
          f"timings={r.timings}, digest={r.digest}")
    return r


def _bases(r) -> Counter:
    return Counter(m["spec"].split("{")[0].split("_v")[0] for m in r.instances)


def test_no_unsupported_no_unresolved(m1):
    classes = Counter(e["class"] for e in m1.errors)
    assert m1.report.unsupported == 0, (m1.report.unsupported_ops, m1.report.unsupported_examples[:5])
    assert classes.get("Unresolved", 0) == 0, [e for e in m1.errors if e["class"] == "Unresolved"][:5]
    assert not m1.errors, classes


def test_instance_counts(m1):
    req = m1.requests[0]
    LP = len(req["prompt_token_ids"])
    steps = [s for s in m1.steps]
    n_decode = len(steps) - 1
    NL = next(t.n for k, t in m1.weights_type.fields if k == "layers")
    got = _bases(m1)
    positions = LP + n_decode
    expect = {
        "Gemm": 4 * NL * positions + (1 + n_decode),          # lm_head over the gathered logits rows only
        "RMSNormTriton": positions,                            # layer 0 input norm
        "RMSNormFusedCuda": (2 * NL - 1) * positions + positions,  # fused norms + final norm over every row (vLLM)
        "RoPE": 2 * NL * positions,
        "Attention": NL * positions,
        "SiluMul": NL * positions,
        "Embedding": positions,
        "TokenSelect": 1 + n_decode,
    }
    assert dict(got) == expect, {k: (got.get(k), expect.get(k)) for k in sorted(set(got) | set(expect)) if got.get(k) != expect.get(k)}
    # DESIGN §4: per decode step 121 Gemm + 1 RMSNormTriton + 60 RMSNormFusedCuda + 60 RoPE + 30 Attention + 30 SiluMul + 1 Embedding + 1 TokenSelect
    if NL == 30:
        dec = Counter(m["spec"].split("{")[0].split("_v")[0] for m in m1.instances if m["step"] == 1)
        assert dict(dec) == {"Gemm": 121, "RMSNormTriton": 1, "RMSNormFusedCuda": 60, "RoPE": 60, "Attention": 30,
                             "SiluMul": 30, "Embedding": 1, "TokenSelect": 1}, dec


def test_return_type_and_tokens(m1):
    params, ret = m1.spec.signature()
    n_tokens = len(m1.sampled[m1.requests[0]["request_id"]])
    assert ret == Array(n_tokens, I32)
    assert n_tokens == 16
    assert [n for n, _ in params] == ["prompt", "weights"]
    assert params[0][1] == Array(len(m1.requests[0]["prompt_token_ids"]), I32)


def test_program_and_digest(m1):
    assert m1.program is not None and m1.digest
    assert m1.program.gates > 0
    print(f"\nProgram: {m1.program.gates} gates, {len(m1.body.nodes)} root instances, Program() {m1.timings.get('program_s')}s, "
          f"digest {m1.timings.get('digest_s')}s, fold {m1.timings.get('fold_s')}s")
