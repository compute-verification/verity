"""`flock-rows`, the verifier's `Flock.Derive.derive` compiled, writes each unit's rows exactly as `ir_lower.netlist` does, on
every pinned template unit (each template's `PINS`). So the Python layout is a mirror of the derivation the verifier runs
(Project doc `verified-lowering-design.md` §1.2): a mismatch can only make an honest proof fail."""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("verity.ml.definitions.registry.prims")
from verity.ml.circuits import library  # noqa: E402
from verity_flock import circuit_types as CT  # noqa: E402
from verity_numerical.bench import lowerings, templates as TM  # noqa: E402

LEAN = Path(__file__).resolve().parents[3] / "core" / "service" / "flock"
REG = lowerings.registry("C-Flock")
#: The parameters each pinned template's `PINS` keys name, in their order.
PIN_PARAMS = {"rope-head": ("D",), "silu-mul": ("I",), "silu-mul-v2": ("I",), "rmsnorm-fused-cuda": ("N", "EPS"),
              "rmsnorm-triton": ("N", "EPS"), "rmsnorm-triton-v5": ("N", "EPS"), "rmsnorm-fused-cuda-vec8": ("N", "EPS", "BLOCK"),
              "rmsnorm-cuda-vec8": ("N", "EPS", "BLOCK"), "ingress-prompt": ("MODEL", "LP", "V"), "token-select-chunk": ("C",),
              "token-select-combine": ("V", "C")}
#: Pinned units of about 3.5M rows (GLM-5.3's width-8 RMSNorms), and RMSNormTriton_v5's, which is rmsnorm-triton's netlist text
#: (test_ir_lowering), derived only with FLOCK_IR_SLOW=1, as test_ir_lowering builds them
SLOW_UNITS = {"rmsnorm-fused-cuda-vec8", "rmsnorm-cuda-vec8", "rmsnorm-triton-v5"}
SLOW = pytest.mark.skipif(not os.environ.get("FLOCK_IR_SLOW"), reason="derives a 3.5M-row unit: set FLOCK_IR_SLOW=1")
#: Every pinned unit with its pin, and one small unpinned unit.
CASES = [(t, dict(zip(names, key if isinstance(key, tuple) else (key,))), pin)
         for t, names in PIN_PARAMS.items() for key, pin in sorted(REG.module(t).PINS.items())]
CASES += [("rmsnorm-fused-cuda", {"N": 64, "EPS": 1e-5}, None)]
#: Each template's smallest case, by its parameters, runs in the quick tier; its larger pins (RMSNorm at 2048 and 4096
#: elements take 25 to 90 s each) derive the same layout code at more elements, so they are `slow`.
SMALLEST = {t: min((tuple(p.values()) for u, p, _ in CASES if u == t),
                   key=lambda v: [(0, x) if isinstance(x, (int, float)) else (1, str(x)) for x in v]) for t, _, _ in CASES}
CASES = [pytest.param(t, p, pin, id=f"{t}-{'-'.join(map(str, p.values()))}",
                      marks=[SLOW] if t in SLOW_UNITS else [] if tuple(p.values()) == SMALLEST[t] else [pytest.mark.slow])
         for t, p, pin in CASES]
needs_lake = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")


def unit_type(low) -> CT.CircuitType:
    """The unit circuit as a flat circuit type, its ports ``in0 ..`` and ``out0 ..`` one word each."""
    U, C = low.units, low.circuit
    ins = [b for j in range(len(U.in_widths)) for b in C.inputs[f"in{j}"]]
    outs = [f for j in range(len(U.out_widths)) for f in C.outputs[f"out{j}"]]
    return CT.from_gf2(C, ins, outs, [(1, w) for w in U.in_widths], [(1, w) for w in U.out_widths], library.PROGRAM, {})


def split(groups) -> str:
    """The port indices where the layout's second and later groups start."""
    return ",".join(map(str, list(itertools.accumulate(len(bits) for _, _, bits in groups))[:-1]))


def flock_rows(low, tmp: Path, in_split: str | None = None) -> subprocess.CompletedProcess:
    t = unit_type(low)
    (tmp / "types.json").write_text(json.dumps({"types": [t.content()]}))
    (tmp / "leaves.json").write_text(json.dumps(low.leaves, sort_keys=True, separators=(",", ":")))
    args = [str(LEAN / ".lake/build/bin/flock-rows"), str(tmp / "types.json"), "--root", t.digest(), "--name", low.name,
            "--leaves", str(tmp / "leaves.json"), "--in-split", in_split or split(low.layout.in_groups),
            "--out-split", split(low.layout.out_groups)]
    if low.cut is not None:
        (tmp / "cut.json").write_text(json.dumps(low.cut, sort_keys=True, separators=(",", ":")))
        args += ["--cut", str(tmp / "cut.json")]
    return subprocess.run(args, capture_output=True, text=True)


@pytest.fixture(scope="module")
def built():
    subprocess.run(["lake", "build", "flock-rows"], cwd=LEAN, check=True, capture_output=True)


def test_every_pinned_template_is_a_case():
    assert {t for t, mod in REG.modules().items() if hasattr(mod, "PINS")} == set(PIN_PARAMS)


@needs_lake
@pytest.mark.parametrize("template,params,pin", CASES)
def test_flock_rows_is_the_layout(built, template, params, pin, tmp_path):
    low = REG.module(template).lowering(TM.subcircuit(template, **params))
    out = flock_rows(low, tmp_path)
    assert out.returncode == 0, out.stderr
    assert out.stdout == low.text
    digest = hashlib.sha256(out.stdout.encode()).hexdigest()
    assert digest == low.sha256
    if pin is not None:
        assert digest == pin


@needs_lake
def test_flock_rows_is_the_grouped_circuit_unit(built, tmp_path):
    """`silu-mul-v2`'s `verity/flock-circuit` unit at I = 1536, three elements (`CIRCUIT_PINS`)."""
    mod = REG.module("silu-mul-v2")
    low = mod.circuit_lowering(TM.subcircuit("silu-mul-v2", I=1536))
    out = flock_rows(low, tmp_path)
    assert out.returncode == 0, out.stderr
    assert out.stdout == low.text and hashlib.sha256(out.stdout.encode()).hexdigest() == mod.CIRCUIT_PINS[1536]


@needs_lake
def test_flock_rows_refuses_a_split_out_of_order(built, tmp_path):
    low = REG.module("rope-head").lowering(TM.subcircuit("rope-head", D=64))
    out = flock_rows(low, tmp_path, in_split="3,1")
    assert out.returncode == 1 and "split" in out.stderr
