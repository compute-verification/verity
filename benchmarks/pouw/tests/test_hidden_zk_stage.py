"""hidden_zk's fast staging (stage.py under HZ_STAGE=fast: fast_stage.py's patches of C-Flock's staging) writes the bytes the
library writes: the circuit, its instances, public file and pin, and the same stage.json but for its timings and its label."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

STAGE = Path(__file__).resolve().parents[1] / "hidden_zk" / "stage.py"
OWN = ("seconds", "maxrss_gb", "staging")


@pytest.mark.slow
def test_fast_staging_writes_the_library_bytes(tmp_path):
    """About 20 s: the (1,1) tile at K=64 staged both ways, each in its own process, since fast_stage patches the library in
    place. Nothing cheaper catches a library change under one of those patches that makes fast staging write other bytes."""
    runs = {mode: subprocess.Popen([sys.executable, str(STAGE), "tile", "64", "2", str(tmp_path / mode)], text=True,
                                   env={**os.environ, "HZ_STAGE": mode}, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            for mode in ("fast", "library")}
    for mode, p in runs.items():
        _, err = p.communicate()
        assert p.returncode == 0, f"{mode}: {err[-2000:]}"
    fast, lib = tmp_path / "fast", tmp_path / "library"
    files = sorted(f.name for f in lib.iterdir())
    assert sorted(f.name for f in fast.iterdir()) == files
    for name in files:
        if name != "stage.json":
            assert (fast / name).read_bytes() == (lib / name).read_bytes(), name
    fs, ls = (json.loads((d / "stage.json").read_text()) for d in (fast, lib))
    assert (fs["staging"], ls["staging"]) == ("fast", "library")
    assert {k: v for k, v in fs.items() if k not in OWN} == {k: v for k, v in ls.items() if k not in OWN}


#: a `UnitStager`'s circuit under ``argv[1]`` (``fast``: `fast_stage` installed first) and layout ``argv[3]`` (empty: flat;
#: ``+structural`` after the mode: struct_lower's template), written to ``argv[2]``/circuit.txt, and its pin and class
STAGER = """
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[4])
mode, out = sys.argv[1], Path(sys.argv[2])
layout, _, structural = sys.argv[3].partition("+")
layout = layout or None
out.mkdir(parents=True)
if mode == "fast":
    import fast_stage
    fast_stage.install(out)
from verity.ml.circuits.boolean import BIT
from verity.ml.circuits.boolean.forms import add
from verity.ml.circuits.boolean.trace import definition
from verity.ml.circuits.defs import bind
from verity.ml.circuits.types import Array
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2
from verity_flock import class_statement as CS
d = bind(definition("FastStageToy", 1, lambda C, a, b, w: add(C, a, b, width=20, carry_out=True) + [C.AND(w[0], w[2047]) ^ w[1000]],
                    (("a", Array(12, BIT)), ("b", Array(20, BIT)), ("w", Array(2048, BIT))), Array(22, BIT), register=False))
st = CS.UnitStager(d, "unit/fast-stage-toy", [("ab", V2, 32, ["a", "b"]), ("w", V2, 2048, ["w"])], [(V2, (0,))],
                   circuit=out / "circuit.txt", layout=layout, structural=bool(structural))
print(json.dumps({"pin": st.pin, "class": st.class_sha512}))
"""


@pytest.mark.slow
@pytest.mark.parametrize("layout", ["", "iv/hybrid/4", "iv/hybrid/4+structural"], ids=["flat", "templated", "structural"])
def test_fast_staging_composes_the_library_circuit_flat_and_templated(layout, tmp_path):
    """About 20 s: a Boolean unit's `UnitStager` circuit, flat, templated from the flat unit (`circuit.compose`'s
    ``layout``) and templated from struct_lower (``structural``, the H1 stagers' form under ``H1_TMPL=structural``), built under
    fast_stage's patches and without them, each in its own process: the same file, pin and class. The flat file is the
    one fast_stage streams from its scratch netlist; a templated one has no unit section to stream, and a structural
    one's unit_sha256 is over its template, never a netlist."""
    runs = {mode: subprocess.Popen([sys.executable, "-c", STAGER, mode, str(tmp_path / mode), layout, str(STAGE.parent)],
                                   text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            for mode in ("fast", "library")}
    got = {}
    for mode, p in runs.items():
        out, err = p.communicate()
        assert p.returncode == 0, f"{mode}: {err[-2000:]}"
        got[mode] = json.loads(out.strip().splitlines()[-1])
    fast, lib = ((tmp_path / m / "circuit.txt").read_bytes() for m in ("fast", "library"))
    assert fast == lib and got["fast"] == got["library"]
    assert fast.startswith(b"flock-circuit/tmpl " if layout else b"flock-circuit ")


#: a flat unit of many calls of two Boolean Definitions (`stamp`'s templates): a chain, a call whose operands share bits, a
#: dead call, constant operand bits and a call on a later row's bits; its pin, class, layout counts and the unit's outputs
#: at one and at three random lanes, under ``argv[1]`` as in `STAGER`
CHAIN = """
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[3])
mode, out = sys.argv[1], Path(sys.argv[2])
out.mkdir(parents=True)
if mode == "fast":
    import fast_stage
    fast_stage.install(out)
import numpy as np
from verity.ml.circuits.boolean import BIT
from verity.ml.circuits.boolean.forms import ZERO
from verity.ml.circuits.boolean.trace import definition
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.types import Array
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2
from verity_flock import class_statement as CS
W, N = 16, 9
WORD = Array(W, BIT)

def mix(C, s, x):
    t = [s[i] ^ s[(i + 1) % W] ^ x[i] for i in range(W)]
    return C.commit([t[i] ^ C.AND(t[(i + 1) % W], t[(i + 3) % W]) for i in range(W)])

MIX = definition("StampChainMix", 1, mix, (("s", WORD), ("x", WORD)), WORD, register=False)
HALF = definition("StampChainHalf", 1, lambda C, s, x: [C.AND(s[i], x[i]) if i < 9 else ZERO for i in range(W)],
                  (("s", WORD), ("x", WORD)), WORD, register=False)

def body(B, S, s, x, p):
    for i in range(N):
        s = B.call(MIX, s, x[i % 3])
    B.call(MIX, x[0], s)
    u = B.call(MIX, s, s)
    v = B.call(HALF, x[1], u)
    w = B.call(MIX, v, x[2])
    w = B.call(HALF, w, p[3])
    return B.call(MIX, w, v)

d = bind(CompositeDefinition("StampChain", 1, (), lambda S: ((("s", WORD), ("x", Array(3, WORD)), ("p", Array(128, WORD))), WORD),
                             body, register=False))
st = CS.UnitStager(d, "unit/stamp-chain", [("s", V2, 16, ["s"]), ("x", V2, 48, ["x"]), ("p", V2, 2048, ["p"])], [(V2, (0,))],
                   circuit=out / "circuit.txt")
x = np.random.default_rng(7).integers(0, 1 << 16, size=(3, 132), dtype=np.uint64)
evals = [st.low.evaluate(x[:n])[0].tolist() for n in (1, 3)]
print(json.dumps({"pin": st.pin, "class": st.class_sha512, "counts": st.low.layout.counts, "evals": evals}))
"""


@pytest.mark.slow
def test_stamped_unit_writes_the_library_bytes(tmp_path):
    """About 15 s: a flat unit of many calls (`CHAIN`) built with fast_stage's stamp and without it, each in its own process:
    the same file, pin, class, counts and values. The single-Definition units above are one call each, so only this catches
    a stamp that renames a call's ANDs, keeps a dead one, or keys a template on less than its operands' sharing."""
    runs = {mode: subprocess.Popen([sys.executable, "-c", CHAIN, mode, str(tmp_path / mode), str(STAGE.parent)],
                                   text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            for mode in ("fast", "library")}
    got, err = {}, {}
    for mode, p in runs.items():
        out, err[mode] = p.communicate()
        assert p.returncode == 0, f"{mode}: {err[mode][-2000:]}"
        got[mode] = json.loads(out.strip().splitlines()[-1])
    assert "STAMP unit/stamp-chain" in err["fast"]
    fast, lib = ((tmp_path / m / "circuit.txt").read_bytes() for m in ("fast", "library"))
    assert fast == lib and got["fast"] == got["library"]
