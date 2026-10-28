"""hidden_zk's fast staging (stage.py under HZ_STAGE=fast: fast_stage.py's patches of C-Flock's staging) writes the bytes the
library writes: the circuit, its instances, public file and pin, and the same stage.json but for its timings and its label."""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
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


def _random_unit(C, seed: int):
    """A random unit of inputs, ANDs, commits, assertions and outputs with constant bits (``C``'s, either class), and its
    `ir_lower.layout` arguments: port names and splits."""
    from verity.ml.circuits.boolean import forms

    rnd = random.Random(seed)
    ins = [C.input(f"in{j}", rnd.choice([16, 32, 48, 64])) for j in range(rnd.randint(1, 3))]
    pool = [b for p in ins for b in p]

    def lin():
        x = forms.ZERO
        for _ in range(rnd.randint(1, 5)):
            x = x ^ rnd.choice(pool)
        return x.inv() if rnd.random() < 0.3 else x

    for _ in range(rnd.randint(0, 600)):
        pool.append(C.commit([lin()])[0] if rnd.random() < 0.1 else C.AND(lin(), lin()))
    for i in range(rnd.randint(0, 3)):
        if not (f := lin()).is_const():
            C.assert_zero(f"a{i}", f)
    for j in range(rnd.randint(1, 3)):
        C.output(f"out{j}", [lin() for _ in range(rnd.randint(0, 40))] + [forms.ZERO] * rnd.randint(0, 5)
                 + ([forms.ONE] if rnd.random() < 0.3 else []))
    names, outs = [f"in{j}" for j in range(len(ins))], [f"out{j}" for j in range(len(C.outputs))]
    return names, outs, [1] if len(names) > 1 and rnd.random() < 0.5 else None, [1] if len(outs) > 1 and rnd.random() < 0.5 else None


def test_the_stream_circuits_memo_is_the_librarys(tmp_path, monkeypatch):
    """About 1 s: `fast_stage.StreamCircuit`'s CSE memo, which holds hashes and reads a hit's sides back from the spool,
    against `forms.Circuit`'s on random ANDs of a few forms (pairs repeated in either order, constants, a form and its
    complement), its spool cut into blocks of 37 ANDs, and half the seeds with every hash in 3 buckets: the same bit for
    each AND."""
    monkeypatch.syspath_prepend(str(STAGE.parent))
    import fast_stage as FS
    from verity.ml.circuits.boolean import forms

    monkeypatch.setattr(FS, "BLOCK", 37)
    monkeypatch.setattr(FS, "_scratch", [tmp_path])
    for seed in range(40):
        monkeypatch.setattr(FS, "_key_hash", (lambda k: hash(k) % 3) if seed % 2 else hash)
        got = []
        for C in (forms.Circuit(cse=True), FS.StreamCircuit(cse=True)):
            rnd = random.Random(seed)
            pool = [*C.input("x", 6), forms.ZERO, forms.ONE]
            ands = []
            for _ in range(500):
                a, b = rnd.choice(pool), rnd.choice(pool)
                w = C.AND(a, b.inv() if rnd.random() < 0.2 else b)
                ands.append(w.key())
                if rnd.random() < 0.15 and len(pool) < 40:
                    pool.append(w ^ rnd.choice(pool))
            got.append((ands, getattr(C, "n", None) or len(C.kind)))
        assert got[0] == got[1], seed
        assert len(set(k for k in got[0][0] if len(k[0]) == 1)) > 100, seed
    FS._drop_spools()
    assert not list(tmp_path.iterdir())


def test_the_stream_layout_is_the_library_layout(tmp_path, monkeypatch):
    """About 3 s: `fast_stage.StreamLayout` of 150 random units, its spool cut into blocks of 37 ANDs and its netlist into
    pieces of 11 rows, against `ir_lower.layout`'s: the netlist, counts, port groups, rows and evaluations (all-zero, which
    `StreamLayout.zero` answers, and random). The H1 units stream too, but have no dead AND, assertion, constant output or
    split port group, and fill few spool blocks."""
    monkeypatch.syspath_prepend(str(STAGE.parent))
    import fast_stage as FS
    from verity.ml.circuits.boolean import forms
    from verity_flock import ir_lower as IL

    monkeypatch.setattr(FS, "BLOCK", 37)
    monkeypatch.setattr(FS, "ROWS", 11)
    monkeypatch.setattr(FS, "_scratch", [tmp_path])
    for seed in range(150):
        C = forms.Circuit()
        args = _random_unit(C, seed)
        want = IL.layout(C, *args)
        S = FS.StreamCircuit()
        assert _random_unit(S, seed) == args
        got = FS.StreamLayout(S, *args)
        assert "".join(FS._netlist(got, "u/t")) == IL.netlist(want, "u/t"), seed
        assert (got.counts, got.useful, got.const) == (want.counts, want.useful, want.const), seed
        assert (got.in_groups, got.out_groups) == (want.in_groups, want.out_groups), seed
        assert (got.rows_a, got.rows_b) == (want.rows_a, want.rows_b), seed
        rnd = random.Random(seed)
        for T, ports in ((1, [[0] for _ in want.in_bits]),
                         (5, [[rnd.getrandbits(b) if rnd.random() < 0.9 else 0 for _ in range(5)] for b in want.in_bits])):
            (wo, wok), (go, gok) = IL.evaluate_layout(want, ports, T), FS.evaluate_layout(got, ports, T)
            assert wok == gok and len(wo) == len(go) and all(np.array_equal(x, y) for x, y in zip(wo, go)), (seed, T)
        FS._drop_spools()
    assert not list(tmp_path.iterdir())
