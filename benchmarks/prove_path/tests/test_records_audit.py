"""The audit from a window's records alone (`records_audit`), without proving, on a window written through the interface
(`verity.ml.windows.interface`) under core's registration:

* the verifier's window is the circuit record's, each template resolved in Verity's catalog, and the record must be the one
  the registration binds;
* the compile check runs the registered compiler as the verifier's table names it, on advice that opens the registered root,
  and refuses by name a compiler it doesn't hold, an opening of another root, a compiler that refuses the advice, one whose
  circuit is another and a length other than the registered one;
* `source` and the negatives refuse what the records don't bind, and `fixed_before_draw` refuses advice re-committed after
  the coins;
* none of it imports `verity_vllm`, and the served compiler's command (``prove_path.py circuit``) refuses advice it can't
  compile and charges the length the registration does."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from verity.core.protocols.sampled_proofs.one_stage import draw as D, registration as R
from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate_batch
from verity.ml.circuits.ops import gemm as G
from verity.ml.windows import interface as IF

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "one_stage"))
import records_audit as RA  # noqa: E402

K = 64
COORD = bind(G.ScaledMmS8Coordinate, K=K, **G.GEMM_SM120_S8)
TID = descriptor_id(COORD)
LAW = "stratified:2"
RUN = "ab" * 64
COMPILER = {"name": "test/compiler", "digest": "cd" * 64}
ADVICE = bytes(range(40))

FAKE = '''import argparse, json, sys
ap = argparse.ArgumentParser()
for f in ("--record", "--expect", "--advice", "--out"):
    ap.add_argument(f)
ap.add_argument("--bits", type=int)
a = ap.parse_args()
data = open(a.advice, "rb").read()
if data.hex() != a.expect:
    sys.exit("advice this compiler can't compile")
open(a.out, "w").write(open(a.record).read())
print(json.dumps({"bits": 8 * len(data) if a.bits is None else a.bits}))
'''


def _write(d: Path, n: int = 5, cap: int = 4, advice: bool = True) -> SimpleNamespace:
    """A window of ``n`` coordinates in members of ``cap``, its advice committed under `COMPILER` (or none), written and
    registered through the interface under core's registration."""
    reg = IF.commit_advice(COMPILER, ADVICE)
    bound = IF.advice_binding(COMPILER, reg.root.hex(), 8 * len(ADVICE)) if advice else None
    win = IF.Window(IF.Circuit("test", ((COORD, n),)), RUN, D.parse_law(LAW), cap, advice=bound)
    rng = np.random.default_rng(0)
    x, w = (rng.integers(-128, 128, (n, K)).astype(np.int8) for _ in range(2))
    sx, sw = ((np.abs(rng.standard_normal(n)) / 127).astype(np.float32) for _ in range(2))
    ins = [x.view(np.uint8), sx.view(np.uint32), w.view(np.uint8), sw.view(np.uint32)]
    win.record(COORD, ins, [np.asarray(evaluate_batch(COORD, *ins, kernel="reference"), dtype=np.uint16)], {"unit": 0})
    win.write(win.close(salt_key=bytes(32)), d, {"label": "test"}, registrar=R)
    doc = json.loads((d / "circuit.json").read_text())
    return SimpleNamespace(dir=d, doc=doc, rec=json.loads((d / "registration.json").read_text()),
                           opening=IF.advice_opening(ADVICE, reg), bound=bound)


def _compilers(tmp: Path, doc: dict, **kw) -> dict:
    """The verifier's table: `COMPILER` as the fake command, which writes ``doc`` less its run fields for `ADVICE` alone."""
    body = json.dumps({**doc, "run": "", "context": ""}).encode()
    record = tmp / f"record-{hashlib.sha512(body).hexdigest()[:16]}.json"
    (tmp / "compiler.py").write_text(FAKE)
    record.write_bytes(body)
    argv = [sys.executable, str(tmp / "compiler.py"), "--record", str(record), "--expect", ADVICE.hex(),
            *(x for k, v in kw.items() for x in (f"--{k}", str(v)))]
    return {COMPILER["name"]: {"digest": COMPILER["digest"], "argv": argv}}


@pytest.fixture
def written(tmp_path):
    return _write(tmp_path / "w")


def test_the_verifiers_window_is_the_circuit_record_under_verity_s_catalog(written):
    mine, doc, sets = RA.verifier_window(written.dir, D.parse_law(LAW), None)
    assert doc == written.doc and mine.run == RUN and mine.partition_digest == written.rec["partition"]
    assert [(m.template, m.n) for m in mine.members] == [(TID, 4), (TID, 1)]
    assert set(sets) == {TID} and callable(sets[TID][0]) and descriptor_id(sets[TID][1]) == TID
    w = written.rec["window"]
    assert w["circuit"] == hashlib.sha512((written.dir / "circuit.json").read_bytes()).hexdigest()
    assert (w["compiler"], w["advice"]) == (written.bound["compiler"], written.bound["advice"])
    (written.dir / "circuit.json").write_text(json.dumps({**doc, "name": "other"}))
    with pytest.raises(ValueError, match="not the circuit record the registration binds"):
        RA.verifier_window(written.dir, D.parse_law(LAW), None)


def test_the_compile_check_runs_the_registered_compiler_on_the_opened_advice(written, tmp_path):
    w, doc, op = written.rec["window"], written.doc, written.opening
    v, bound = RA.compile_check(doc, w, op, _compilers(tmp_path, doc), tmp_path / "c")
    assert v["ok"] and bound == written.bound and v["detail"]["bits"] == 8 * len(ADVICE)
    assert IF.circuit_of(json.loads((tmp_path / "c" / "compiled.json").read_text())) == IF.circuit_of(doc)
    other = dict(op, salts=[("00" if op["salts"][0][:2] != "00" else "01") + op["salts"][0][2:], *op["salts"][1:]])
    short = {**doc, "members": [{**doc["members"][0], "n": 3}, *doc["members"][1:]]}
    for name, (compilers, window, opening), codes in (
            ("a compiler the verifier doesn't hold", ({}, w, op), ["compiler", "compiles", "circuit", "length"]),
            ("another digest", (_compilers(tmp_path, doc), {**w, "compiler": {**COMPILER, "digest": "00" * 64}}, op),
             ["compiler", "advice", "compiles", "circuit", "length"]),
            ("an opening of another root", (_compilers(tmp_path, doc), w, other), ["advice", "compiles", "circuit", "length"]),
            ("a compiler refusing the advice", (_compilers(tmp_path, doc, expect="00"), w, op), ["compiles", "circuit", "length"]),
            ("another circuit", (_compilers(tmp_path, short), w, op), ["circuit"]),
            ("another length", (_compilers(tmp_path, doc, bits=8 * len(ADVICE) + 1), w, op), ["length"])):
        v, bound = RA.compile_check(doc, window, opening, compilers, tmp_path / "c")
        assert (v["codes"], bound) == (codes, None), name


def test_source_and_the_negatives_refuse_what_the_records_dont_bind(written, tmp_path):
    w, doc = written.rec["window"], written.doc
    compilers = _compilers(tmp_path, doc)
    v, bound = RA.compile_check(doc, w, written.opening, compilers, tmp_path / "c")
    held = {"run": RUN, "circuit": w["circuit"], "compiled": v["ok"], **bound, "compilers": compilers,
            "opening": written.opening, "output": json.loads((tmp_path / "c" / "compiled.json").read_text())}
    assert RA.source(w, held)["ok"]
    assert RA.source(w, {**held, "run": "00" * 64})["codes"] == ["run"]
    assert RA.source(w, {**held, "compiled": False})["codes"] == ["compiled"]
    got = RA.negatives(w, held, doc, tmp_path / "n")
    assert set(got) == {"registration-naming-another-circuit", "registration-naming-another-run",
                        "registration-naming-another-compiler", "registration-naming-another-advice-root",
                        "registration-charging-another-length", "registration-binding-no-compiler-or-advice",
                        "opening-under-another-salt", "circuit-record-not-the-compilers-output"}
    assert all(x["refused"] for x in got.values()), got
    fixed = _write(tmp_path / "fixed", advice=False)
    fw = fixed.rec["window"]
    held = {"run": RUN, "circuit": fw["circuit"], "compiled": True, "compiler": None, "advice": None}
    assert "compiler" not in fw and RA.source(fw, held)["ok"]
    got = RA.negatives(fw, held, fixed.doc, tmp_path / "n")
    assert set(got) == {"registration-naming-another-circuit", "registration-naming-another-run",
                        "registration-binding-a-compiler-for-a-fixed-circuit"} and all(x["refused"] for x in got.values())


def test_the_advice_is_fixed_before_the_draw_and_refused_re_committed_after_it(written):
    import a5 as A5

    rec, held = written.rec, {**written.bound, "opening": written.opening, "run": RUN, "circuit": written.rec["window"]["circuit"],
                              "compiled": True}
    receipt = {"registration": R.digest(rec), "verifier": A5.VERIFIER, "received": 1}
    assert RA.fixed_before_draw(receipt, rec, written.bound)["ok"]
    assert RA.fixed_before_draw({**receipt, "registration": "00" * 64}, rec, written.bound)["codes"] == ["after-the-draw"]
    assert RA.fixed_before_draw(receipt, rec, written.bound, "challenger")["codes"] == ["after-the-draw"]
    assert RA.fixed_before_draw(receipt, rec, None)["codes"] == ["compiler", "advice"]
    late = RA.after_draw(receipt, rec, held)
    assert late == {"refused": True, "codes": ["after-the-draw"], "before_draw_checks_pass": True}


def test_the_audit_from_the_records_imports_no_vllm(written, tmp_path):
    """The verifier's window, the compile check, `source`, the negatives and the draw-order check, in a fresh process: no
    `verity_vllm` module is loaded."""
    compilers = tmp_path / "compilers.json"
    compilers.write_text(json.dumps(_compilers(tmp_path, written.doc)))
    (tmp_path / "opening.json").write_text(json.dumps(written.opening))
    code = f"""
import json, sys
from pathlib import Path
sys.path.insert(0, {str(HERE.parent)!r})
import records_audit as RA
from verity.core.protocols.sampled_proofs.one_stage import draw as D, registration as R
import a5 as A5
d, t = Path({str(written.dir)!r}), Path({str(tmp_path)!r})
mine, doc, sets = RA.verifier_window(d, D.parse_law({LAW!r}), None)
rec = json.loads((d / "registration.json").read_text())
compilers, opening = json.loads((t / "compilers.json").read_text()), json.loads((t / "opening.json").read_text())
v, bound = RA.compile_check(doc, rec["window"], opening, compilers, t / "c")
held = {{"run": mine.run, "circuit": rec["window"]["circuit"], "compiled": v["ok"], **bound, "compilers": compilers,
        "opening": opening, "output": json.loads((t / "c" / "compiled.json").read_text())}}
assert v["ok"] and RA.source(rec["window"], held)["ok"]
assert all(x["refused"] for x in RA.negatives(rec["window"], held, doc, t / "n").values())
receipt = {{"registration": R.digest(rec), "verifier": A5.VERIFIER}}
assert RA.fixed_before_draw(receipt, rec, bound)["ok"] and RA.after_draw(receipt, rec, held)["refused"]
print(json.dumps(sorted(m for m in sys.modules if m.split(".")[0] == "verity_vllm")))
"""
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr[-3000:]
    assert json.loads(p.stdout.strip().splitlines()[-1]) == []


def test_the_served_compilers_command_charges_the_registered_length_and_refuses_what_it_cant_compile(tmp_path, monkeypatch, capsys):
    """``prove_path.py circuit``, the served window's registered compiler as `records_audit.compile_check` runs it, on the
    synthetic serve-openai run of `test_prove_path` (its I2 compile stubbed, the window over its Programs stubbed): it writes the
    circuit record and prints the length the registration charges, and exits nonzero on advice it can't compile."""
    import prove_path as PP
    from test_prove_path import _served
    from verity_vllm.pipeline import serving_rows as SRW

    s = _served(tmp_path, monkeypatch)
    loaded = SimpleNamespace(progs={"r0": {}}, dirs={"r0": "d"}, arrivals={"r0": 0})
    monkeypatch.setattr(PP.I2Compiler, "load", lambda self, rank, world, requests: loaded)
    seen = {}
    monkeypatch.setattr(SRW, "population", lambda progs, partition, run, dirs, **kw: (seen.update(run=run, kw=kw) or "W", []))
    monkeypatch.setattr(SRW, "circuit_record", lambda window: {"format": IF.CIRCUIT, "window": window})
    (tmp_path / "part.json").write_text(json.dumps({"templates": [], "law": "subset:1"}))
    (tmp_path / "advice.bin").write_bytes(s.data)
    base = ["circuit", "--serve-run", str(s.run), "--window", "0", "--row-dir", s.a.row_dir, "--partition",
            str(tmp_path / "part.json"), "--out", str(tmp_path / "out.json")]
    capsys.readouterr()
    assert PP.main([*base, "--advice", str(tmp_path / "advice.bin")]) == 0
    assert json.loads(capsys.readouterr().out) == {"bits": s.bound["advice"]["bits"]}
    assert json.loads((tmp_path / "out.json").read_text()) == {"format": IF.CIRCUIT, "window": "W"} and seen["run"] == ""
    (tmp_path / "bad.bin").write_bytes(b"\xff" * 7)
    assert PP.main([*base, "--advice", str(tmp_path / "bad.bin")]) == 2
    assert "prove_path.py circuit:" in capsys.readouterr().err
