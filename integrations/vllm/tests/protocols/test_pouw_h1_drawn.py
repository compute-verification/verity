"""``benchmarks/pouw/h1_prove/drawn.py`` on a window serve-openai's H1 arm served over J10-R units at one K (the pouw
setting ``unit`` j10r, the served tests' stand-in tile, `test_pouw_h1_served.StandInJ10R`), on a gate session whose
period opened its draw K: ``stage`` rebuilds every value a drawn call's statement reads from the prover's state and the
served weights at the roots the window registered, stages the call on the circuit file `pouw_h1_circuits` built for K,
and ``statements`` and ``check`` hand core's credit statements its ``proved`` accepts (`credit._verified`, then
`credit._proved`), here with a stand-in ``flock-verify`` that accepts a session only when its proof binds every file
credit hands it (`test_pouw_credit_statements.FAKE`'s). A prover state that is not what was served, and b/<K> under
another key, are refused before anything is staged."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from verity.core.protocols.pouw import credit as CR  # noqa: E402
from verity.core.protocols.pouw.circuit import h1tile as HT  # noqa: E402
from verity.core.protocols.pouw.circuit import h1window as HW  # noqa: E402
from verity_vllm.pipeline import commit_coins as CC  # noqa: E402
from verity_vllm.pipeline import serve_h1 as SH  # noqa: E402
from verity_vllm.protocols.pouw.h1 import prover as PV  # noqa: E402
from verity_vllm.protocols.pouw.h1 import served as SV  # noqa: E402

from tests.pipeline.test_commit_coins import GATE, PEER  # noqa: E402
from tests.pipeline.test_serve_h1 import PROMPTS, _PeriodAuditor, _Server  # noqa: E402
from tests.protocols import pouw_h1_circuits as PC  # noqa: E402
from tests.protocols.test_pouw_h1_commit import K, ROWS, STEPS  # noqa: E402
from tests.protocols.test_pouw_h1_served import _unit  # noqa: E402
from verity.ml.circuits.boolean import NOT, XOR, operands  # noqa: E402
from verity.ml.circuits.defs import CompositeDefinition  # noqa: E402
from verity.ml.circuits.ports import In, Out, Ports  # noqa: E402
from verity.ml.circuits.refs import Coll, coll_concat  # noqa: E402
from verity.ml.circuits.types import Value  # noqa: E402

V1 = Value(1)
J10R_UNIT = HW.tile_unit


def _sig(S):
    return HW._chunked(S.K, J10R_UNIT(S.K, (S.R, S.C), HT.PouwH1Tile5)).signature()


def _body(B, S, a, b, zy, salt, position, roots, call, blocks):
    bit = lambda c, i: Coll(V1, c.refs.slice(i, i + 1))  # noqa: E731
    receipt, ok = (t for _n, t in _sig(S)[1].fields)
    x = B.call(XOR, operands(bit(a, 0), bit(b, 0)))
    return coll_concat(_sig(S)[1], [coll_concat(receipt, [B.call(XOR, operands(bit(a, i), bit(zy, i)))
                                                          for i in range(receipt.leaves)]),
                                    B.call(NOT, B.call(XOR, operands(x, x)))])


#: J10-R's tile unit's signature (`h1window.tile_unit` over `h1tile.PouwH1Tile5`, so its statement's rows and public row
#: are J10-R's), its receipt bits of its A rows and record and its ok 1: a circuit of a few hundred gates
StandInJ10R = CompositeDefinition("H1DrawnTestTileJ10R", 0, ("K", "R", "C"), _sig, _body, register=False,
                                  ports=Ports(params={"salt": In(kind="coin"), "position": In(kind="position")},
                                              ret={"receipt": Out(sink=True)}))

DRAWN = PC.H1_PROVE / "drawn.py"
WKEY = bytes(range(32))
FAKE = f"""#!{sys.executable}
import hashlib, json, sys
from pathlib import Path
a = sys.argv[1:]
opt = {{a[i]: a[i + 1] for i in range(len(a) - 1) if a[i].startswith("--") and a[i] != "--session"}}
sessions = a[a.index("--session") + 1:]
files = [Path(opt[k]).read_bytes() for k in ("--circuit", "--public", "--program", "--partition", "--registered")]
files[3:] = [json.dumps(json.loads(f), sort_keys=True).encode() for f in files[3:]]
bound = hashlib.sha512(b"".join(hashlib.sha512(f).digest() for f in files)).digest()
reg = json.loads(files[4])
for s in sessions:
    rec = json.loads((Path(s) / "session.json").read_text())
    ok = rec["proof"] == hashlib.sha512(bound + json.dumps(rec["unit_draw"]["units"]).encode()).hexdigest()
    print("VERDICT " + json.dumps({{"session": s, "accepted": ok, "why": "" if ok else "the proof does not verify",
                                   "program_sha512": hashlib.sha512(files[2]).hexdigest(),
                                   "registered": {{p: {{"value": e["value"], "root": e["root"]}}
                                                  for p, e in reg["ports"].items()}} if ok else {{}}}}))
"""


def _drawn():
    spec = importlib.util.spec_from_file_location("h1_prove_drawn", DRAWN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def served(tmp_path, monkeypatch):
    """One window served over the stand-in J10-R tile at K on a gate session of period p1 (draw K 2), with a prover dir
    and b/<K> under `WKEY`; its K's circuit file built by `pouw_h1_circuits` (``--strata K``); the served weights as
    drawn.py's ``--weights``."""
    j10r = _unit(defn=StandInJ10R)
    monkeypatch.setattr(SV, "unit_of", lambda name: j10r if name == "j10r" else None)
    monkeypatch.setattr(HW, "tile_unit", lambda k, tile, definition=None:
                        (j10r if definition is HT.PouwH1Tile5 else _unit())(k))
    monkeypatch.setenv("HZ_STAGE", "library")
    srv, auditor = _Server(monkeypatch), _PeriodAuditor(tmp_path, monkeypatch)
    sessions = SH.GateSessions(CC.Gate(GATE, auditor.run), PEER, STEPS, wait=auditor.settle, listens=lambda port: True)
    d = tmp_path / "serve"
    arm = SH.H1Windows(srv.model, srv.com, ROWS, STEPS, engine=object(), out=d, config={"pouw": {"unit": "j10r", "ks": [K]}},
                       key=WKEY, sessions=sessions, prover=d / "private" / "h1")
    arm.open(0)
    srv.serve(PROMPTS[0], 0)
    out = arm.close(0)
    srv.src.uninstall()
    assert out["tied"] and out["refused"] == [], out
    PC.build(d / "h1", tmp_path / "circuits", only=(K,),
             lower=lambda k, tile, bind, path, j10r=False: PC._lower(k, *tile, bind, path, j10r))
    w = {}
    for i, layer in enumerate(arm.layers):
        w[f"w{i}"], w[f"s{i}"] = layer.weight.numpy(), layer.weight_scale.reshape(-1).numpy()
    np.savez(tmp_path / "weights.npz", **w)
    (tmp_path / "wkey").write_bytes(WKEY)
    exe = tmp_path / "flock-verify"
    exe.write_text(FAKE)
    exe.chmod(0o755)
    DR = _drawn()
    monkeypatch.setattr(DR, "reference_check", lambda *a: None)
    return DR, d, tmp_path


def _stage(DR, d, tmp, *more):
    return DR.main(["stage", str(d), "0", "--circuits", str(tmp / "circuits"), "--work", str(tmp / "work"), "--k",
                    str(_PeriodAuditor.K), "--weights", str(tmp / "weights.npz"), *more])


def _prove(plan: dict, d: Path) -> None:
    """Each staged call proved as `fake`'s verifier accepts: one session drawing all its instances, its proof bound to
    the files credit hands the verifier (the call's circuit file, its public file, the Program, its partition and the
    ``--registered`` file credit builds, which stage wrote beside it)."""
    raw = __import__("gzip").decompress((d / "h1" / SH.PROGRAM).read_bytes())
    program = hashlib.sha512(raw).hexdigest()
    part = json.dumps(CR.partition_of(program, CR.shape_of(raw)), sort_keys=True).encode()
    for x in plan["calls"]:
        c = Path(x["dir"])
        reg = json.dumps(json.loads((c / "registered.json").read_text()), sort_keys=True).encode()
        files = [Path(x["circuit"]).read_bytes(), (c / "pub.bin").read_bytes(), raw, part, reg]
        bound = hashlib.sha512(b"".join(hashlib.sha512(f).digest() for f in files)).digest()
        drew = list(range(x["instances"]))
        (c / "sessions" / "s0").mkdir(parents=True)
        (c / "sessions" / "s0" / "session.json").write_text(json.dumps(
            {"unit_draw": {"units": drew}, "proof": hashlib.sha512(bound + json.dumps(drew).encode()).hexdigest()}))


def test_a_served_windows_drawn_calls_are_staged_proved_and_credits_proved_accepts_them(served, capsys):
    DR, d, tmp = served
    assert _stage(DR, d, tmp, "--weights-key", str(tmp / "wkey")) == 0, capsys.readouterr().err
    plan = json.loads((tmp / "work" / DR.PLAN).read_text())
    manifest = json.loads((tmp / "circuits" / PC.MANIFEST).read_text())
    assert plan["calls"] and all(x["circuit"] == str((tmp / "circuits" / PC.file_name(K)).resolve()) for x in plan["calls"])
    assert all(hashlib.sha512(Path(x["circuit"]).read_bytes()).hexdigest() == manifest["files"][f"k{K}"]
               for x in plan["calls"])
    assert all((Path(x["dir"]) / f).exists() for x in plan["calls"] for f in ("inst.bin", "pub.bin", "pin.txt", "circuit.txt"))
    _prove(plan, d)
    st = tmp / "statements.json"
    assert DR.main(["statements", "--work", str(tmp / "work"), "--out", str(st)]) == 0
    assert [s["call"] for s in json.loads(st.read_text())] == [x["call"] for x in plan["calls"]]
    capsys.readouterr()
    assert DR.main(["check", str(d), "0", "--circuits", str(tmp / "circuits"), "--statements", str(st), "--lean",
                    str(tmp / "flock-verify"), "--k", str(_PeriodAuditor.K)]) == 0
    got = capsys.readouterr().out
    assert got.startswith("PROVED ") and json.loads(got.split(" ", 1)[1])["units"] == sorted(
        u for x in plan["calls"] for u in x["units"])
    first = json.loads(st.read_text())
    first[0]["sessions"] = []
    st.write_text(json.dumps(first))
    assert DR.main(["check", str(d), "0", "--circuits", str(tmp / "circuits"), "--statements", str(st), "--lean",
                    str(tmp / "flock-verify"), "--k", str(_PeriodAuditor.K)]) == 1
    assert capsys.readouterr().out.startswith("REFUSED ")


def test_a_prover_state_or_weights_key_other_than_the_served_ones_is_refused(served, capsys):
    DR, d, tmp = served
    assert _stage(DR, d, tmp) == 1 and "b/<K>'s salt key" in capsys.readouterr().err
    (tmp / "other").write_bytes(bytes(32))
    assert _stage(DR, d, tmp, "--weights-key", str(tmp / "other")) == 1
    assert "is not the tree the window registered" in capsys.readouterr().err
    for f in (d / "private" / "h1" / "w0" / PV.CALLS).glob("*.npz"):
        with np.load(f) as x:
            kept = {n: x[n].copy() for n in x.files}
        if kept["a"].size:
            kept["a"].reshape(-1)[0] ^= 1
        np.savez(f, **kept)
    assert _stage(DR, d, tmp, "--weights-key", str(tmp / "wkey")) == 1
    assert "is not the tree the window registered" in capsys.readouterr().err
