"""`drawn` (``benchmarks/pouw/h1_prove/drawn.py``'s method) on a window serve-openai's H1 arm served over J10-R units at one K (the pouw
setting ``unit`` j10r, over a stand-in tile with J10-R's signature, `StandInJ10R`, or under `REAL` J10-R's own), on a
gate session whose period opened its sample K: ``stage`` rebuilds every value a sampled call's statement reads from the
prover's state and the served weights at the roots the window registered, stages the call on the circuit file
`pouw_h1_circuits` built for K, and ``statements`` and ``check`` hand core's credit statements its ``proved`` accepts (`credit._verified`, then
`credit._proved`), here with a stand-in ``flock-verify`` that accepts a session only when its proof binds every file
credit hands it (`test_pouw_credit_statements.FAKE`'s). A prover state that is not what was served, and b/<K> under
another key, are refused before anything is staged. A window served over v1's tile unit at a K committed in the lanes
form (`pouw_h1_circuits --lanes`) has each sampled proof block staged as its lanes and combine, which credit's lanes form
accepts (`lanes_served`)."""
from __future__ import annotations

import functools
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from verity.circuits.boolean import AND, NOT, XOR, operands
from verity.circuits.defs import CompositeDefinition, bind
from verity.circuits.ports import In, Out, Ports
from verity.circuits.refs import Coll, coll_concat
from verity.circuits.types import Value
from verity.protocols.pouw.prover.circuit import h1tile as HT
from verity.protocols.pouw.prover.circuit import h1window as HW
from verity.protocols.pouw.prover.circuit import h1zy as Z
from verity.protocols.pouw.verifier import core_check as PCC
from verity.protocols.pouw.verifier import credit as CR
from verity.protocols.pouw.verifier import leaf_lanes as LL
from verity.protocols.spot.verifier.one_stage import proofs as PF

from tests.pipeline.test_commit_coins import GATE, PEER
from tests.pipeline.test_serve_h1 import PROMPTS, _PeriodAuditor, _Server
from tests.protocols import pouw_h1_circuits as PC
from tests.protocols.test_pouw_h1_commit import ROWS, STEPS, K, Model
from tests.protocols.test_pouw_h1_served import _unit
from verity_vllm.pipeline import commit_coins as CC
from verity_vllm.pipeline import serve_h1 as SH
from verity_vllm.protocols.pouw.h1 import drawn as DRAWN
from verity_vllm.protocols.pouw.h1 import drawn_rows as DR
from verity_vllm.protocols.pouw.h1 import prover as PV
from verity_vllm.protocols.pouw.h1 import served as SV

V1 = Value(1)
J10R_UNIT = HW.tile_unit


def _sig(S):
    return HW._chunked(S.K, J10R_UNIT(S.K, (S.R, S.C), HT.PouwH1Tile5)).signature()


def _bit(c, i):
    return Coll(V1, c.refs.slice(i, i + 1))


def _fold(B, bits):
    while len(bits) > 1:
        bits = [B.call(XOR, operands(*bits[i:i + 2])) if i + 1 < len(bits) else bits[i] for i in range(0, len(bits), 2)]
    return bits[0]


def _zeros(B, *colls):
    """0, read by two AND rows per 16 bits of `colls`: with u and v the XORs of the 16 bits' halves, AND(u, v) ^
    AND(u, NOT v) ^ u, all XORed together. XOR and NOT are linear in the proof system, so only an AND row reads its
    inputs' columns, and the form compiler folds AND(t, NOT t) and a XOR of a form with itself away (`forms.Circuit`)."""
    bits = [_bit(c, i) for c in colls for i in range(c.type.leaves)]
    assert len(bits) % 16 == 0, len(bits)
    zs = []
    for i in range(0, len(bits), 16):
        u, v = _fold(B, bits[i:i + 8]), _fold(B, bits[i + 8:i + 16])
        zs += [B.call(AND, operands(u, v)), B.call(AND, operands(u, B.call(NOT, v))), u]
    return _fold(B, zs)


#: a seed (a coin port and a position port, no sink returned) over J10-R's salt and position: `_zeros` of their bits
StandInSeed = CompositeDefinition("H1DrawnTestSeed", 0, ("K", "R", "C"),
                                  lambda S: ([p for p in _sig(S)[0] if p[0] in ("salt", "position")], V1),
                                  lambda B, S, salt, position: _zeros(B, salt, position), register=False,
                                  ports=Ports(params={"salt": In(kind="coin"), "position": In(kind="position")}))


def _body(B, S, a, b, zy, salt, position, roots, call, blocks):
    receipt, ok = (t for _n, t in _sig(S)[1].fields)
    seed = B.call(bind(StandInSeed, K=S.K, R=S.R, C=S.C), salt, position)
    x = B.call(XOR, operands(seed, _zeros(B, a, b, zy, roots, call, blocks)))
    return coll_concat(_sig(S)[1], [coll_concat(receipt, [B.call(NOT, B.call(NOT, _bit(zy, i)))
                                                          for i in range(receipt.leaves)]),
                                    B.call(NOT, x)])


#: J10-R's tile unit's signature (`h1window.tile_unit` over `h1tile.PouwH1Tile5`, so its statement's rows and public row
#: are J10-R's), its receipt its record's leading bits (`_stand_in_receipts`) and its ok 1, the NOT of `_zeros` of every
#: input (the coin and position through `StandInSeed`), since the Lean verifier refuses a known port (pub) or a port
#: read in full that no computed row reads in every bit (`KnownRead`)
StandInJ10R = CompositeDefinition("H1DrawnTestTileJ10R", 0, ("K", "R", "C"), _sig, _body, register=False,
                                  ports=Ports(params={"salt": In(kind="coin"), "position": In(kind="position")},
                                              ret={"receipt": Out(sink=True)}))

def _ok_body(signature):
    def body(B, S, first, *_rest):
        x = _bit(first, 0)
        return coll_concat(signature(S)[1], [B.call(NOT, B.call(XOR, operands(x, x)))])
    return body


#: the leaf statements' signatures (`h1zy.PouwH1LaneZy`'s and `h1zy.PouwH1Combine`'s over the leaf's TILE), each ok 1: the
#: lane and combine files a lanes-form K commits, a circuit of a few gates on `drawn.leaf_rows`
StandInLane = CompositeDefinition("H1DrawnTestLane", 0, ("TILE",), Z._lane_signature, _ok_body(Z._lane_signature),
                                  register=False, ports=Z._lane_ports)
StandInCombine = CompositeDefinition("H1DrawnTestCombine", 0, ("TILE",), Z._combine_signature,
                                     _ok_body(Z._combine_signature), register=False)
#: the lanes-form K: the least whose leaf's combine is proved (`h1zy.LEAST_K`)
LK = Z.LEAST_K


def _stand_in_leaf(k, kind, program, path):
    """`pouw_h1_circuits.lower_leaf`'s file of the stand-in statement over the leaf's TILE, bound as the leaf's is."""
    from verity_flock import class_statement as CS
    defn, _desc, _part, bound = PC.leaf_bound(PC.rebuilt(gzip.decompress(Path(program).read_bytes())), k, kind)
    stand_in = bind(StandInLane if kind == LL.LANE else StandInCombine, TILE=defn.bindings["TILE"])
    CS.UnitStager(stand_in, DR.LEAF_NAMES[kind], DR.leaf_rows(kind, k, (32, 32)), DR.LEAF_OUT_ROWS, circuit=Path(path),
                  bind=bound)


def _v1_receipt(served, ta, tb, tsa, tsw, zy, salt, position, roots, u0, ij, k):
    rc, ok = Z.reference(ta, tb, tsa, tsw, zy, salt, position, roots)
    assert ok == 1
    return rc


def _j10r_receipt(served, ta, tb, tsa, tsw, zy, salt, position, roots, u0, ij, k):
    """J10-R's word reference's receipt (`h1zy.reference` with ``j10r``), what `drawn.call_receipts` rebuilds."""
    blocks = PCC.j10r_blocks(served.m, k, (32, 32), *ij)
    rc, ok = Z.reference(ta, tb, tsa, tsw, zy, salt, position, roots, j10r=(u0, blocks))
    assert ok == 1
    return rc


def _stand_in_receipt(served, ta, tb, tsa, tsw, zy, salt, position, roots, u0, ij, k):
    """`StandInJ10R`'s receipt: its record's leading bits, the record's first z cells as u32 words."""
    return list(zy[0][:served.receipt_shape(0)[1] // 2])


def _stand_in_receipts(z, win, k, roots):
    """`drawn.call_receipts` under `StandInJ10R`: each unit's receipt its record row's leading u16 words."""
    return np.asarray(z["w_zy"], np.uint16)[:, :CR.shape_of(win["raw"])["tiles"][k]["receipt"]]


def _with_receipts(commit, receipt=_v1_receipt):
    """`Served.commit` with each call's zy receipts registered (`Served.receipts`, serve-openai's ``--h1-receipts``):
    `receipt` (by default v1's `h1zy.reference`) of each of its units' padded A rows, weight rows and their records
    (`h1zy.records`), the run's Program salt, the unit's position and the call's roots (and its first position and the
    unit's tile, J10-R's), as the word reference computes the receipt the statement's o0 (or the combine) reads."""
    def with_receipts(self, run, call, *a, **kw):
        if self.receipts is None:
            def receipts(c, codes, s_a, w, s_w):
                u0, n = HW.call_units(self.program)[c.index]
                pre = self.program.fn.bindings["PREFIX"]
                roots = (self.registered[self.a_table(c.index)].root, self.registered[pre + f"b/{c.k}"].root)
                aa, sa = np.zeros((self.m, c.k), np.int8), np.zeros(self.m, "<f4")
                aa[:len(codes)], sa[:len(codes)] = codes, s_a
                ww, sw = np.asarray(w, np.int8), np.asarray(s_w, "<f4")
                out = []
                for t in range(n):
                    i, j = divmod(t, c.n // 32)
                    ta, tb = aa[32 * i:32 * (i + 1)].view(np.uint8).tolist(), ww[32 * j:32 * (j + 1)].view(np.uint8).tolist()
                    tsa = [int(v) for v in sa[32 * i:32 * (i + 1)].view("<u4")]
                    tsw = [int(v) for v in sw[32 * j:32 * (j + 1)].view("<u4")]
                    rc = receipt(self, ta, tb, tsa, tsw, Z.records(ta, tb, tsa, tsw), run._salt, u0 + t, roots, u0,
                                 (i, j), c.k)
                    out.append([int(v) & 0xFFFFFFFF for v in rc])
                return np.array(out, np.uint32)
            self.receipts = receipts
        return commit(self, run, call, *a, **kw)
    return with_receipts


#: VERITY_VLLM_H1_DRAWN_J10R=1: the window served over J10-R's own tile unit (`served.unit_of` ``j10r``), so ``--recompose``
#: holds its lowered outputs to the word reference sampled stages without composing (`drawn.unit_outputs`); run remotely
REAL = os.environ.get("VERITY_VLLM_H1_DRAWN_J10R") == "1"
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
                                                  for p, e in reg["ports"].items()}} if ok else {{}},
                                   "openings": rec.get("openings")}}))
"""


@pytest.fixture
def served(tmp_path, monkeypatch):
    """One window served over the stand-in J10-R tile at K on a gate session of period p1 (sample K 2), with each call's
    receipts (`_with_receipts`: J10-R's word reference's, or under the stand-in its own, which sampled then rebuilds), a
    prover dir and b/<K> under `WKEY`; its K's circuit file built by `pouw_h1_circuits` (``--strata K``); the served
    weights as sampled's ``--weights``."""
    receipt = _j10r_receipt if REAL else _stand_in_receipt
    monkeypatch.setattr(SV.Served, "commit", _with_receipts(SV.Served.commit, receipt))
    if not REAL:
        j10r = _unit(defn=StandInJ10R)
        monkeypatch.setattr(SV, "unit_of", lambda name: j10r if name == "j10r" else None)
        monkeypatch.setattr(HW, "tile_unit", lambda k, tile, definition=None:
                            (j10r if definition is HT.PouwH1Tile5 else _unit())(k))
    srv, auditor = _Server(monkeypatch), _PeriodAuditor(tmp_path, monkeypatch)
    sessions = SH.GateSessions(CC.ChallengerGate(GATE, auditor.run), PEER, STEPS, wait=auditor.settle, listens=lambda port: True)
    d = tmp_path / "serve"
    arm = SH.H1Windows(srv.model, srv.com, ROWS, STEPS, engine=object(), out=d, config={"pouw": {"unit": "j10r", "ks": [K]}},
                       key=WKEY, sessions=sessions, prover=d / "private" / "h1")
    arm.open(0)
    srv.serve(PROMPTS[0], 0)
    out = arm.close(0)
    srv.src.uninstall()
    assert out["tied"] and out["refused"] == [], out
    PC.build(d / "h1", tmp_path / "circuits", only=(K,),
             lower=lambda k, tile, bind, path, j10r=False: PC._lower(k, tile[0], tile[1], bind, path, j10r))
    _inputs(arm, tmp_path)
    DR = DRAWN
    if not REAL:
        monkeypatch.setattr(DR, "reference_check", lambda *a: None)
        monkeypatch.setattr(DR, "call_receipts", _stand_in_receipts)
    return DR, d, tmp_path


def _inputs(arm, tmp_path: Path) -> None:
    """sampled's inputs beside the serve dir: the served weights (``--weights``), `WKEY` and the stand-in ``flock-verify``."""
    w = {}
    for i, layer in enumerate(arm.layers):
        w[f"w{i}"], w[f"s{i}"] = layer.weight.numpy(), layer.weight_scale.reshape(-1).numpy()
    np.savez(tmp_path / "weights.npz", **w)
    (tmp_path / "wkey").write_bytes(WKEY)
    exe = tmp_path / "flock-verify"
    exe.write_text(FAKE)
    exe.chmod(0o755)


@pytest.fixture
def lanes_served(tmp_path, monkeypatch):
    """One window served over v1's tile unit at K = `LK` with each call's zy receipts (`_with_receipts`) on a gate session
    of period p1 (sample K 2), with a prover dir and b/<K> under `WKEY`; K committed in the lanes form by
    `pouw_h1_circuits` (``--lanes K``) on the stand-in leaf statements (`_stand_in_leaf`)."""
    monkeypatch.setattr(SV.Served, "commit", _with_receipts(SV.Served.commit))
    monkeypatch.setattr(PC, "rebuilt", functools.cache(PC.rebuilt))
    srv, auditor = _Server(monkeypatch, model=Model(ks=(LK, LK))), _PeriodAuditor(tmp_path, monkeypatch)
    sessions = SH.GateSessions(CC.ChallengerGate(GATE, auditor.run), PEER, STEPS, wait=auditor.settle, listens=lambda port: True)
    d = tmp_path / "serve"
    arm = SH.H1Windows(srv.model, srv.com, ROWS, STEPS, engine=object(), out=d, key=WKEY, sessions=sessions,
                       prover=d / "private" / "h1")
    arm.open(0)
    srv.serve(PROMPTS[0], 0)
    out = arm.close(0)
    srv.src.uninstall()
    assert out["tied"] and out["refused"] == [], out
    doc = PC.build(d / "h1", tmp_path / "circuits", lanes=(LK,), lower_leaf=_stand_in_leaf)
    assert sorted(doc["files"]) == sorted([CR.LEAF_PROGRAMS, f"{LL.COMBINE}{LK}", f"{LL.LANE}{LK}"])
    assert CR.lanes_files(doc["files"]) is None
    _inputs(arm, tmp_path)
    return DRAWN, d, tmp_path


def _stage(DR, d, tmp, weights_key=None, recompose=False):
    return DR.run("stage", SimpleNamespace(serve_dir=d, window=0, circuits=tmp / "circuits", work=tmp / "work", prover=None,
                                           k=_PeriodAuditor.K, weights_key=weights_key, gate=None, model=None,
                                           checkpoint=None, weights=tmp / "weights.npz", recompose=recompose,
                                           checkpoint_weights=None))


def _check(DR, d, tmp, st, monkeypatch):
    return DR.run("check", SimpleNamespace(serve_dir=d, window=0, circuits=tmp / "circuits", statements=st,
                                           lean=tmp / "flock-verify", k=_PeriodAuditor.K, verified=_verified(monkeypatch)))


def _session(x: dict, program: bytes, part: dict, reg: bytes, openings: dict | None = None) -> None:
    """Staged statement `x` proved as `fake`'s verifier accepts: one session sampling all its instances, its proof bound to
    the files credit hands the verifier (its circuit file, its public file, the Program, its partition and the
    ``--registered`` file credit builds), reporting `openings` as the rows it opened."""
    c = Path(x["dir"])
    files = [Path(x["circuit"]).read_bytes(), (c / "pub.bin").read_bytes(), program,
             json.dumps(part, sort_keys=True).encode(), json.dumps(json.loads(reg), sort_keys=True).encode()]
    bound = hashlib.sha512(b"".join(hashlib.sha512(f).digest() for f in files)).digest()
    drew = list(range(x["instances"]))
    (c / "sessions" / "s0").mkdir(parents=True)
    (c / "sessions" / "s0" / "session.json").write_text(json.dumps(
        {"unit_draw": {"units": drew}, "proof": hashlib.sha512(bound + json.dumps(drew).encode()).hexdigest(),
         **({} if openings is None else {"openings": openings})}))


def _bound(x: dict, program: bytes, part: dict) -> None:
    """Staged statement `x`'s program.json and partition.json, which the ZK server's Hello is set up on, are the Program
    and partition object credit verifies it under."""
    c = Path(x["dir"])
    assert (c / "program.json").read_bytes() == program
    assert json.loads((c / "partition.json").read_text()) == part


def _prove(plan: dict, d: Path) -> None:
    """Each staged call proved (`_session`) on the window's Program and partition and the ``--registered`` file stage wrote
    beside it."""
    raw = gzip.decompress((d / "h1" / SH.PROGRAM).read_bytes())
    part = CR.partition_of(hashlib.sha512(raw).hexdigest(), CR.shape_of(raw))
    for x in plan["calls"]:
        _bound(x, raw, part)
        _session(x, raw, part, (Path(x["dir"]) / "registered.json").read_bytes())


def _prove_leaves(plan: dict, win: dict) -> None:
    """Each staged leaf statement proved (`_session`) on its leaf Program and partition (credit's `leaf_partition_of`) and
    credit's own ``--registered`` file, reporting its opening of each row the verifier holds openings of (the produced
    record's and the known rows', `produced.wanted`)."""
    produced = {e["value"]: e["root"] for e in plan["produced"]["produced"]}
    for x in plan["leaves"]:
        u, st, k = x["unit"], tuple(x["statement"]), x["k"]
        got, why = CR.leaf_registered_file(win["program"], win["layout"], win["shape"], u, st, win["roots"], win["root"],
                                           win["steps"], win["salt"], win["draw"]["root"], produced)
        assert why is None, why
        reg, known = got
        assert (Path(x["dir"]) / "registered.json").read_text() == json.dumps(reg)
        leaf = bytes.fromhex(plan["leaf_programs"][f"{st[0]}{k}"])
        part = CR.leaf_partition_of(hashlib.sha512(leaf).hexdigest(), CR.leaf_unit_id(win["shape"], k, st[0]))
        _bound(x, leaf, part)
        held = {v: dict(map(tuple, rows)) for v, rows in {**plan["produced"]["openings"], **known}.items()}
        openings = {p: [[e["reads"][0], held[e["value"]][e["reads"][0]]]] for p, e in reg["ports"].items()
                    if e["value"] in held}
        _session(x, leaf, part, json.dumps(reg).encode(), openings)


def _verified(monkeypatch):
    """`drawn.check`'s verifier as ``benchmarks/pouw/h1_prove/drawn.py`` binds it: `credit._verified` with the Lean
    verifier at ``--lean`` under the window's sample, then `credit._proved`."""
    def verified(statements, files, win, units, lean):
        monkeypatch.setattr(PF, "executable", lambda: lean)
        proofs, why = CR._verified(statements, files, win["raw"], win["layout"], win["entries"], win["root"], units,
                                   win["steps"], PF.verify, win["salt"], win["draw"])
        return proofs, why if why is not None else CR._proved(units, proofs)
    return verified


def _seeded():
    count = iter(range(1 << 30))
    return lambda n: hashlib.shake_256(next(count).to_bytes(8, "little")).digest(n)


@pytest.mark.slow
def test_a_served_windows_drawn_calls_are_staged_proved_and_credits_proved_accepts_them(served, monkeypatch, capsys):
    DR, d, tmp = served
    assert _stage(DR, d, tmp, weights_key=tmp / "wkey", recompose=True) == 0, capsys.readouterr().err
    plan = json.loads((tmp / "work" / DR.PLAN).read_text())
    manifest = json.loads((tmp / "circuits" / PC.MANIFEST).read_text())
    assert plan["calls"] and all(x["circuit"] == str((tmp / "circuits" / PC.file_name(K)).resolve()) for x in plan["calls"])
    assert all(hashlib.sha512(Path(x["circuit"]).read_bytes()).hexdigest() == manifest["files"][f"k{K}"]
               for x in plan["calls"])
    assert all((Path(x["dir"]) / f).exists() for x in plan["calls"] for f in ("inst.bin", "pub.bin", "pin.txt", "circuit.txt"))
    prefix = CR.shape_of(gzip.decompress((d / "h1" / SH.PROGRAM).read_bytes()))["prefix"]
    for x in plan["calls"]:
        reg = json.loads((Path(x["dir"]) / "registered.json").read_text())
        pub = (Path(x["dir"]) / "pub.bin").read_bytes()
        head = json.loads(pub[:pub.index(b"\n")])
        assert reg["ports"][CR.RECEIPT_PORT]["value"] == head["registered"][CR.RECEIPT_PORT]["value"] == \
            f"{prefix}receipt@{x['call']}"
        assert sorted(head["shared_rows"]) == sorted(reg["ports"])
    _prove(plan, d)
    st = tmp / "statements.json"
    assert DR.run("statements", SimpleNamespace(work=tmp / "work", out=st)) == 0
    assert [s["call"] for s in json.loads(st.read_text())] == [x["call"] for x in plan["calls"]]
    capsys.readouterr()
    assert _check(DR, d, tmp, st, monkeypatch) == 0
    got = capsys.readouterr().out
    assert got.startswith("PROVED ") and json.loads(got.split(" ", 1)[1])["units"] == sorted(
        u for x in plan["calls"] for u in x["units"])
    first = json.loads(st.read_text())
    first[0]["sessions"] = []
    st.write_text(json.dumps(first))
    assert _check(DR, d, tmp, st, monkeypatch) == 1
    assert capsys.readouterr().out.startswith("REFUSED ")


@pytest.mark.slow
def test_staging_on_the_committed_file_writes_the_recomposed_statements_bytes(served, monkeypatch, capsys):
    """`FileStager`, which reads K's committed circuit file and never composes it, writes the inst.bin and pub.bin the
    recomposed `UnitStager` does on the same rows, salts and outputs (here the stand-in's lowered outputs)."""
    DR, d, tmp = served
    outs: dict[int, object] = {}
    monkeypatch.setattr(DR, "reference_check", lambda z, o, *_a: outs.__setitem__(int(z["call"]), o))
    monkeypatch.setattr(DR, "_random", _seeded())
    assert _stage(DR, d, tmp, weights_key=tmp / "wkey", recompose=True) == 0, capsys.readouterr().err
    (tmp / "work").rename(tmp / "recomposed")
    monkeypatch.setattr(DR, "unit_outputs", lambda z, *_a: outs[int(z["call"])])
    monkeypatch.setattr(DR, "_random", _seeded())
    assert _stage(DR, d, tmp, weights_key=tmp / "wkey") == 0, capsys.readouterr().err
    plan = json.loads((tmp / "work" / DR.PLAN).read_text())
    assert plan["calls"] and sorted(outs) == [x["call"] for x in plan["calls"]]
    for x in plan["calls"]:
        for f in ("inst.bin", "pub.bin", "pin.txt"):
            assert (Path(x["dir"]) / f).read_bytes() == (tmp / "recomposed" / f"c{x['call']}" / f).read_bytes(), f


@pytest.mark.slow
def test_a_prover_state_or_weights_key_other_than_the_served_ones_is_refused(served, capsys):
    DR, d, tmp = served
    assert _stage(DR, d, tmp) == 1 and "b/<K>'s salt key" in capsys.readouterr().err
    (tmp / "other").write_bytes(bytes(32))
    assert _stage(DR, d, tmp, weights_key=tmp / "other") == 1
    assert "is not the tree the window registered" in capsys.readouterr().err
    for f in (d / "private" / "h1" / "w0" / PV.CALLS).glob("*.npz"):
        with np.load(f) as x:
            kept = {n: x[n].copy() for n in x.files}
        if kept["a"].size:
            kept["a"].reshape(-1)[0] ^= 1
        np.savez(f, **kept)
    assert _stage(DR, d, tmp, weights_key=tmp / "wkey") == 1
    assert "is not the tree the window registered" in capsys.readouterr().err


def test_a_call_statements_port_that_is_no_registered_read_is_refused_by_name():
    """`drawn.complete` is the Lean verifier's ``registered reads (complete)``: every port of the statement's circuit is
    a port of credit's ``--registered`` file. o0, the receipt, is refused with the serving flag that registers it."""
    ports = ["a_k64_0", "b_k64_0", "zy", "pub", "o0", "o1"]
    DRAWN.complete(ports, {"ports": dict.fromkeys(ports, {})}, 3)
    with pytest.raises(DRAWN.Refused, match=r"call 3's statement's port o0 is no port of credit's registered reads.*"
                                            r"receipt@3.*--h1-receipts"):
        DRAWN.complete(ports, {"ports": dict.fromkeys([p for p in ports if p != "o0"], {})}, 3)
    with pytest.raises(DRAWN.Refused, match="port zy is no port of credit's registered reads") as e:
        DRAWN.complete(ports, {"ports": dict.fromkeys(["a_k64_0", "b_k64_0", "pub", "o1"], {})}, 3)
    assert "--h1-receipts" not in str(e.value)
    z = {"units": np.array([5, 6]), "r_o0": np.array([0, 1]), "v_o0": np.array(["receipt@0"] * 2),
         "w_o0": np.array([[1, 2], [3, 4]], np.uint16)}
    DRAWN.outputs_held(z, {"o0": z["w_o0"]})
    with pytest.raises(DRAWN.Refused, match="unit 6: its o0 is not row 1 of receipt@0, which the window registered"):
        DRAWN.outputs_held(z, {"o0": np.array([[1, 2], [3, 5]], np.uint16)})


@pytest.mark.slow
def test_a_window_without_its_receipts_or_a_unit_whose_receipt_is_not_the_registered_one_is_refused(served, monkeypatch,
                                                                                                     capsys):
    """A call statement's o0 is opened at the call's registered receipt table: a window that registers none is refused
    before anything is staged, by the port the Lean verifier would refuse, and a staged o0 that is not the row the
    window registered is refused before it is written."""
    DR, d, tmp = served
    window_of = DR.window_of
    monkeypatch.setattr(DR, "window_of", lambda *a: {**(w := window_of(*a)), "roots": {
        v: r for v, r in w["roots"].items() if "receipt@" not in v}})
    assert _stage(DR, d, tmp, weights_key=tmp / "wkey") == 1
    err = capsys.readouterr().err
    assert "port o0 is no port of credit's registered reads" in err and "--h1-receipts" in err, err
    assert not (tmp / "work" / DR.PLAN).exists()
    monkeypatch.setattr(DR, "window_of", window_of)
    honest = DR.call_receipts
    monkeypatch.setattr(DR, "unit_outputs", lambda z, *a: {"o0": honest(z, *a) ^ np.uint16(1),
                                                           "o1": np.tile(DR.OK, (len(z["units"]), 1))})
    assert _stage(DR, d, tmp, weights_key=tmp / "wkey") == 1
    assert "its o0 is not row" in (err := capsys.readouterr().err) and "receipt@" in err, err


@pytest.mark.slow
def test_a_lanes_form_ks_drawn_units_are_staged_as_their_lanes_and_combine_and_credit_accepts_them(lanes_served,
                                                                                                   monkeypatch, capsys):
    """Each sampled proof block at a lanes-form K is staged as `leaf_lanes.statements` on the committed lane and combine files,
    its partials one produced record under the window's sample, its registered files credit's own; credit's lanes form
    accepts the statements and refuses a leaf without its combine, and a session that reports another opening of a
    partials row (P-shared). A window that registers no receipts is refused before anything is staged. Slow (the window
    Program over v1's real tile unit at K 384 takes about a minute to declare, and the lanes' word references run in
    Python): it is the one test that takes a served window through the lanes form's staging and credit's verification of
    it."""
    DR, d, tmp = lanes_served
    win = DR.window_of(d, 0)
    drawn = sorted(u for us in DR.drawn_calls(win, _PeriodAuditor.K).values() for u in us)
    assert _stage(DR, d, tmp, weights_key=tmp / "wkey") == 0, capsys.readouterr().err
    plan = json.loads((tmp / "work" / DR.PLAN).read_text())
    assert plan["calls"] == [] and sorted({x["unit"] for x in plan["leaves"]}) == drawn
    for u in drawn:
        assert [x["statement"] for x in plan["leaves"] if x["unit"] == u] == [list(s) for s in LL.statements(32, 32)]
    assert sorted(e["value"] for e in plan["produced"]["produced"]) == sorted(
        LL.served(win["shape"]["prefix"], LL.PARTIALS, u) for u in drawn)
    _prove_leaves(plan, win)
    st = tmp / "statements.json"
    assert DR.run("statements", SimpleNamespace(work=tmp / "work", out=st)) == 0
    evidence = json.loads(st.read_text())
    assert sorted(evidence) == list(CR.LANES_KEYS) and len(evidence["statements"]) == len(plan["leaves"])
    capsys.readouterr()
    assert _check(DR, d, tmp, st, monkeypatch) == 0
    got = capsys.readouterr().out
    assert got.startswith("PROVED ") and json.loads(got.split(" ", 1)[1])["units"] == drawn, got

    partial = plan["produced"]["openings"][LL.served(win["shape"]["prefix"], LL.PARTIALS, drawn[0])][0][1]
    lane0 = next(x for x in plan["leaves"] if x["unit"] == drawn[0] and x["statement"] == [LL.LANE, 0])
    rec = Path(lane0["dir"]) / "sessions" / "s0" / "session.json"
    honest = rec.read_text()
    tampered = json.loads(honest)
    tampered["openings"][LL.PARTIALS] = [[0, ("0" if partial[0] != "0" else "1") + partial[1:]]]
    rec.write_text(json.dumps(tampered))
    assert _check(DR, d, tmp, st, monkeypatch) == 1
    assert "registration" in (out := capsys.readouterr().out) and "P-shared" in out, out
    rec.write_text(honest)

    st.write_text(json.dumps({**evidence, "statements": [s for s in evidence["statements"]
                                                         if s.get("statement") != [LL.COMBINE, None]]}))
    assert _check(DR, d, tmp, st, monkeypatch) == 1
    assert capsys.readouterr().out.startswith("REFUSED missing")

    window_of = DR.window_of
    monkeypatch.setattr(DR, "window_of", lambda *a: {**(w := window_of(*a)), "roots": {
        v: r for v, r in w["roots"].items() if f"{LL.RECEIPT}@" not in v}})
    assert _stage(DR, d, tmp, weights_key=tmp / "wkey") == 1
    assert "serve the window with its zy receipts" in capsys.readouterr().err
