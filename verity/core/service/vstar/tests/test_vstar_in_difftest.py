"""V*'s per-part port inputs in Lean (`VStar.In`, served by `vstar-difftest`'s ``part_in``) against what the Python stager
writes to each part's ``inst.bin``: per port of the part's circuit, in order, its table rows (u16 words) and their salts as
fingerprints (the first 8 bytes of SHA-512), and the row each instance reads.

The drawn cases stage one statement each with the stager's own code on a small consistent session (openings that climb to
their tops, both reps' running sums on one chain, the inner prover's message rows registered as ``rec-c<i>``): a level
(`rec_outer.stage` at the stager's `rec_vstage.coefficients` and `rec_outer.Chain.build`), a part of the algebra
(`rec_vstage._stage_part`, reading what `rec_vstage.stage` hands it) and a cap (`rec_cap.stage`). The prover's fresh salts
(a level's ``row`` and ``aux``) are the case's, handed to the stager through ``hm96``'s salt source. With
``VERITY_VSTAR_IN_DEPLOYED=1`` two recorded sessions are staged with `rec_vstage.stage` and each statement held to Lean's:
`rec-coins`' (16,16,8) session, whose leaves are plain, at every level, cap and part of the algebra; and rec-algebra's
fixture session (the deployed algebra shape, m = 25, k_log = 22) at every part of the algebra and every level's cap (its
openings are hm96 leaves, whose levels the stager refuses to stage)."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import verity_difftest as D
from hypothesis import strategies as st
from verity_difftest import settings
from verity_flock import rec_algebra as A
from verity_flock import rec_cap as RC
from verity_flock import rec_live as RL
from verity_flock import rec_outer as O
from verity_flock import rec_residuals as RR
from verity_flock import rec_vstage as S
from verity_flock import rec_vstar as V

from verity.ml.commitments import hm96

LEAN = Path(__file__).resolve().parents[1]
FLOCK_TESTS = LEAN.parents[2] / "ml" / "flock" / "tests"
pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

SEED = 20261026
SALT = hm96.SHA512.salt_bytes
BC = 128
LANES = 8
PROGRAM = "5e" * 64
#: a part of the algebra: ``m0 · v0 + acc0[0]``, ``m1[1] + acc1[1]`` (`InnerRepCheck`'s S, read through `rec_residuals.parts`' form)
ALG_S = {"gf": {"nw": 7, "nv": 1, "ops": [[["w", 0]], [["v", 0]]],
                "res": [{"mul": [[0, 1]], "add": [["w", 3]]}, {"mul": [], "add": [["w", 2], ["w", 6]]}]},
         "ports": [["m0", 1], ["m1", 2], ["acc0", 2], ["acc1", 2]]}


def short(b: bytes) -> str:
    return hashlib.sha512(b).digest()[:8].hex()


def ports_of(d: Path) -> dict:
    """A staged statement's ``inst.bin`` (`circuit.write` with shared rows: the header line, every port's table rows in the
    circuit's port order, their salts, their ``b || c``, then each instance's refs) as `part_in` answers: each port's rows
    and salts as fingerprints, and its refs."""
    data = (d / "inst.bin").read_bytes()
    nl = data.index(b"\n") + 1
    head = json.loads(data[:nl])
    ports = json.loads((d / "circuit.txt").read_text().split("\n", 2)[1][5:])["ports"]
    at, tables = nl, []
    for name, words, *_ in ports:
        k = head["shared_rows"][name]
        tables.append((name, [data[at + 2 * words * i:at + 2 * words * (i + 1)] for i in range(k)]))
        at += 2 * words * k
    total = sum(len(rows) for _, rows in tables)
    refs_at = at + (SALT + BC) * total
    n = head["instances"]
    if refs_at + 4 * n * len(ports) != len(data):
        raise AssertionError(f"{d}: inst.bin is not the shared-row layout")
    refs = np.frombuffer(data[refs_at:], "<u4").reshape(n, len(ports))
    out = []
    for p, (name, rows) in enumerate(tables):
        salts = [data[at + SALT * i:at + SALT * (i + 1)] for i in range(len(rows))]
        at += SALT * len(rows)
        out.append([name, {"rows": [short(r) for r in rows], "salts": [short(s) for s in salts], "refs": refs[:, p].tolist()}])
    return {"ports": out}


# --- the stager's inputs, as Python objects and as `part_in`'s JSON -------------------------------------------------------------

def h128(x: int) -> str:
    return f"{x:032x}"


def _top(nodes: list[bytes], salts: list[bytes]) -> V.Top:
    return V.Top(tuple(nodes), tuple(salts), tuple(RL.commit_rows(RL.top_rows(b"".join(nodes)), salts)))


def session(g: random.Random, levels: list[tuple[int, int, int]], rounds: list[int], nv: int) -> SimpleNamespace:
    """A small consistent session: ``levels`` (H, c, openings a rep) of `LANES` lanes, level 0's top both reps' one tree, every
    opening climbing to its own node of its rep's top; message rounds of ``rounds`` elements; each rep's v of ``nv``."""
    def salt():
        return g.randbytes(SALT)
    qs, tops = [], {0: [], 1: []}
    for lv, (H, c, q) in enumerate(levels):
        reps = [0] if lv == 0 else [0, 1]
        for rep in reps:
            nodes = [g.randbytes(64) for _ in range(2 ** c)]
            free = list(range(2 ** c))
            g.shuffle(free)
            for r in ([0, 1] if lv == 0 else [rep]):
                for j in range(q):
                    t, row = free.pop(), [g.getrandbits(128) for _ in range(LANES)]
                    sibs = [g.randbytes(64) for _ in range(H)]
                    pos = (t << H) | g.getrandbits(H) if H else t
                    nodes[t] = V.climb(V.leaf(row), pos, sibs)
                    qs.append(V.Query(r, lv, j, pos, H + c, c, nodes[t], row, None, sibs))
            top = _top(nodes, [salt() for _ in nodes])
            for r in ([0, 1] if lv == 0 else [rep]):
                tops[r].append(top)
    info = [{"kappa": [[[g.getrandbits(128) for _ in range(LANES)] for _ in range(2)] for _ in levels],
             "queries": [([q.pos for q in sorted(qs, key=lambda q: q.j) if q.rep == r and q.level == lv],
                          [g.getrandbits(128) for _ in range(levels[lv][2])]) for lv in range(len(levels))]} for r in range(2)]
    mrows = [[[g.getrandbits(16) for _ in range(RR.port_words(n))] for _ in range(2)] for n in rounds]
    msalts = [[salt() for _ in range(2)] for _ in rounds]
    return SimpleNamespace(qs=qs, tops=tops, info=info, vs=[[g.getrandbits(128) for _ in range(nv)] for _ in range(2)],
                           mrs=[{"rows": r, "salts": s, "commits": RL.commit_rows(r, s)} for r, s in zip(mrows, msalts)],
                           acc_salts=[salt() for _ in range(len(qs) + 1)],
                           fresh={f"L{lv}": {p: [salt() for _ in range(2 * levels[lv][2])] for p in ("row", "aux")}
                                  for lv in range(len(levels))})


def src_json(s: SimpleNamespace) -> dict:
    nlev = len(s.info[0]["kappa"])
    return {"msgs": [[[mr["rows"][r], mr["salts"][r].hex()] for r in range(2)] for mr in s.mrs],
            "openings": [{"rep": q.rep, "level": q.level, "j": q.j, "pos": q.pos, "row": [h128(x) for x in q.row],
                          "sibs": [x.hex() for x in q.sibs]} for q in s.qs],
            "tops": [[[[n.hex(), t.hex()] for n, t in zip(top.nodes, top.salts)] for top in s.tops[r]] for r in sorted(s.tops)],
            "acc_salts": [x.hex() for x in s.acc_salts],
            "kappa": [[[[h128(x) for x in row] for row in k] for k in s.info[r]["kappa"]] for r in range(2)],
            "eq_a": [[[h128(x) for x in s.info[r]["queries"][lv][1]] for lv in range(nlev)] for r in range(2)],
            "vs": [[h128(x) for x in v] for v in s.vs],
            "fresh": {d: {p: [x.hex() for x in xs] for p, xs in ports.items()} for d, ports in s.fresh.items()}}


def _vf(s: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(reps=[SimpleNamespace(info=s.info[r], v=s.vs[r]) for r in range(2)])


def _chain(s: SimpleNamespace) -> tuple[O.Chain, dict]:
    coefs = S.coefficients(_vf(s), s.qs)
    return O.Chain.build(PROGRAM, s.qs, coefs, s.acc_salts), coefs


def _tops(s: SimpleNamespace, level: int) -> O.TopValue:
    return O.TopValue.build(PROGRAM, level, {rep: t[level] for rep, t in s.tops.items()})


def _fresh(salts: list[bytes]):
    """`hm96`'s salt source yielding ``salts`` in order: the stager's fresh salts, the case's."""
    it = iter(salts)
    return mock.patch.object(type(hm96.SHA512), "fresh_salt", lambda self: next(it))


# --- the stager, one statement --------------------------------------------------------------------------------------------------

def stage_level(s: SimpleNamespace, level: int, d: Path) -> tuple[dict, dict]:
    chain, coefs = _chain(s)
    lv = next(x for x in O.levels(s.qs) if x["level"] == level)
    with _fresh(s.fresh[f"L{level}"]["row"] + s.fresh[f"L{level}"]["aux"]):
        res = O.stage(lv, d, chain, coefs, _tops(s, level))
    return res, {"kind": "level", "level": level, "LANES": lv["LANES"], "H": lv["H"], "n": len(lv["queries"])}


def stage_part(s: SimpleNamespace, Sp: dict, v: list[int], d: Path) -> tuple[dict, dict]:
    """`rec_vstage._stage_part` on part ``Sp`` (its v elements ``v``), with the reads `rec_vstage.stage` hands it."""
    chain, _ = _chain(s)
    regs, own = S.messages(PROGRAM, s.mrs)
    a0, a1 = zip(*(chain.rep_ends(r) for r in range(2)))
    reads = {nm: (regs[nm], [0, 1]) for nm in regs}
    reads["acc0"], reads["acc1"] = (chain.reg, list(a0)), (chain.reg, list(a1))
    sh = SimpleNamespace(m=22, k_log=6, claims=2, holo=False)
    S._PART.update(sh=sh, parts=[SimpleNamespace(S=Sp, res=list(range(len(Sp["gf"]["res"]))), v=v)], reads=reads,
                   vs=[list(r) for r in s.vs], out=d, evaluate=False, own=own, forge=None, mrs=s.mrs)
    try:
        res, _ = S._stage_part(0)
    finally:
        S._PART.clear()
    shutil.move(str(d / "alg" / "p0"), str(d / "part"))
    return res, {"kind": "inner", "ports": Sp["ports"], "v": v}


def stage_cap(s: SimpleNamespace, level: int, d: Path) -> tuple[dict, dict]:
    tops = _tops(s, level)
    return RC.stage(tops, d, evaluate=False), {"kind": "cap", "level": level, "N": tops.own["leaves"]}


#: the stager's ports (`ports_of`) by the canonical text of the input Lean reads for them
_STAGED: dict[str, dict] = {}


def staged(seed: int, kind: str, levels: list) -> dict:
    """One statement of a drawn session staged with the stager's own code: the input Lean reads (the session's ``src``, the
    staged program's SHA-512 and the statement), the stager's ports kept under it for `stager`. ``levels``: (H, c, openings a
    rep), the openings capped to what a top of ``2^c`` nodes holds."""
    lv = [(H, c, max(1, min(q, 2 ** c // (2 if i == 0 else 1)))) for i, (H, c, q) in enumerate(levels)]
    s = session(random.Random(seed), lv, [1, 2], 3)
    level = seed % len(lv)
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        if kind == "level":
            res, stmt = stage_level(s, level, d)
        elif kind == "inner":
            (res, stmt), d = stage_part(s, ALG_S, [2], d), d / "part"
        else:
            res, stmt = stage_cap(s, level, d)
        value = {"src": src_json(s), "prog": res.get("program_sha512", ""), "stmt": stmt}
        _STAGED[D.canonical(value)] = ports_of(d)
    return value


def stager(value: dict) -> dict:
    """The reference: the ports the stager wrote to ``inst.bin`` on the input Lean reads."""
    return _STAGED[D.canonical(value)]


PART_IN = D.Port("vstar-difftest", "part_in", stager, st.nothing(), package=LEAN)
server = D.server_fixture(PART_IN)


@pytest.mark.parametrize("kind", ["level", "inner", "cap"])
def test_part_in_agrees_with_the_stagers_inst_bin(server, kind):
    """Each kind of part on drawn sessions: Lean's port inputs are the stager's ``inst.bin``, port for port."""
    g = random.Random(SEED)
    for _ in range(4):
        levels = [(g.randint(0, 2), g.randint(1, 2), g.randint(1, 2)) for _ in range(g.randint(1, 2))]
        assert D.replay(PART_IN, staged(g.getrandbits(32), kind, levels), server)[0] == "output"


def test_a_wrong_input_moves_the_fingerprints(server):
    """The comparison is not vacuous: one bit of a fresh salt moves exactly the port that reads it."""
    honest = staged(7, "level", [(1, 2, 2)])
    assert D.replay(PART_IN, honest, server)[0] == "output"
    bad = json.loads(json.dumps(honest))
    x = bad["src"]["fresh"]["L0"]["aux"][0]
    bad["src"]["fresh"]["L0"]["aux"][0] = f"{int(x[:2], 16) ^ 1:02x}" + x[2:]
    def ports(v):
        return json.loads(server.call("part_in", v)[1])["ports"]
    assert [nm for (nm, a), (_, b) in zip(ports(honest), ports(bad), strict=True) if a != b] == ["aux"]


def top_sizes(v: dict) -> dict:
    """A rep's tops' node counts in `VStar.In.Src.tops`' order: level 0's (`root_B`, ``2^cap`` of fast100(m)'s first level),
    then the later levels' in the order the rep's schedule commits them (`rec_live.rep_rounds`, `holo_rounds`)."""
    try:
        cap = A.fast100(v["m"]).levels[0].cap
        rounds = (RL.holo_rounds if v["holo"] else RL.rep_rounds)(v["m"], v["k_log"], v["claims"])
    except (ValueError, AssertionError) as e:
        raise D.Reject(str(e)) from None
    return {"level0": 1 << cap, "later": [n for _, tops, _, _ in rounds for n in tops]}


TOP_SIZES = D.Port("vstar-difftest", "top_sizes", top_sizes,
                   st.fixed_dictionaries({"m": st.integers(20, 37), "k_log": st.integers(6, 24), "claims": st.integers(0, 12),
                                          "holo": st.booleans()}), package=LEAN)


def test_top_sizes_agree(server):
    """Lean's order of a rep's tops (`RecVStar.topSizes`, which `VStarBuild.topsOf` lists them in after level 0) is
    `rec_live`'s schedule's, across fast100's m, k_log and claims, holo or not; m outside 22 to 35 is refused by both. k_log is at least 6, a block's 64 elements: below it Lean's
    `SchedRound` (whether a round absorbs, not how much) can't tell z_partial's round from a 2-element one, which
    `holo_rounds` refuses (m = 22, k_log = 0)."""
    D.run(TOP_SIZES, examples=200, server=server)


def _tops_in_schedule_order(server, sh, ts: dict) -> None:
    """The stager's tops (`rec_vstar.tops`, `Src.tops`), each rep's: level 0's `root_B` (both reps' one top) at index 0, then
    the later levels' in `topSizes` order, as Lean's `top_sizes` gives them at the session's shape."""
    v = {"m": sh.m, "k_log": sh.k_log, "claims": sh.claims, "holo": bool(sh.holo)}
    kind, out = D.replay(TOP_SIZES, v, server)
    assert kind == "output", out
    want = json.loads(out)
    assert ts[0][0] == ts[1][0], "level 0's top is not both reps' one root_B"
    for rep, tops in ts.items():
        assert [len(t.nodes) for t in tops] == [want["level0"], *want["later"]], rep


# --- the deployed session ------------------------------------------------------------------------------------------------------

SET = FLOCK_TESTS / "fixtures" / "hidden-outputs" / "flat" / "gemm-coordinate"


def _inner(tmp: Path) -> Path:
    """rec-algebra's fixture session replayed through the firewall (`rec_live.Proxy`, seeded salts), as
    ``verity_flock``'s `test_rec_vstage` replays it."""
    rec = json.loads((SET / "records-cpu" / "honest.json").read_text())["record"]
    g, pending = random.Random(5), bytearray()

    def rand(n: int) -> bytes:
        out = bytes(pending[:n])
        del pending[:n]
        assert len(out) == n, "a coin the record does not hold"
        return out

    def coins(hexes):
        return b"".join(RL.f128_bytes(int(c, 16)) for c in hexes)
    proxy = RL.Proxy(RL.CoinServer(rand), lambda: g.randbytes(192),
                     schedule=RL.schedule_for({"dense_m": 25, "k_log": 22, "nbl": 3, "regions": 3}))
    def send(*req):
        return RL.decode_resp(proxy.handle(RL.encode_req(*req)))
    hello = json.loads(rec["hello"])
    del hello["coins"]
    lk = rec["link"]
    pending += coins(x for p in lk["points"] for x in p)
    assert send(RL.REQ_HELLO, json.dumps(hello))[0] == RL.RESP_OK
    assert send(RL.REQ_COMMIT, bytes.fromhex(lk["root_f"]), [(t, bytes.fromhex(v)) for t, v in lk["root_b"].items()], [])[0] == RL.RESP_COINS
    assert send(RL.REQ_LINK, coins(lk["y"]))[0] == RL.RESP_OK
    for st_ in rec["streams"]:
        assert send(RL.REQ_OPEN, st_["stream"], st_["parent"] or "", st_["label"].encode(), st_["at"])[0] == RL.RESP_OK
        for rd in st_["rounds"]:
            pending += coins(rd["coins"])
            assert send(RL.REQ_ROUND, st_["stream"], bytes.fromhex(rd["msg"]), rd["n"])[0] == RL.RESP_COINS
    for r in (0, 1):
        assert send(RL.REQ_PROOF, "circuit", r, (SET / "records-cpu" / f"honest.rep{r}.proof").read_bytes())[0] == RL.RESP_OK
    assert send(RL.REQ_FINISH)[0] == RL.RESP_VERDICT
    proxy.write(tmp)
    return tmp


def _salts_of(d: Path, port_: str) -> list[bytes]:
    data = (d / "inst.bin").read_bytes()
    nl = data.index(b"\n") + 1
    head = json.loads(data[:nl])
    ports = json.loads((d / "circuit.txt").read_text().split("\n", 2)[1][5:])["ports"]
    at = nl + sum(2 * w * head["shared_rows"][nm] for nm, w, *_ in ports)
    for nm, _w, *_ in ports:
        k = head["shared_rows"][nm]
        if nm == port_:
            return [data[at + SALT * i:at + SALT * (i + 1)] for i in range(k)]
        at += SALT * k
    raise AssertionError(port_)


def _agrees(server, inner: Path, fold: dict, out: Path, salts: Path, levels: list[int] | None, pts: dict | None = None,
            version: int = 1) -> list[str]:
    """Stage ``inner`` (`rec_vstage.stage`, the given levels, the algebra) and hold every staged statement to Lean's port
    inputs on the stager's inputs: the inner session's records, the salts file, the levels' fresh salts read back from their
    ``inst.bin``. With ``levels=[]`` each level's cap is staged on its own (`rec_cap.stage`). Returns the statements held."""
    res = S.stage(inner, fold, out, levels=levels, salts=salts, pts=pts, evaluate=False, part_jobs=1, version=version)
    public, private, proofs = V.load(inner)
    vf = S.verifier(public, fold, pts)
    qs = V.queries(public, private, proofs)
    ts = {rep: V.tops(public, private, f"circuit/rep{rep}", "circuit") for rep in sorted(proofs)}
    mrs = S.message_rounds(public, private, vf.sh)
    _tops_in_schedule_order(server, vf.sh, ts)
    nlev = len(vf.reps[0].info["kappa"])
    caps = {"cap": (0, res["cap"])} if "cap" in res else {}
    caps |= {RC.dir_name(int(k[1:])): (int(k[1:]), c) for k, c in res.get("caps", {}).items()}
    if levels == []:
        for lv in range(nlev):
            nm = "cap" if lv == 0 else RC.dir_name(lv)
            caps[nm] = (lv, RC.stage(O.TopValue.build(res["program"], lv, {r: t_[lv] for r, t_ in ts.items()}), out / nm, False))
    src = {"msgs": [[[mr["rows"][r], mr["salts"][r].hex()] for r in range(2)] for mr in mrs],
           "openings": [{"rep": q.rep, "level": q.level, "j": q.j, "pos": q.pos, "row": [h128(x) for x in q.row],
                         "sibs": [x.hex() for x in q.sibs]} for q in qs],
           "tops": [[[[n.hex(), t.hex()] for n, t in zip(top.nodes, top.salts)] for top in ts[r]] for r in sorted(ts)],
           "acc_salts": [x.hex() for x in S.acc_salts(salts, len(qs) + 1)],
           "kappa": [[[[h128(x) for x in row] for row in k] for k in vf.reps[r].info["kappa"]] for r in range(2)],
           "eq_a": [[[h128(x) for x in vf.reps[r].info["queries"][lv][1]] for lv in range(nlev)] for r in range(2)],
           "vs": [[h128(x) for x in rv.v] for rv in vf.reps],
           "fresh": {nm: {p: [x.hex() for x in _salts_of(out / nm, p)] for p in ("row", "aux")} for nm in res["levels"]}}
    checked = []

    def agrees(nm: str, prog: str, stmt: dict, d: Path) -> None:
        value = {"src": src, "prog": prog, "stmt": stmt}
        _STAGED[D.canonical(value)] = ports_of(d)
        assert D.replay(PART_IN, value, server)[0] == "output", nm
        checked.append(nm)

    for nm, lv in res["levels"].items():
        agrees(nm, lv["program_sha512"], {"kind": "level", "level": lv["level"], "LANES": lv["LANES"], "H": lv["H"],
                                          "n": lv["instances"]}, out / nm)
    for j, (p, s) in enumerate(zip(RR.parts(RR.spec(vf.sh)), res["algebra"], strict=True)):
        agrees(f"alg/p{j}", s["program_sha512"], {"kind": "inner", "ports": p.S["ports"], "v": p.v}, out / "alg" / f"p{j}")
    for nm, (level, c) in caps.items():
        agrees(nm, c.get("program_sha512", ""), {"kind": "cap", "level": level, "N": c["instances"]}, out / nm)
    print(json.dumps({"session": str(inner), "checked": checked}))
    return checked


DEPLOYED = pytest.mark.skipif(not settings.get("VERITY_VSTAR_IN_DEPLOYED"), reason="stages a recorded session's every "
                              "statement (the pod's): set VERITY_VSTAR_IN_DEPLOYED=1")
COINS = FLOCK_TESTS / "fixtures" / "rec-coins" / "16-16-8-n64"


def _fetched(d: Path, fixture: str):
    return pytest.mark.skipif(not d.is_dir(), reason=f"{d} is not fetched: `uv run research data fetch-fixtures "
                              f"verity/ml/flock/tests/fixtures/{fixture}`")


@pytest.mark.slow
@DEPLOYED
@_fetched(COINS, "rec-coins")
def test_the_recorded_plain_leaf_session_agrees(server, tmp_path):
    """`rec-coins`' recorded (16,16,8) session at 64 instances, whose inner leaves are plain: every level, every level's cap
    and every part of the algebra staged with `rec_vstage.stage` (at the session's fold, points and running-sum salts) is
    Lean's port inputs."""
    checked = _agrees(server, COINS / "inner-honest", json.loads((COINS / "fold-honest.json").read_text()), tmp_path / "out",
                      COINS / "acc-salts.bin", None, json.loads((COINS / "points-honest.json").read_text()), version=0)
    assert any(c.startswith("L") for c in checked) and "cap" in checked and any(c.startswith("alg/") for c in checked)


@pytest.mark.slow
@DEPLOYED
@_fetched(SET, "hidden-outputs")
def test_the_deployed_algebra_shape_agrees(server, tmp_path):
    """rec-algebra's fixture session (m = 25, k_log = 22, the deployed algebra shape) replayed through the firewall: every
    part of the algebra and every level's cap is Lean's port inputs (its openings are hm96 leaves, so the stager stages no
    level of it)."""
    INNER = json.loads((FLOCK_TESTS / "fixtures" / "inner_claims.json").read_text())
    fold = {"m": INNER["m"], "k_log": INNER["k_log"], "extras": INNER["reps"][0]["extras"],
            "reps": [{"comb": d["comb"]} for d in INNER["reps"]]}
    checked = _agrees(server, _inner(tmp_path / "inner"), fold, tmp_path / "out", tmp_path / "acc-salts.bin", [])
    assert "cap" in checked and any(c.startswith("alg/") for c in checked)
