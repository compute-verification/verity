"""`verity_flock.rec_session`, candidate 3's inner session: the holo schedule (`rec_live.holo_rounds`), h as the hidden
circuits' share of the lincheck's fold (`hidden_fold`, against the fold written out entry by entry), and the relay over a
socket against a fake prover that replays rec-algebra's fixture session: the prover sees its own session and coins, the
verifier's record has h in a round of its own after z_partial's coinless one on the rep's one coin stream, and
`rec_vstage` stages the holo algebra from that record, with h read from the session's rows, and it holds (Lean's fold of
the fixture, `fixtures/inner_claims.json`, split as public comb + h). The stacked table's stream, as candidate 3's
opening-only prover sends it (``flock-circuit stacked``), is held to `rec_live.schedule_for`'s, and the relay runs each
rep's sparse phase (`rec_sparse.prove`) on the coin server's coins, which `rec_vstage` reads back and stages as RecSparse. That the verifier's public comb plus h is Lean's own
full fold (``InnerFold.lean --hide``) is the pod's (``pod/88-rec-c3-inner.sh``, `check_fold`)."""
from __future__ import annotations

import json
import os
import queue
import random
import re
import shutil
import socket
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from verity.core.protocols.sampled_proofs.one_stage import partition as P, registered as RG, registration as R
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.ops.boolean import gf2k as G
from verity.ml.circuits.ops.boolean.gf2k import mul
from verity.ml.circuits.types import Value
from verity_flock import circuit as FC
from verity_flock import inner_registered as IR
from verity_flock import rec_algebra as A
from verity_flock import rec_live as RL
from verity_flock import rec_outer as O
from verity_flock import rec_residuals as RR
from verity_flock import rec_session as RS
from verity_flock import rec_sparse as SP
from verity_flock import rec_vstage as S
from verity_flock import rec_vstar as V

HERE = Path(__file__).resolve().parent
SET = HERE / "fixtures" / "hidden-outputs" / "flat" / "gemm-coordinate"
INNER = json.loads((HERE / "fixtures" / "inner_claims.json").read_text())
STATEMENT = {"dense_m": 25, "k_log": 22, "nbl": 3, "regions": 3}
M, K_LOG = STATEMENT["dense_m"], STATEMENT["k_log"]
_XOR8 = PrimitiveDefinition("RecSessionTestXor8", 1, (("a", Value(8)), ("b", Value(8))), Value(8), lambda a, b: a ^ b,
                            register=False)
_UNIT = bind(CompositeDefinition("RecSessionTestUnit", 0, (), lambda S: ((("x", Value(8)), ("w", Value(8))), Value(8)),
                                 lambda B, S, x, w: B.call(_XOR8, x, w), register=False))


def _private(n: int = 3):
    """A private circuit's program and query: ``n`` instances of a toy unit."""
    return P.population_program(_UNIT, n), P.template_instance_query(_UNIT)


def _partition():
    """`_private`'s program and its ``verity/partition/v1`` object, which the toy statements' META names."""
    program, query = _private()
    return program, P.partition(program, query)


def _registered_top(stage: Path, cap: bytes, seed: int) -> tuple[dict, str, dict]:
    """``cap`` registered with the statement staged in ``stage`` as the value stacked (`rec_live.register_top` under salts
    from ``seed``, `inner_registered.bind`), read back as `rec_session.serve` reads it (`rec_session.staged_stacked`): the
    circuit registration META binds, its digest and the prover's opening."""
    program, part = _partition()
    g = random.Random(seed)
    reg = RL.register_top(part["program"], RL.STACKED, cap, lambda: g.randbytes(192))
    IR.bind(stage, program, part, [reg["entry"]], {RL.STACKED: {"nodes": reg["nodes"], "salts": reg["salts"]}})
    rec, bound, own = RS.staged_stacked(stage)
    assert own == {k: reg[k] for k in ("nodes", "salts", "top")}
    return rec, bound, own


@pytest.mark.parametrize("shape", [(25, 22, 8), (22, 10, 2)], ids=str)
def test_the_holo_schedule_gives_h_a_round_after_z_partials_coinless_one(shape):
    """z_partial's round squeezes nothing and h's round (64 elements) squeezes r′; every other round is the plain schedule's,
    the message rounds are the holo algebra's (`rec_algebra.rounds`), and only the coinless round is no draw."""
    m, k_log, claims = shape
    plain, holo, k = RL.rep_rounds(m, k_log, claims), RL.holo_rounds(m, k_log, claims), RL.zp_round(m, k_log)
    assert plain[k] == (1, [], 0, 64) and holo[k:k + 2] == [(0, [], 0, 64), (1, [], 0, 64)]
    assert holo[:k] + holo[k + 2:] == plain[:k] + plain[k + 1:]
    hs = A.Shape(m, k_log, claims, holo=True)
    assert [e for *_, e in holo if e] == [len(r) for r in A.rounds(hs)] and A.h_round(hs) == sum(1 for *_, e in holo[:k + 1] if e)
    c = A.Coins([[1], [], [2, 3]])
    assert (c.vec(1), c.vec(0), c.vec(2)) == ([1], [], [2, 3])
    sch = RL.schedule_for({"dense_m": m, "k_log": k_log, "nbl": 3, "regions": (claims - 2) // 2}, holo=True)
    assert sch["streams"]["circuit/rep0"]["rounds"] == len(plain) + 1
    assert [r["n"] for r in sch["rounds"] if r["stream"] == "circuit/rep0"][k:k + 2] == [0, 1]


def _dense_fold(nets: dict, ranges: list[dict], k_log: int, alpha: int, lw: list[int], rho: list[int], ts: list[int]) -> list[int]:
    """The hidden slots' share of ``α·A₀ᵀe + B₀ᵀe`` over all 2^k_log columns, ``e = lw ⊗ eq(ρ)``, halved at ``ts`` in turn:
    ``Σ_u eq(ts reversed, u) · v[s + 64 u]`` at ``s`` (`CircuitFold.folded` written out)."""
    e_in = A.eq_table(rho)
    e = [mul(lw[i % 64], e_in[i // 64]) for i in range(1 << k_log)]
    v = [0] * (1 << k_log)
    for r in ranges:
        net, sl = nets[r["net"]], r["slot_log"]
        for q in range(r["pos0"], r["pos0"] + r["count"]):
            for i in range(len(net["a"])):
                g = (q << sl) | i
                for c in net["a"][i]:
                    v[(q << sl) | c] ^= mul(alpha, e[g])
                for c in net["b"][i]:
                    v[(q << sl) | c] ^= e[g]
    eq_r = A.eq_table(ts[::-1])
    out = [0] * 64
    for u, w in enumerate(eq_r):
        for s in range(64):
            out[s] ^= mul(w, v[s + 64 * u])
    return out


def _unit_text(name: str, useful: int, g: random.Random, width: int) -> list[str]:
    rows = []
    for _ in range(useful):
        a = [g.randrange(width) for _ in range(g.randrange(3))]
        b = [g.randrange(width) for _ in range(g.randrange(3))]
        rows.append(" ".join(map(str, [len(a), *a, len(b), *b])))
    return [f"flock-ir-unit/v2 {name}/v0 {useful} {useful - 1} 1 0 1 16 1 1 1 16", *rows, 'LEAVES {"in": []}']


def test_h_is_the_dense_fold_of_the_hidden_slots(tmp_path):
    """On a small circuit file (k_log 10; two hidden circuits of 2^7- and 2^8-bit slots, one public circuit and a mask range
    that are not read), `hidden_circuits` reads the hidden rows as the verifier's parser does and `hidden_fold` (the
    structured fold: one base a circuit, projected and weighted per range) is the dense fold of their slots."""
    g, k_log = random.Random(11), 10
    blocks = {"unit": _unit_text("unit", 100, g, 128), "aux": _unit_text("aux", 200, g, 256), "pub": _unit_text("pub", 90, g, 128)}
    ranges = [{"type": "unit", "net": "unit", "slot_log": 7, "pos0": 4, "count": 4},
              {"type": "unit", "net": "aux", "slot_log": 8, "pos0": 1, "count": 1},
              {"type": "unit", "net": "pub", "slot_log": 7, "pos0": 0, "count": 2},
              {"type": "mask", "slot_log": 7, "pos0": 7, "count": 1}]
    text = ["flock-circuit toy", "META " + json.dumps({"k_log": k_log, "ranges": ranges})]
    for name, lines in blocks.items():
        text += [f"CIRCUIT {name} {len(lines)}", *lines]
    (tmp_path / "circuit.txt").write_text("\n".join(text) + "\n")
    nets, hidden, _meta = RS.hidden_circuits(tmp_path / "circuit.txt", ("unit", "aux"))
    assert sorted(nets) == ["aux", "unit"] and [r["net"] for r in hidden] == ["unit", "aux"]
    assert {n: (v["unit_log"], len(v["a"])) for n, v in nets.items()} == {"unit": (7, 100), "aux": (8, 200)}
    pt = lambda n: [g.getrandbits(128) for _ in range(n)]
    for _ in range(2):
        alpha, lw, rho, ts = g.getrandbits(128), pt(64), pt(k_log - 6), pt(k_log - 6)
        h = RS.hidden_fold(nets, hidden, alpha, lw, rho, ts)
        assert h == _dense_fold(nets, hidden, k_log, alpha, lw, rho, ts) and any(h)
    with pytest.raises(ValueError, match="hidden circuits"):
        RS.hidden_circuits(tmp_path / "circuit.txt", ("unit", "absent"))
    typed = text[:1] + ["META " + json.dumps({"k_log": k_log, "ranges": ranges, "unit_type": "unit", "types": {}})] + text[2:]
    (tmp_path / "typed.txt").write_text("\n".join(typed) + "\n")
    with pytest.raises(ValueError, match="typed statement"):
        RS.hidden_circuits(tmp_path / "typed.txt", ("unit", "aux"))
    wide = list(text)
    wide[text.index(f"CIRCUIT unit {len(blocks['unit'])}") + 2] = "1 128 0"
    (tmp_path / "wide.txt").write_text("\n".join(wide) + "\n")
    with pytest.raises(ValueError, match="outside its 2"):
        RS.hidden_circuits(tmp_path / "wide.txt", ("unit", "aux"))


def _client(addr: str, rec: dict) -> list:
    """The fake prover: the fixture's honest session as Flock's live prover sends it (its Hello without the coin scheme:
    the session ran on seeded coins), each answer kept."""
    host, port = addr.rsplit(":", 1)
    coins = lambda hexes: b"".join(RL.f128_bytes(int(c, 16)) for c in hexes)
    got = []
    with socket.create_connection((host, int(port))) as c, c.makefile("rb") as rf, c.makefile("wb") as wf:
        def send(*req):
            RL.write_frame(wf, RL.encode_req(*req))
            got.append((req[:2] if req[0] == RL.REQ_ROUND else req[:1], RL.decode_resp(RL.read_frame(rf))))
            return got[-1][1]

        hello = json.loads(rec["hello"])
        del hello["coins"]
        lk = rec["link"]
        send(RL.REQ_HELLO, json.dumps(hello))
        send(RL.REQ_COMMIT, bytes.fromhex(lk["root_f"]), [(t, bytes.fromhex(v)) for t, v in lk["root_b"].items()], [])
        send(RL.REQ_LINK, coins(lk["y"]))
        for st in rec["streams"]:
            send(RL.REQ_OPEN, st["stream"], st["parent"] or "", st["label"].encode(), st["at"])
            for rd in st["rounds"]:
                send(RL.REQ_ROUND, st["stream"], bytes.fromhex(rd["msg"]), rd["n"])
        for r in (0, 1):
            send(RL.REQ_PROOF, "circuit", r, (SET / "records-cpu" / f"honest.rep{r}.proof").read_bytes())
        send(RL.REQ_FINISH)
    return got


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    """`rec_session.serve` on loopback, its coin server returning the fixture's recorded coins in the order the prover's
    rounds drew them (z_partial's coinless round draws none), and the fake prover's answers."""
    assert (SET / "stage" / "circuit.txt").is_file(), (
        f"{SET} is not fetched: `uv run research data fetch-fixtures verity/ml/flock/tests/fixtures/hidden-outputs`")
    rec = json.loads((SET / "records-cpu" / "honest.json").read_text())["record"]
    pending = bytearray(b"".join(RL.f128_bytes(int(c, 16)) for c in
                                 [x for p in rec["link"]["points"] for x in p] +
                                 [c for st in rec["streams"] for rd in st["rounds"] for c in rd["coins"]]))

    def rand(n: int) -> bytes:
        out = bytes(pending[:n])
        del pending[:n]
        assert len(out) == n, "a coin the record does not hold"
        return out

    out, q = tmp_path_factory.mktemp("session"), queue.Queue()

    def run():
        try:
            q.put(("done", RS.serve("127.0.0.1:0", out, STATEMENT, SET / "stage" / "circuit.txt", rand=rand,
                                    ready=lambda addr: q.put(("ready", addr)))))
        except BaseException as e:  # noqa: BLE001 - handed to the test, which raises it
            q.put(("error", e))

    th = threading.Thread(target=run)
    th.start()
    kind, addr = q.get()
    if kind == "error":
        raise addr
    got = _client(addr, rec)
    kind, summary = q.get()
    th.join()
    if kind == "error":
        raise summary
    return out, summary, got, rec


def _rep(record: dict, stream: str) -> list[dict]:
    [st] = [s for s in record["streams"] if s["stream"] == stream]
    return st["rounds"]


def test_the_prover_sees_its_own_session_and_the_verifier_h_in_its_own_round(served):
    """The prover's every round is answered with its recorded coins, and ``prover/session.json`` is its session (each
    round's message, coin count and coins), what upstream's replay checks its proofs against. The coin server's record has
    the holo schedule: z_partial's round with a ``c_k`` and no coins, then h's round with a ``c_k`` and r′, on the same
    stream; its h is the hidden circuits' fold at the rep's fold points, which the fixture's coins give."""
    out, summary, got, rec = served
    assert summary["session"] and summary["h_round"] == RL.zp_round(M, K_LOG) + 1
    want = [[int(c, 16) for c in rd["coins"]] for st in rec["streams"] for rd in st["rounds"]]
    assert [r for (req, r) in got if req[0] == RL.REQ_ROUND] == [(RL.RESP_COINS, c) for c in want]
    assert got[-1][1][0] == RL.RESP_VERDICT
    mine = json.loads((out / "prover" / "session.json").read_text())
    for st in rec["streams"]:
        assert [(r["msg"], r["n"], r["coins"]) for r in _rep(mine, st["stream"])] == [(r["msg"], r["n"], r["coins"]) for r in st["rounds"]]
        assert (out / "prover" / f"circuit.rep{st['stream'][-1]}.bin").read_bytes() == (SET / "records-cpu" / f"honest.rep{st['stream'][-1]}.proof").read_bytes()
    public, private = (json.loads((out / f).read_text()) for f in ("public.json", "private.json"))
    assert RL.schedule_of(public) == RL.schedule_for(STATEMENT, holo=True) == json.loads((out / "schedule.json").read_text())
    nets, ranges, _ = RS.hidden_circuits(SET / "stage" / "circuit.txt")
    k = RL.zp_round(M, K_LOG)
    hs = RS.session_h(private, M, K_LOG)
    for r in (0, 1):
        stream = f"circuit/rep{r}"
        rounds = sorted((x for x in public["rounds"] if x["stream"] == stream), key=lambda x: x["k"])
        prv = {p["k"]: p for p in private["rounds"] if p["stream"] == stream}
        assert [(x["n"], x["coins"], len(x["commits"])) for x in rounds[k:k + 2]] == [(0, [], 1), (1, _rep(rec, stream)[k]["coins"], 1)]
        assert prv[k]["elements"] == [G.hex128(x) for x in RL.split_round(bytes.fromhex(_rep(rec, stream)[k]["msg"]), False).free]
        f = A.fold_points(A.Shape(M, K_LOG, 2), A.rep_rounds(rec, stream))
        assert hs[r] == RS.hidden_fold(nets, ranges, f["alpha"], f["lw"], f["rho_in"], f["ts"]) and any(hs[r])
        assert prv[k + 1]["rows"] == RL.message_rows(hs[r])


def _holo_fold(hs: list[list[int]]) -> dict:
    """``InnerFold.lean --hide unit``'s output as the fixture gives it: Lean's whole fold (InnerClaims' dump) less h."""
    return {"m": INNER["m"], "k_log": INNER["k_log"], "extras": INNER["reps"][0]["extras"], "hidden": ["unit"],
            "reps": [{"comb": [G.hex128(G.of_hex(c) ^ x) for c, x in zip(d["comb"], h, strict=True)], "full": d["comb"]}
                     for d, h in zip(INNER["reps"], hs, strict=True)]}


def test_rec_vstage_stages_the_holo_algebra_with_h_read_from_the_session(served):
    """`rec_vstage` on the holo session: the fold points from the coins as the plain session's (z_partial's coinless round
    is no draw), the shape holo, h's round among the message rounds (its row the one its ``c_k`` commits), and per rep the
    holo algebra's residuals zero on the rounds' rows with the public comb; h's row one bit off moves the lincheck alone.
    A plain verifier (no hidden circuits) refuses the session: it has one message round too many."""
    out, _, _, rec = served
    public, private = (json.loads((out / f).read_text()) for f in ("public.json", "private.json"))
    meta, n = S.meta(SET / "stage" / "circuit.txt"), S.instances(SET / "stage" / "pub-4.bin")
    pts = S.points(public, meta, n)
    for r, p in enumerate(pts["reps"]):
        f = A.fold_points(A.Shape(M, K_LOG, 2), A.rep_rounds(rec, f"circuit/rep{r}"))
        assert p["ts"] == [G.hex128(x) for x in f["ts"]] and p["alpha"] == G.hex128(f["alpha"])
    hs = RS.session_h(private, M, K_LOG)
    fold = _holo_fold(hs)
    assert RS.check_fold(fold, private) == {"hidden": ["unit"], "comb_plus_h_is_full": [True, True], "h_nonzero": [True, True]}
    vf = S.verifier(public, fold, pts)
    sh = vf.sh
    assert sh.holo and (sh.m, sh.k_log, sh.claims) == (M, K_LOG, 8)
    mrs = S.message_rounds(public, private, sh)
    hr = A.h_round(sh)
    assert len(mrs) == len(A.rounds(sh)) and [mrs[hr]["rows"][r] for r in (0, 1)] == [RL.message_rows(h)[0] for h in hs]
    S.messages(O.session_program((out / "public.json").read_bytes()), mrs)
    spec = RR.spec(sh)
    nz = lambda o: [i for i in range(len(spec["gf"]["res"])) if any(o[8 * i:8 * i + 8])]
    for r in (0, 1):
        p = A.decode(sh, (out / "prover" / f"circuit.rep{r}.bin").read_bytes())
        acc0 = [0x1234 + r, 0x5678 + r]
        cs = A.consistency(sh, p, vf.reps[r].info)
        rows = [mr["rows"][r] for mr in mrs] + [RR.words(acc0, 64), RR.words([acc0[0] ^ cs[0], acc0[1] ^ cs[1]], 64)]
        assert RR.residual_words(spec, rows, vf.reps[r].v) == [0] * RR.OUT_WORDS
        rows[hr] = [rows[hr][0] ^ 1, *rows[hr][1:]]
        assert nz(RR.residual_words(spec, rows, vf.reps[r].v)) == [1]
    plain = {**fold, "reps": [{"comb": d["full"]} for d in fold["reps"]]}
    del plain["hidden"]
    with pytest.raises(ValueError, match="message rounds"):
        S.message_rounds(public, private, S.verifier(public, plain, pts).sh)
    bad = {**fold, "reps": [{**fold["reps"][0], "comb": [G.hex128(G.of_hex(fold["reps"][0]["comb"][0]) ^ 1), *fold["reps"][0]["comb"][1:]]},
                            fold["reps"][1]]}
    assert RS.check_fold(bad, private)["comb_plus_h_is_full"] == [False, True]


class _Proxy:
    """The proxy's part the relay drives: each round answered with its ``n`` coins (7s), every request kept."""

    def __init__(self):
        self.stopped, self.reqs = None, []

    def handle(self, req: bytes) -> bytes:
        self.reqs.append(RL.decode_req(req))
        return RL.resp_coins([7] * self.reqs[-1][3])

    def stop(self, why: str) -> None:
        self.stopped = self.stopped or why


def test_the_relay_splits_z_partials_round_and_refuses_another():
    """At z_partial's index the relay sends z_partial's frames ending in a squeeze of no coins, then h's 64 elements and the
    prover's own squeeze, computing h between them at the fold points of the coins the prover has had; the prover gets r′.
    A round there that is not z_partial's (64 elements, then a squeeze of one coin) stops the session before the proxy sees
    it, and an h the relay cannot compute stops it before r′ is drawn."""
    k, zp = RL.zp_round(M, K_LOG), [G.of_hex(f"{i:032x}") for i in range(64)]
    prefix = [[0] * 6, [0] * (M - 13)] + [[i + 1] for i in range(k - 2)]
    content = RS.observe(zp) + RS._frame(RL.OP_SQUEEZE, RL.KIND_SCALAR, 1)
    seen = []

    def h_of(pts):
        seen.append(pts)
        return [x ^ 1 for x in zp]

    proxy = _Proxy()
    relay = RS.Relay(proxy, M, K_LOG, h_of)
    relay.coins["circuit/rep0"] = list(prefix)
    assert RL.decode_resp(relay.handle(RL.encode_req(RL.REQ_ROUND, "circuit/rep0", content, 1))) == (RL.RESP_COINS, [7])
    (_, s0, c0, n0), (_, s1, c1, n1) = proxy.reqs
    assert (s0, s1, n0, n1) == ("circuit/rep0", "circuit/rep0", 0, 1)
    assert RL.split_round(c0, False).free == zp and RL.frames(c0)[-1][:3] == (RL.OP_SQUEEZE, RL.KIND_SLICE, 0)
    assert RL.split_round(c1, False).free == [x ^ 1 for x in zp] and c1[-16:] == content[-16:]
    assert seen == [A.fold_points(A.Shape(M, K_LOG, 2), prefix + [[0]])] and relay.coins["circuit/rep0"][-1] == [7]
    assert relay.split["circuit/rep0"]["content"] == content
    for bad, n, why in [(RS.observe(zp[:63]) + content[-16:], 1, "not z_partial's"),
                        (RS.observe(zp) + RS._frame(RL.OP_SQUEEZE, RL.KIND_SLICE, 2), 2, "not z_partial's"),
                        (content, 1, "h: ArithmeticError")]:
        def boom(_):
            raise ArithmeticError("no h")

        proxy = _Proxy()
        relay = RS.Relay(proxy, M, K_LOG, boom)
        relay.coins["circuit/rep0"] = list(prefix)
        resp = RL.decode_resp(relay.handle(RL.encode_req(RL.REQ_ROUND, "circuit/rep0", bad, n)))
        assert resp[0] == RL.RESP_REFUSED and why in resp[1] and why in proxy.stopped
        assert [r[3] for r in proxy.reqs] == ([0] if why.startswith("h") else [])


#: the statement digest every binding round of `_stacked_session` absorbs
STATEMENT_DIGEST = bytes(range(32))


def _stacked_session(proxy, roots: dict, sh: tuple, m_r: int | None, order=("circuit/rep0", "circuit/rep1", "stacked/rep0"),
                     binds=None, prover: bool = False, digests=None):
    """A holo session at ``sh`` and the stacked table's stream at ``m_r``, as the relay would hand them to the proxy: Hello
    (canonical JSON, as the prover writes it) naming the tables of ``roots`` and Commit with their caps, then each stream's
    rounds built from its round tuples (a stream's first round binds its table: the r1cs label, the statement digest, its root; then its
    fresh tops, its other copies of the root, its free message, the ring switch's nonce, its squeeze), its proof and Finish.
    ``binds``: the caps the rounds bind and copy, if not the committed ones; a table with no root in ``roots`` (R) binds its
    own, the first of its binding round's tops. ``prover``: the reps' rounds as the prover sends them to the relay
    (z_partial's round whole, `rec_live.rep_rounds`). ``digests``: by stream, the statement digest its binding round
    absorbs, if not `STATEMENT_DIGEST`."""
    m, k_log, claims = sh
    g, binds = random.Random(5), binds or roots
    hello = {"profile": "fast100", "reps": 2, "flavor": "rs", "tables": list(roots), "rounds": "bytes retained",
             "digests": "sha512", "link": {"sigma": "ab" * 64, "m": RL.PT_LOCAL + m - k_log, "points": 2}}
    out = [proxy.handle(RL.encode_req(RL.REQ_HELLO, RL.canonical(hello))),
           proxy.handle(RL.encode_req(RL.REQ_COMMIT, b"root-f", list(roots.items()), [])),
           proxy.handle(RL.encode_req(RL.REQ_LINK, bytes(32)))]
    b = lambda p: RS._frame(RL.OP_BYTES, 0, len(p), p)
    for s in order:
        table = s.split("/")[0]
        rounds = RL.stacked_rounds(m_r) if table == RL.STACKED else (RL.rep_rounds if prover else RL.holo_rounds)(m, k_log, claims)
        label = f"flock-circuit/{table}/fast100/rep{s[-1]}" if table == RL.STACKED else f"flock-circuit/fast100/rep{s[-1]}"
        out.append(proxy.handle(RL.encode_req(RL.REQ_OPEN, s, "", label.encode(), 0)))
        for k, (n, tops, level0, e) in enumerate(rounds):
            own = k == 0 and table not in roots
            c = RS._frame(RL.OP_LABEL, 0, len(RL.LABEL_R1CS), RL.LABEL_R1CS) + b((digests or {}).get(s, STATEMENT_DIGEST)) + b(binds[table]) if k == 0 else b""
            c += b"".join(b(g.randbytes(64 * t)) for t in tops[own:]) + b(binds[table]) * (level0 - (k == 0 and not own))
            c += (RS.observe([g.getrandbits(128) for _ in range(e)]) if e else b"") + (b(g.randbytes(8)) if e == 128 else b"")
            out.append(proxy.handle(RL.encode_req(RL.REQ_ROUND, s, c + RS._frame(RL.OP_SQUEEZE, RL.KIND_SLICE, n), n)))
        out.append(proxy.handle(RL.encode_req(RL.REQ_PROOF, table, int(s[-1]), f"{s} proof".encode())))
    out.append(proxy.handle(RL.encode_req(RL.REQ_FINISH)))
    return [RL.decode_resp(r) for r in out]


def test_the_stacked_stream_is_held_to_the_schedule_the_opening_only_prover_runs(tmp_path):
    """`schedule_for(..., stacked=m_R)` is rec-stage-2's interface (``flock-circuit stacked``): Hello and Commit are the
    statement's alone (the circuit table's), and ``stacked/rep0`` is ``[(7, [128], 0, 128)] + rep_rounds(m_R, 6, 2)[m_R +
    1:]`` after both reps: its binding round commits R's 2^7-node cap as a salted top, and Ligerito's copy of it later is a
    level-0 copy. A session sending that stream as the subcommand does (binding round with the r1cs label, the statement digest
    and R's cap, s_hat_v and a nonce) is held to it whole: the verifier's record has the prover's Hello, byte for byte, and R's
    level 0 is that round's top. Its binding round must commit a cap no other stream bound, it must follow the circuit's
    reps, and a Hello naming R is refused."""
    sh, m_r = (22, 10, 2), 22
    statement = {"dense_m": sh[0], "k_log": sh[1], "nbl": sh[0] - sh[1], "regions": 0}
    plain, sch = RL.schedule_for(statement, holo=True), RL.schedule_for(statement, holo=True, stacked=23)
    rs = [(7, [128], 0, 128)] + RL.rep_rounds(23, 6, 2)[24:]
    assert sch["hello"] == plain["hello"] and sch["hello"]["tables"] == ["circuit"] and sch["roots"] == plain["roots"]
    assert sch["streams"] == {**plain["streams"], "stacked/rep0": {"parent": "", "at": 0, "rounds": len(rs)}}
    assert sch["rounds"][:len(plain["rounds"])] == plain["rounds"]
    assert [(r["n"], r["tops"], r["level0"], r["commits"]) for r in sch["rounds"][len(plain["rounds"]):]] == \
        [(n, t, l0, int(e > 0)) for n, t, l0, e in rs]
    with pytest.raises(ValueError, match="22 to 35"):
        RL.stacked_rounds(21)

    sch = RL.schedule_for(statement, holo=True, stacked=m_r)
    roots = {"circuit": random.Random(1).randbytes(64 << 7)}
    binds = {**roots, "stacked": random.Random(2).randbytes(64 << 7)}
    registration, bound, reg = _registered_top(_toy_circuit(tmp_path, 8)[0].parent, binds["stacked"], 6)
    coins = RL.CoinServer(random.Random(3).randbytes)
    salts = random.Random(4)
    proxy = RL.Proxy(coins, lambda: salts.randbytes(192), schedule=sch, registration=registration, registered={RL.STACKED: reg},
                     bound=bound)
    resp = _stacked_session(proxy, roots, sh, m_r, binds=binds)
    assert all(r[0] != RL.RESP_REFUSED for r in resp) and json.loads(resp[-1][1])["accepted"] is None, proxy.stopped
    rec = coins.record()
    assert RL.schedule_of(rec) == sch and list(rec["link"]["roots"]) == ["circuit"] and rec["hello"]["tables"] == ["circuit"]
    first = next(r for r in rec["rounds"] if r["stream"] == "stacked/rep0")
    assert (first["k"], first["n"], first["level0"], len(first["commits"]), [len(t) for t in first["tops"]]) == (0, 7, 0, 1, [128])
    assert first["tops"][0] == reg["top"]
    assert sum(r["level0"] for r in rec["rounds"] if r["stream"] == "stacked/rep0") == 1
    sess = proxy.write(tmp_path)
    record = json.loads((sess / "session.json").read_text())
    assert record["hello"] == proxy.hello == RL.canonical(json.loads(record["hello"]))
    assert json.loads(record["hello"])["tables"] == ["circuit"] and list(record["link"]["root_b"]) == ["circuit"]
    st = {s["stream"]: s for s in record["streams"]}
    assert st["stacked/rep0"]["root_sha512"] == RL._dg(binds["stacked"])
    assert (sess / "stacked.rep0.bin").read_bytes() == b"stacked/rep0 proof"
    public, private = (json.loads((tmp_path / f).read_text()) for f in ("public.json", "private.json"))
    r_tops = V.tops(public, private, "stacked/rep0", RL.STACKED)
    assert b"".join(r_tops[0].nodes) == binds["stacked"] and list(r_tops[0].bcs) == [bytes.fromhex(x) for x in first["tops"][0]]
    assert len(r_tops) == sum(len(r["tops"]) for r in rec["rounds"] if r["stream"] == "stacked/rep0")
    assert b"".join(V.tops(public, private, "circuit/rep0")[0].nodes) == roots["circuit"]
    assert json.loads((tmp_path / RL.REGISTRATION).read_text()) == registration
    assert RL.top_entry(registration["program"], RL.STACKED, [bytes.fromhex(b) for b in first["tops"][0]]) == \
        RL.registered_entry(registration, RL.STACKED)
    V.registered(public, registration, bound)
    other = RL.register_top(registration["program"], RL.STACKED, binds["stacked"], lambda g=random.Random(9): g.randbytes(192))
    (tmp_path / RL.REGISTRATION).write_text(json.dumps(IR.registration_record(*_partition(), [other["entry"]])))
    with pytest.raises(ValueError, match="a registration other than the one this session was held to"):
        proxy.write(tmp_path)

    for kw, why in [({"binds": {**roots, "stacked": roots["circuit"]}}, "stacked/rep0: the binding round's cap is not stacked's registered top"),
                    ({"binds": {**roots, "stacked": random.Random(7).randbytes(64 << 7)}},
                     "stacked/rep0: the binding round's cap is not stacked's registered top"),
                    ({"binds": binds, "digests": {"stacked/rep0": bytes(32)}},
                     "stacked/rep0: the binding round's statement digest is not the circuit reps'"),
                    ({"binds": binds, "digests": {"circuit/rep1": bytes(32)}},
                     "circuit/rep1: the binding round's statement digest is not the other reps'"),
                    ({"binds": binds, "order": ("circuit/rep0", "stacked/rep0", "circuit/rep1")}, "FIREWALL-REFUSED: round "),
                    ({"roots": binds}, "FIREWALL-REFUSED: Hello")]:
        proxy = RL.Proxy(RL.CoinServer(random.Random(3).randbytes), schedule=sch, registration=registration,
                         registered={RL.STACKED: reg}, bound=bound)
        _stacked_session(proxy, **{"roots": roots, "sh": sh, "m_r": m_r, **kw})
        assert why in (proxy.stopped or ""), (why, proxy.stopped)


def test_rs_top_is_registered_at_staging_and_meta_binds_its_registration(tmp_path):
    """The hole the registration closes: R is built from the hidden circuit after the coins that fix its opening point, and
    its root reaches the coin server only at ``stacked/rep0``'s binding round. So R's top is registered with the statement
    at staging (`rec_live.register_top`, `inner_registered.bind`): one more registered value, ``stacked``, of the circuit
    registration, the ``verity/registration/v2`` record one-stage ``registration.record`` writes over the partition object
    META names (one-stage's ``stratified`` law, the statement's scheme, no roots, window ``stand-in``), beside the other
    values registered at staging (a description's): 128 rows of 64 words, one a node, each under its own salt, which only
    the prover's file holds. META's ``registration`` is the record's digest, so the circuit's SHA-512 (both files' headers)
    and the statement digest bind it, and the same circuit and salts give the same record. The firewall refuses a
    registration whose digest is not META's, a binding round when R has no registered top, or binds a cap that is not it;
    a top is registered only before Hello and before any coin, once, only with an opening that opens it and whose top makes
    the registration's entry, and never for a table whose root Commit carries. The schedule names no registered value.
    `rec_vstar.registered` refuses a record whose digest is not META's, rebuilds the entry from the binding round's top and
    refuses a record whose entry is another, and one that has none."""
    sh, m_r = (22, 10, 2), 22
    statement = {"dense_m": sh[0], "k_log": sh[1], "nbl": sh[0] - sh[1], "regions": 0}
    sch = RL.schedule_for(statement, holo=True, stacked=m_r)
    roots = {"circuit": random.Random(1).randbytes(64 << 7)}
    binds = {**roots, "stacked": random.Random(2).randbytes(64 << 7)}
    program, part = _partition()
    stage = _toy_circuit(tmp_path, 8)[0].parent
    desc = RG.Registered.commit(part["program"], "prog", [[1, 2, 3]])
    with_desc = IR.bind(stage, program, part, [desc.entry()])
    assert IR.meta(stage)[IR.REGISTRATION_KEY] == R.digest(with_desc) and IR.bound(stage) == with_desc
    with pytest.raises(ValueError, match="carries no stacked entry: R was not registered at staging"):
        RS.staged_stacked(stage)
    registration, bound, reg = _registered_top(stage, binds["stacked"], 6)
    meta, text = IR.meta(stage), (stage / "circuit.txt").read_text()
    entry = RL.registered_entry(registration, RL.STACKED)
    assert bound == meta[IR.REGISTRATION_KEY] == R.digest(registration)
    assert registration == IR.registration_record(program, part, [desc.entry(), entry])
    assert (registration["program"], registration["partition"]) == (meta["program_sha512"], meta["partition"]["digest"])
    assert (registration["scheme"], registration["roots"], registration["window"]) == (FC.SCHEME, [], IR.WINDOW)
    for f in ("pub-1.bin", "inst-1.bin"):
        head = json.loads((stage / f).read_bytes().split(b"\n", 1)[0])
        assert head["circuit_sha512"] == FC.digest(text) and head["units"]["classes"] == [FC.class_digest(text)]
    again = _toy_circuit(tmp_path / "again", 8)[0].parent
    IR.bind(again, program, part, [desc.entry()])
    assert _registered_top(again, binds["stacked"], 6) == (registration, bound, reg), "the circuit and the salts alone"
    dom = RG.domain(registration["program"], RL.STACKED, RL.ROW_WORDS, 128)
    assert registration["format"] == R.FORMAT_V2 and entry == \
        {"value": "stacked", "root": entry["root"], "leaves": 128, "words": 64, "domain": dom.domain_id.hex()}
    assert reg["top"] == [b.hex() for b in RL.commit_top(binds["stacked"], [bytes.fromhex(s) for s in reg["salts"]])]
    assert len(set(reg["salts"])) == 128 and "registered" not in json.dumps(RL.check_schedule(sch))
    assert entry["root"] not in json.dumps(sch)
    assert not any(x in json.dumps(registration) + text for x in reg["salts"]), "the salts are the prover's alone"
    with pytest.raises(ValueError, match="a value is registered once"):
        IR.bind(stage, program, part, [entry])
    with pytest.raises(ValueError, match="META binds program"):
        IR.bind(stage, *(lambda pq: (pq[0], P.partition(*pq)))(_private(4)), [desc.entry()])
    (stage / IR.REGISTRATION).write_text(json.dumps(with_desc))
    for read in (IR.bound, RS.staged_stacked):
        with pytest.raises(ValueError, match="is not the registration META binds"):
            read(stage)
    (stage / IR.REGISTRATION).write_text(json.dumps(registration))

    proxy = RL.Proxy(RL.CoinServer(random.Random(3).randbytes), schedule=sch)
    _stacked_session(proxy, roots, sh, m_r, binds=binds)
    assert proxy.stopped == "rec proxy: FIREWALL-REFUSED: stacked/rep0: stacked has no root at Commit and no registered top"

    proxy = RL.Proxy(RL.CoinServer(random.Random(3).randbytes), schedule=sch, registration=registration, bound=bound)
    proxy.register(RL.STACKED, reg)
    with pytest.raises(ValueError, match="registered twice"):
        proxy.register(RL.STACKED, reg)
    with pytest.raises(ValueError, match="registered twice, or a table whose root is committed at Commit"):
        RL.Proxy(RL.CoinServer(), schedule=sch, registration=registration, registered={"circuit": reg}, bound=bound)
    with pytest.raises(ValueError, match="does not open it"):
        RL.Proxy(RL.CoinServer(), schedule=sch, registration=registration, bound=bound,
                 registered={RL.STACKED: {**reg, "salts": reg["salts"][::-1]}})
    other = RL.register_top(registration["program"], RL.STACKED, binds["stacked"], lambda g=random.Random(8): g.randbytes(192))
    forged = IR.registration_record(program, part, [desc.entry(), other["entry"]])
    v1 = {**registration, "format": R.FORMAT}
    not_meta = "the registration is not the one the statement's META binds"
    for bad, dg, why in [(with_desc, R.digest(with_desc), "the private circuit's registration (verity/registration/v2) names no stacked"),
                         (v1, R.digest(v1), "names no stacked"), (None, bound, not_meta), (registration, None, not_meta),
                         (forged, bound, not_meta), (forged, R.digest(forged), "the registered top is not the registration's stacked")]:
        with pytest.raises(ValueError, match=re.escape(why)):
            RL.Proxy(RL.CoinServer(), schedule=sch, registration=bad, registered={RL.STACKED: reg}, bound=dg)
    late = RL.Proxy(RL.CoinServer(random.Random(3).randbytes), schedule=sch, registration=registration, bound=bound)
    _stacked_session(late, roots, sh, m_r, binds=binds, order=())
    with pytest.raises(ValueError, match="registered before the session, never after Hello or a coin"):
        late.register(RL.STACKED, reg)
    early = RL.CoinServer(random.Random(3).randbytes)
    early.commit_coins(sch, lambda: bytes(192))
    with pytest.raises(ValueError, match="never after Hello or a coin"):
        RL.Proxy(early, schedule=sch, registration=registration, registered={RL.STACKED: reg}, bound=bound)

    proxy = RL.Proxy(coins := RL.CoinServer(random.Random(3).randbytes), schedule=sch, registration=registration,
                     registered={RL.STACKED: reg}, bound=bound)
    _stacked_session(proxy, roots, sh, m_r, binds=binds)
    assert proxy.stopped is None
    public = coins.record()
    first = next(r for r in public["rounds"] if r["stream"] == "stacked/rep0" and r["k"] == 0)
    assert first["tops"][0] == reg["top"]
    assert not any(x in json.dumps(public) for x in reg["salts"])
    V.registered(public, registration, bound)
    moved = {**registration, "registered": [desc.entry(), {**entry, "domain": desc.entry()["domain"]}]}
    not_meta = "the record is not the circuit registration META binds"
    for bad, dg, why in [(forged, R.digest(forged), "stacked/rep0's binding round's top is not stacked's registered top"),
                         (moved, R.digest(moved), "stacked/rep0's binding round's top is not stacked's registered top"),
                         (forged, bound, not_meta), (registration, None, not_meta), (None, bound, not_meta),
                         (with_desc, R.digest(with_desc), "stacked has no root at Commit and no registered top"),
                         (v1, R.digest(v1), "stacked has no root at Commit and no registered top")]:
        with pytest.raises(ValueError, match=re.escape(f"FIREWALL-REFUSED: registration: {why}")):
            V.registered(public, bad, dg)


def test_rs_shape_is_metas_as_staging_commits_r(tmp_path):
    """The registered entry is a cap of 128 nodes, which fixes neither R's m nor its claims, so staging also writes R's
    opening-only ``Shape(m, 6, claims)`` into META (`rec_session.STACKED_SHAPE_KEY`), which the circuit's SHA-512 and the
    statement digest bind; `serve` holds the stacked stream's schedule to it (`stacked_shape`) and refuses a META naming
    none, another m than the bucket's, or other claims than the two reps'."""
    sp = SP.Shape(9, 7, 2, 0)
    stage = _toy_circuit(tmp_path, 8)[0].parent
    with pytest.raises(ValueError, match="META's stacked is None, not R's shape"):
        RS.stacked_shape(stage, sp)
    for bad in ({"m": SP.table_m(sp) + 1, "claims": RL.REPS}, {"m": SP.table_m(sp), "claims": RL.REPS + 1}):
        IR.bind_meta(stage, RS.STACKED_SHAPE_KEY, bad)
        with pytest.raises(ValueError, match="not R's shape"):
            RS.stacked_shape(stage, sp)
    IR.bind_meta(stage, RS.STACKED_SHAPE_KEY, {"m": SP.table_m(sp), "claims": RL.REPS})
    assert RS.stacked_shape(stage, sp) == SP.table_m(sp) == 22
    text = (stage / "circuit.txt").read_text()
    head = json.loads((stage / "pub-1.bin").read_bytes().split(b"\n", 1)[0])
    assert head["circuit_sha512"] == FC.digest(text), "META's shape is in the circuit the statement digest covers"


def test_serve_refuses_to_start_without_the_stacked_entry_meta_binds(tmp_path):
    """`rec_session.serve` with the stacked table reads R's registration from the staged statement beside its circuit
    (`staged_stacked`), and refuses before it listens when META binds no circuit registration, when the record carries
    no ``stacked`` entry, when the staged record's digest is not META's, when the prover holds no opening of the entry, or
    when META names no R shape or another than the bucket's (`stacked_shape`)."""
    circuit, _ = _toy_circuit(tmp_path, 8)
    stage, out = circuit.parent, tmp_path / "out"
    statement = {"dense_m": 22, "k_log": 10, "nbl": 12, "regions": 0}

    def serve():
        RS.serve("127.0.0.1:0", out, statement, circuit, sparse=SP.Shape(9, 7, 2, 0), stacked=Path("flock-circuit"),
                 ready=lambda addr: pytest.fail(f"served on {addr}"))

    with pytest.raises(ValueError, match="META binds no circuit registration"):
        serve()
    program, part = _partition()
    IR.bind(stage, program, part, [RG.Registered.commit(part["program"], "prog", [[1, 2, 3]]).entry()])
    with pytest.raises(ValueError, match="carries no stacked entry"):
        serve()
    registration, _, _ = _registered_top(stage, random.Random(2).randbytes(64 << 7), 6)
    (stage / IR.REGISTRATION).write_text(json.dumps({**registration, "window": {"kind": "stand-in", "of": "another"}}))
    with pytest.raises(ValueError, match="is not the registration META binds"):
        serve()
    (stage / IR.REGISTRATION).write_text(json.dumps(registration))
    with pytest.raises(ValueError, match="META's stacked is None, not R's shape"):
        serve()
    private = (stage / IR.PRIVATE).read_bytes()
    (stage / IR.PRIVATE).unlink()
    with pytest.raises(ValueError, match="the prover holds no opening of the registered value stacked"):
        serve()
    (stage / IR.PRIVATE).write_bytes(private)
    IR.bind_meta(stage, RS.STACKED_SHAPE_KEY, {"m": 23, "claims": RL.REPS})
    with pytest.raises(ValueError, match=r"META's stacked is \{'claims': 2, 'm': 23\}, not R's shape \{'m': 22, 'claims': 2\}"):
        serve()
    assert not out.exists()

def _toy_circuit(d: Path, seed: int) -> tuple[Path, dict]:
    """A statement staged in ``d/stage-<seed>``: its circuit file at k_log 10 whose one hidden circuit ``unit`` (100 rows of
    2^7-bit slots) has a range of 3 slots at 4, META naming `_partition`'s program and partition object, and the instance
    files' headers (what `inner_registered.bind` rewrites)."""
    lines = _unit_text("unit", 100, random.Random(seed), 128)
    rng = {"type": "unit", "net": "unit", "slot_log": 7, "pos0": 4, "count": 3}
    part = _partition()[1]
    meta = {"k_log": 10, "ranges": [rng], "program_sha512": part["program"], "partition": {"rule": P.FORMAT, "digest": P.digest(part)}}
    text = "\n".join(["flock-circuit toy", "META " + json.dumps(meta), f"CIRCUIT unit {len(lines)}", *lines]) + "\n"
    stage = d / f"stage-{seed}"
    stage.mkdir(parents=True)
    (stage / "circuit.txt").write_text(text)
    head = {"circuit_sha512": FC.digest(text), "instances": 1, "units": {"classes": [FC.class_digest(text)]}}
    (stage / "inst-1.bin").write_bytes(json.dumps(dict(head, rows=True)).encode() + b"\n")
    (stage / "pub-1.bin").write_bytes(FC.public_file(head, b""))
    return stage / "circuit.txt", rng


def _sparse_relay(circuit: Path, phase_of=lambda ph: ph, stacked: dict | None = None):
    """The relay's sparse phase on ``circuit``'s hidden circuit; with ``stacked`` (``bin``, ``dir``, ``forge``), R staged
    with a copy of ``circuit``'s statement in ``dir/stage`` (`rec_session.stage_stacked`, ``forge`` ``cell`` at staging) and
    read back as `rec_session.serve` reads it (`staged_stacked`, R.bin and ROOT.json copied into ``dir``), then opened."""
    sp, statement = SP.Shape(9, 7, 2, 0), {"dense_m": 22, "k_log": 10, "nbl": 12, "regions": 0}
    nets, ranges, _ = RS.hidden_circuits(circuit)
    phase = phase_of((sp, SP.entries(sp, nets["unit"]["a"], nets["unit"]["b"]), ranges[0]))
    sch = RL.schedule_for(statement, holo=True, sparse=sp, stacked=stacked and SP.table_m(sp))
    salts = random.Random(4)
    registration, bound, registered, st = None, None, {}, None
    if stacked:
        d = Path(stacked["dir"])
        stage = d / "stage"
        shutil.copytree(circuit.parent, stage)
        forge = stacked.get("forge")
        RS.stage_stacked(stage, stacked["bin"], sp, *_partition(), "cell" if forge == "cell" else None, lambda: salts.randbytes(192))
        assert IR.meta(stage)[RS.STACKED_SHAPE_KEY] == {"m": SP.table_m(sp), "claims": RL.REPS}
        assert RS.stacked_shape(stage, sp) == SP.table_m(sp)
        registration, bound, registered[RL.STACKED] = RS.staged_stacked(stage)
        for f in (RS.STACKED_TABLE, RS.STACKED_ROOT):
            shutil.copy(stage / f, d / f)
        st = {"bin": stacked["bin"], "dir": d, "forge": forge, "raw": (d / RS.STACKED_TABLE).read_bytes(), "stage": stage}
    draws, coin_rng = [], random.Random(3)

    def rand(n: int) -> bytes:
        draws.append(dict(registered))
        return coin_rng.randbytes(n)

    proxy = RL.Proxy(RL.CoinServer(rand), lambda: salts.randbytes(192), schedule=sch, registration=registration,
                     registered=registered, bound=bound)
    proxy.draws = draws
    relay = RS.Relay(proxy, 22, 10, lambda p: RS.hidden_fold(nets, ranges, p["alpha"], p["lw"], p["rho_in"], p["ts"]), phase, 2, st)
    resp = _stacked_session(relay, {"circuit": random.Random(1).randbytes(64 << 7)}, (22, 10, 2), None,
                            order=("circuit/rep0", "circuit/rep1"), prover=True)
    return relay, sch, resp


@pytest.fixture(scope="module")
def sparse_session(tmp_path_factory):
    d = tmp_path_factory.mktemp("sparse")
    circuit, rng = _toy_circuit(d, 8)
    relay, sch, resp = _sparse_relay(circuit)
    relay.proxy.write(d / "inner")
    public, private = (json.loads((d / "inner" / f).read_text()) for f in ("public.json", "private.json"))
    vf = SimpleNamespace(sh=A.Shape(22, 10, 2, holo=True), reps=[SimpleNamespace(rounds=S.rep_coins(public, r)) for r in (0, 1)])
    return SimpleNamespace(d=d, circuit=circuit, rng=rng, relay=relay, sch=sch, resp=resp, public=public, private=private, vf=vf)


def test_the_relay_runs_each_reps_sparse_phase_after_both_reps_on_the_coin_servers_coins(sparse_session):
    """At the prover's Finish the relay opens each rep's sparse stream as its child at its end and runs `rec_sparse.prove`
    there (h the rep's, entries the hidden circuit's), each message a round of its own on the coin server's coins: the
    record is `schedule_for(..., sparse=)`'s, both reps' rounds before either sparse stream's; `rec_vstage.sparse_reps`
    reads back from the public record and the prover's rows the coins and messages the relay ran (v2's coinless round
    skipped), which the reference accepts; the prover's own record holds only its reps. Entries that are not h's circuit
    stop the session at H1, and the prover's Finish is answered refused."""
    ss = sparse_session
    relay, proxy = ss.relay, ss.relay.proxy
    assert proxy.stopped is None and json.loads(ss.resp[-1][1])["accepted"] is None
    rec = proxy.coins.record()
    sp = relay.sparse[0]
    assert RL.schedule_of(rec) == ss.sch and rec["streams"]["sparse/rep1"] == \
        {"parent": "circuit/rep1", "at": len(RL.holo_rounds(22, 10, 2)), "rounds": len(RL.sparse_rounds(sp))}
    order = ["circuit/rep0", "circuit/rep1", "sparse/rep0", "sparse/rep1"]
    assert [r["stream"] for r in rec["rounds"]] == sorted((r["stream"] for r in rec["rounds"]), key=order.index)
    got = S.sparse_reps(ss.public, ss.private, ss.vf, sp, ss.rng)
    assert got == [relay.sparse_out[r] for r in (0, 1)]
    for c, m in got:
        assert SP.sparse_ref(sp, SP.rows(sp, m, SP.verifier_row(sp, c))) == [0] * SP.out_words(sp)
    assert [s["stream"] for s in relay.prover_record()["streams"]] == ["circuit/rep0", "circuit/rep1"]

    def flip(ph):
        ents = list(ph[1])
        i = next(i for i, e in enumerate(ents) if e[2])
        ents[i] = (*ents[i][:3], ents[i][3] ^ 1)
        return (ph[0], ents, ph[2])

    relay, _, resp = _sparse_relay(ss.circuit, flip)
    assert "sparse phase: AssertionError: H1" in relay.proxy.stopped and json.loads(resp[-1][1])["accepted"] is False


@pytest.mark.slow
def test_rec_vstage_stages_the_relays_sparse_phase_as_rec_sparse(sparse_session, tmp_path):
    """`rec_vstage.stage_sparse` stages the relay's sparse phase as RecSparse over both reps, its message ports the rows the
    firewall committed (h the algebra's h round's), and both instances hold. Building RecSparse's statement is the cost."""
    ss = sparse_session
    program = O.session_program((ss.d / "inner" / "public.json").read_bytes())
    regs, own = S.messages(program, S.message_rounds(ss.public, ss.private, ss.vf.sh))
    st = S.stage_sparse(ss.public, ss.private, ss.vf, program, regs, own, ss.relay.sparse[0], ss.rng, tmp_path, evaluate=False)
    assert st["holds"] == [True, True] and st["nonzero_residuals"] == [[], []]


BIN = os.environ.get("FLOCK_CIRCUIT", "")


@pytest.fixture(scope="module")
def stacked_session(tmp_path_factory):
    if not (BIN and Path(BIN).exists()):
        pytest.skip("FLOCK_CIRCUIT names no flock-circuit build")
    d = tmp_path_factory.mktemp("stacked")
    circuit, rng = _toy_circuit(d, 8)
    relay, sch, resp = _sparse_relay(circuit, stacked={"bin": BIN, "dir": d / "stacked"})
    return SimpleNamespace(d=d, circuit=circuit, rng=rng, relay=relay, sch=sch, resp=resp, read=_opening(d / "inner", relay, rng))


def _opening(d: Path, relay, rng) -> SimpleNamespace:
    """The session written to ``d``, read as the verifier reads R's opening (`rec_vstage.stacked_reads`), its opening-only
    residuals at the verifier's points and the sparse phase's claims, and `rec_vstar`'s check of R's openings."""
    relay.proxy.write(d)
    public, private = (json.loads((d / f).read_text()) for f in ("public.json", "private.json"))
    sp = relay.sparse[0]
    vf = SimpleNamespace(sh=A.Shape(22, 10, 2, holo=True), reps=[SimpleNamespace(rounds=S.rep_coins(public, r)) for r in (0, 1)])
    sr = S.stacked_reads(d, public, private, vf, sp, rng)
    rounds = [[G.of_hex(c) for c in r["coins"]] for r in public["rounds"] if r["stream"] == "stacked/rep0"]
    pts = [SP.registered_claim(sp, c, r) for r, (c, _) in enumerate(sr.reps)]
    proof = V.load(d, RL.STACKED)[2][0]
    res = A.residuals(sr.sh, A.decode(sr.sh, proof), rounds, [], pts, claims=[m["claim"] for _, m in sr.reps])[0]
    return SimpleNamespace(public=public, private=private, vf=vf, sr=sr, residuals=[i for i, x in enumerate(res) if x],
                           vstar=V.check(public, private, {0: proof}, RL.STACKED))


def test_the_relay_commits_r_and_opens_it_at_the_sparse_claims_in_the_same_session(stacked_session, tmp_path):
    """With ``flock-circuit stacked`` (P5) R (`rec_sparse.table` of the hidden circuit's entries, ``--root-out``) is
    registered at staging (`rec_session.stage_stacked`: its top salted once, the entry ``stacked`` of the circuit
    registration whose digest META binds, its salts in the prover's file alone, and read back by `staged_stacked`), the relay
    passes the prover's Hello and Commit on as they are, and once both reps' sparse streams have run opens R at their
    registered claims (P.json) on a connection of its own into the same session, answering its Finish itself: the record is
    `schedule_for(..., sparse=, stacked=table_m)`'s, R's stream last, its binding round absorbing the circuit reps'
    statement digest (``--statement-digest``, the firewall's) and committing ROOT.json's cap as the
    registered top, ``b || c`` for ``b || c`` (`rec_vstar.registered` accepts it); `rec_vstar` accepts R's openings, and the
    opening-only model's residuals at the verifier's points and the sparse phase's claims are zero; the verifier's record
    and the prover's have one Hello, the prover's bytes, and the circuit table's Commit. R registered with cell 0 flipped
    fails both claims' ring switches; rep 0's point one bit off in P.json (in ``x[1:]``, which the prover's ring switch
    reads) fails rep 0's and Ligerito's c0 and c1, whose batched sums are at the verifier's points: each session runs to its
    Finish. R's cell 0 flipped after its registration is refused at R's binding round, before any of R's coins; another
    circuit's R has another registered top. No salt of R's top is in a file the verifier reads: the staged statement's
    circuit, public file and circuit registration, the session's record, its public coins and its registration copy."""
    ss = stacked_session
    proxy = ss.relay.proxy
    assert proxy.stopped is None and json.loads(ss.resp[-1][1])["accepted"] is None
    rec = proxy.coins.record()
    order = ["circuit/rep0", "circuit/rep1", "sparse/rep0", "sparse/rep1", "stacked/rep0"]
    assert RL.schedule_of(rec) == ss.sch and ss.sch["streams"]["stacked/rep0"]["rounds"] == len(RL.stacked_rounds(22))
    assert [r["stream"] for r in rec["rounds"]] == sorted((r["stream"] for r in rec["rounds"]), key=order.index)
    assert proxy.streams["stacked/rep0"]["cap"].hex() == json.loads((ss.d / "stacked" / "ROOT.json").read_text())["cap"]
    assert [len(t) for t in next(r for r in rec["rounds"] if r["stream"] == "stacked/rep0")["tops"]] == [128]
    pr, vr = ss.relay.prover_record(), proxy.session_record()
    assert vr["hello"] == pr["hello"] == RL.canonical(json.loads(pr["hello"])) and list(rec["link"]["roots"]) == ["circuit"]
    assert json.loads(pr["hello"])["tables"] == ["circuit"] and list(pr["link"]["root_b"]) == list(vr["link"]["root_b"]) == ["circuit"]
    assert pr["config"]["tables"] == [{"name": "circuit", "binding": pr["config"]["tables"][0]["binding"]}]
    assert ss.read.vstar["accepted"] and ss.read.residuals == []
    reg = proxy.registered[RL.STACKED]
    assert proxy.statement_digest == STATEMENT_DIGEST and proxy.draws and all(d == {RL.STACKED: reg} for d in proxy.draws)
    assert "".join(reg["nodes"]) == json.loads((ss.d / "stacked" / "ROOT.json").read_text())["cap"]
    assert next(r for r in rec["rounds"] if r["stream"] == "stacked/rep0")["tops"][0] == reg["top"]
    assert (ss.d / "stacked" / "R.bin").read_bytes() == SP.table(*ss.relay.sparse[:2])
    stage = ss.relay.stacked["stage"]
    registration = json.loads((ss.d / "inner" / RL.REGISTRATION).read_text())
    entry = RL.registered_entry(registration, RL.STACKED)
    assert registration == proxy.registration == IR.bound(stage) and proxy.bound == IR.meta(stage)[IR.REGISTRATION_KEY]
    assert entry == RL.top_entry(registration["program"], RL.STACKED, [bytes.fromhex(b) for b in reg["top"]])
    assert registration["format"] == R.FORMAT_V2 and (entry["leaves"], entry["words"]) == (128, 64)
    V.registered(ss.read.public, registration, proxy.bound)
    assert IR.private(stage, RL.STACKED)["salts"] == reg["salts"]
    visible = [stage / "circuit.txt", stage / "pub-1.bin", stage / IR.REGISTRATION, ss.d / "inner" / "public.json",
               ss.d / "inner" / RL.REGISTRATION, *sorted((ss.d / "inner" / "session").iterdir())]
    for f in visible:
        b = f.read_bytes()
        assert not any(x.encode() in b or bytes.fromhex(x) in b for x in reg["salts"]), f"{f} holds a salt of R's top"
    for forge, bad in [("cell", [0, 1]), ("point", [0, 2, 3])]:
        relay, _, resp = _sparse_relay(ss.circuit, stacked={"bin": BIN, "dir": tmp_path / forge, "forge": forge})
        assert relay.proxy.stopped is None and json.loads(resp[-1][1])["accepted"] is None, relay.proxy.stopped
        got = _opening(tmp_path / forge / "inner", relay, ss.rng)
        assert got.vstar["accepted"] and got.residuals == bad, (forge, got.residuals)
    relay, _, resp = _sparse_relay(ss.circuit, stacked={"bin": BIN, "dir": tmp_path / "late", "forge": "late"})
    assert relay.proxy.stopped == "rec proxy: FIREWALL-REFUSED: stacked/rep0: the binding round's cap is not stacked's registered top"
    assert json.loads(resp[-1][1]) == {"accepted": False, "why": relay.proxy.stopped}
    assert not any(r["stream"] == "stacked/rep0" for r in relay.proxy.coins.rounds)
    other = _toy_circuit(tmp_path, 9)[0].parent
    other = RS.stage_stacked(other, BIN, ss.relay.sparse[0], *_partition())
    assert other["nodes"] != reg["nodes"] and other["entry"]["root"] != entry["root"]


def test_rs_opening_registers_its_messages_in_the_verifiers_round_order(stacked_session):
    """`rec_vstage.opening_messages`: port ``m<i>`` reads ``rec-opening-c<i>``, round i of ``Flock/RecOpening.lean``'s
    ``rounds`` (each claim's ring switch, then Ligerito's), which is the i-th round of ``stacked/rep0`` that commits a message,
    at a round `rec_live.stacked_rounds` absorbs one in; the verifier's own entry of each is its round's ``c_k``, which every
    prover row opens. A prover that exchanges c0 and c1 (the two claims' ring switches, equal in size) registers round 1's row
    as c0 and round 0's as c1, and the verifier's per-row check refuses both; a record whose message sits at a round that
    absorbs none is refused before anything is registered."""
    from verity.core.protocols.sampled_proofs.one_stage import registered as RG

    r = stacked_session.read
    program = O.session_program((stacked_session.d / "inner" / "public.json").read_bytes())
    regs, own = S.opening_messages(r.public, r.private, r.sr.sh, program)
    rounds = A.rounds(r.sr.sh)
    assert list(regs) == [f"m{i}" for i in range(len(rounds))] and rounds[0] == [("rs", 0, v) for v in range(128)]
    msgs = sorted((x for x in r.public["rounds"] if x["stream"] == "stacked/rep0" and x["commits"]), key=lambda x: x["k"])
    assert [x["k"] for x in msgs] == [k for k, x in enumerate(RL.stacked_rounds(22)) if x[3]]

    def opens(i: int, reg) -> bool:
        e = own[f"m{i}"]
        dom = RG.domain(program, f"rec-opening-c{i}", reg.words, 1)
        return e["domain"] == dom.domain_id.hex() and RG.verify_read(dom, bytes.fromhex(e["root"]), 0, reg.bcs[0], reg.path(0))

    for i, x in enumerate(msgs):
        reg = regs[f"m{i}"]
        assert reg.value == f"rec-opening-c{i}" and reg.bcs[0].hex() == x["commits"][0] and opens(i, reg)
        assert own[f"m{i}"] == RR.commits_entry(program, f"rec-opening-c{i}", reg.words, [bytes.fromhex(x["commits"][0])])
    def swap(i: int, j: int):
        return RG.Registered.commit(program, f"rec-opening-c{i}", regs[f"m{j}"].rows, regs[f"m{j}"].salts)

    assert opens(0, swap(0, 0)) and opens(1, swap(1, 1))
    assert not opens(0, swap(0, 1)) and not opens(1, swap(1, 0))
    moved = json.loads(json.dumps(r.public))
    by = {x["k"]: x for x in moved["rounds"] if x["stream"] == "stacked/rep0"}
    by[2]["commits"], by[1]["commits"] = by[1]["commits"], []
    with pytest.raises(ValueError, match=r"committed at rounds \[0, 2,"):
        S.opening_messages(moved, r.private, r.sr.sh, program)


@pytest.mark.slow
def test_rec_vstage_stages_rs_openings_and_its_registered_opening(stacked_session, tmp_path):
    """`rec_vstage.stage_stacked` on the relay's session: R's levels as RecOpen statements on the chain (R's openings after
    both reps', its tops under their own value) and its registered opening reading the stacked stream's messages, the
    chain's rows either side of R's openings and RecSparse's ``rec-claim``; every opening opens and sums, and the opening
    holds. What the residual check of the cheaper test can't: the statements V* stages from a live record. About 3
    minutes, most of it building the opening's statement."""
    ss, r = stacked_session, stacked_session.read
    sp = ss.relay.sparse[0]
    program = O.session_program((ss.d / "inner" / "public.json").read_bytes())
    chain = O.Chain.build(program, r.sr.qs, r.sr.coefs)
    inst = SP.session_rows(sp, [m for _, m in r.sr.reps], [SP.verifier_row(sp, c) for c, _ in r.sr.reps])
    claim = SP.registrations(sp, program, inst)["claim"]
    st = S.stage_stacked(r.public, r.private, program, r.sr, chain, r.sr.coefs, claim, tmp_path, evaluate=False)
    assert st["m"] == 22 and st["openings"] == len(r.sr.qs) and st["levels"]
    assert all(not lv["not_opened"] and not lv["not_summed"] for lv in st["levels"].values()), st["levels"]
    assert [p["holds"] for p in st["opening"]] == [[True]] * len(st["opening"])
