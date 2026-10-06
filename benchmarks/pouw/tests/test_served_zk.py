"""served_zk's window through the proof service (`served_zk/served_commit.py` over P1, `sampled_proofs.service`): a
window of served calls' fp8-p rows committed as C-Flock commits a unit's rows, its tiles padded with filler tiles to a
grid level, registered with exactly PoUW's public items, received by a `Stream`, drawn once, each drawn tile's proof
opened against the window root, delivered once, and the final outcome publishing exactly the public items."""
from __future__ import annotations

import hashlib
import io
import json
import random
import stat
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from verity.primitives.commitments import MerkleTree
from verity.primitives.commitments.rowleaf import ROLE_X
from verity.protocols.accounting.work.pouw import window as Wd
from verity.protocols.accounting.work.pouw.audit import (HM96, HM96_TRANSCRIPT, TILE_SCHEMA, Epoch, Workload, hm96_salt,
                                                         transcript_domain, weights_root)
from verity.protocols.accounting.work.pouw.circuit import leaves as L
from verity.protocols.accounting.work.pouw.circuit import words as W
from verity.protocols.accounting.work.pouw.coins import Challenger as Coins
from verity.protocols.accounting.work.pouw.coins import CoinOpening, coin_reason
from verity.protocols.accounting.work.pouw.protocol import Context, Matmul, Shape
from verity.protocols.accounting.work.pouw.schemes import pearl_c as C
from verity.protocols.accounting.work.pouw.schemes.pearl_c_device import DEVICES
from verity.protocols.verification.sampled_proofs import service as S
from verity.protocols.verification.sampled_proofs.one_stage import registration as R

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "served_zk"))
import served_commit as SC  # noqa: E402

PD = SC._pd()

K, M, N = 128, 3, 4
#: the pass's calls (5 and 9): 16 rows by 64 columns, one -h3 tile each, holding the window's M + 1 by N words
CALLS, PASS_N = (5, 9), 64
#: each pass's tile leaves and tile commitment tree, by run
TILES: dict[str, tuple[list[bytes], MerkleTree]] = {}
SALT_KEY = bytes(range(32))


def _rows(seed: int, n: int) -> list[bytes]:
    x = np.random.default_rng(seed).standard_normal((n, K)).astype("<f4")
    return [r.tobytes() for r in x]


def _coins(seed: int = 7) -> Coins:
    """The epoch's coin challenger on a seeded entropy (the challenger's own generator stands in)."""
    return Coins(0, random.Random(seed).randbytes)


def _root_a(run: str, m: int) -> str:
    return hashlib.sha512(f"{run}/a{m}".encode()).hexdigest()


def _manifest(run: str, coins: Coins | None) -> dict:
    """A served pass's manifest as the challenger and the check read it: its calls, its weights' roots and its salt, the
    salt coin of `coins` opened after the weights' root, or (None) a beacon's, as a pass recorded before epoch coins
    carries; and its -h3 tile commitment, over a leaf of the run's for each tile (`TILES`)."""
    roots = {2: hashlib.sha512(run.encode()).digest()}
    work = Workload(tuple(Matmul(m, Shape(16, K, PASS_N), 2) for m in CALLS))
    man = {"run": run, "scheme": PD.SCHEMES[0], "instance": {"k_min": K}, "hm96_salt_key": SALT_KEY.hex(),
           "matmuls": [{"index": m, "m": 16, "k": K, "n": PASS_N, "weight": 2, "root_a": _root_a(run, m)} for m in CALLS],
           "weights": {"2": {"root": roots[2].hex()}}}
    if coins is None:
        beacon = bytes(32)
        epoch = Epoch.unverified(work, roots, beacon)
        man.update(schema=PD.MANIFEST_SCHEMA_RECORDED, beacon=beacon.hex())
    else:
        opening = coins.open("salt", weights_root(roots))
        epoch = Epoch.start(work, roots, coins.commitment, opening)
        man.update(schema=PD.MANIFEST_SCHEMA, coins=coins.commitment.to_json(), salt_opening=opening.to_json())
    leaves = [hashlib.sha512(f"{run}/tile{t}".encode()).digest() for t in range(len(CALLS))]
    tree = MerkleTree(transcript_domain(epoch, len(leaves), HM96),
                      {t: x + hm96_salt(SALT_KEY, HM96_TRANSCRIPT, t) for t, x in enumerate(leaves)}, lambda _: TILE_SCHEMA)
    TILES[run] = (leaves, tree)
    return {**man, "salt": epoch.salt.hex(), "commitment": {"root": tree.commitment.root.hex(), "count": len(leaves)}}


def _raw(man: dict) -> bytes:
    return json.dumps(man, sort_keys=True).encode()


def _params(run: str, raw: bytes) -> dict:
    return {"context": f"served-zk/{run}/{SC.manifest_name(raw)}", "deployment": f"served-zk/{run}",
            "k": K, "eps": "1/2", "delta": "1/10", "grid_bits": Wd.GRID_BITS, "range": [1, 1 << 40], "entries": 4, "w_rows": N}


#: each window's pass by its parameters' context: (the epoch's coins, the manifest's bytes, the manifest)
PASSES: dict[str, tuple[Coins, bytes, dict]] = {}


@pytest.fixture(scope="module")
def served():
    scheme = C.PearlC(k_min=K, forming="v1", device=DEVICES["sm120"], hashing="h3")
    tag = lambda s: hashlib.sha256(s).digest()  # noqa: E731
    ctx = Context(tag(b"salt"), Matmul(5, Shape(16, K, N), 2), tag(b"root_a"), tag(b"root_b"))
    noise = scheme.noise(ctx)
    a = _rows(1, M) + [bytes(4 * K)]                 # a zero row, which forming refuses
    words = {"a": SC.form(scheme, ctx, noise, K, "a", a, list(range(M + 1))),
             "b": SC.form(scheme, ctx, noise, K, "b", _rows(2, N), list(range(N)))}
    return scheme, ctx, noise, words


def _calls(words, run: str = "r-request") -> list[dict]:
    """Two calls of the same rows: 2·M·N = 24 real tiles, grid level 23, padded to 26 with a filler entry of 1 x 2 zero rows."""
    return [{"record": {"matmul": m, "shape": [16, K, N], "weight": 2, "real_rows": M + 1, "served": {"root_a": _root_a(run, m)}},
             "words": words} for m in CALLS]


def _weights() -> list[bytes]:
    return [np.arange(K, dtype="<u2").tobytes() for _ in range(N)]


def _window(words, run="r-request", coins=None, recorded=False):
    """A window of two calls from a pass of `run`, salted by `coins`' salt coin (a seeded one by default) or, `recorded`,
    by a beacon."""
    coins = None if recorded else coins or _coins()
    man = _manifest(run, coins)
    params = _params(run, _raw(man))
    PASSES[params["context"]] = (coins, _raw(man), man)
    root_w, _ = SC.commit_weights(params, _weights())
    public, advice, priv = SC.commit_window(params, _calls(words, run), root_w)
    return params, public, advice, priv


def _register(params, public, **extra):
    return SC.register(params, public, extra)


def _codes(e: pytest.ExceptionInfo) -> set[str]:
    return set(e.value.verdict.codes)


OUT_SALT = bytes(range(192))


def _pub(bca: bytes, bcb: bytes, bco: bytes = bytes(128)) -> bytes:
    """A unit proof's public file as `verity_flock.circuit.write` lays it out: its header, then instance 0's b || c in port
    order (the A row, the B row, the leaf's advice and place, the output row)."""
    return (json.dumps({"instances": 1, "frame_v3": {"schemas": SC.PUB_SCHEMAS}}).encode() + b"\n" + bca + bcb + bytes(256)
            + bco)


def _leaf(params: dict, advice: dict, t: int, leaf: bytes | None = None, at: int | None = None) -> dict:
    """Unit `t`'s output and its tile's opening, as `stage_tile` makes them: its call's tile in the window's pass (the
    run its parameters name) or the filler's; `leaf` and `at` in place of the tile's."""
    e, i, j = Wd.tile_at(advice["entries"], t)
    entry = advice["entries"][e]
    if entry["kind"] == "filler":
        return SC.leaf_opening(leaf or SC.filler_leaf(K), W.leaf_at(i % 64, j % 64, 64) if at is None else at, OUT_SALT)
    leaves, tree = TILES[params["deployment"].split("/", 1)[1]]
    tid = CALLS.index(entry["record"]["matmul"])
    return SC.leaf_opening(leaf or leaves[tid], W.leaf_at(i, j, 16) if at is None else at, OUT_SALT, tid,
                           hm96_salt(SALT_KEY, HM96_TRANSCRIPT, tid), list(tree.open(tid).path))


def _out_bc(lo: dict) -> bytes:
    return L.commit("leaf-h3", ROLE_X, L.words_leaf_h3(bytes.fromhex(lo["leaf"]), lo["at"]), bytes.fromhex(lo["out_salt"]))


def _unit_pub(params, advice, priv, t: int, lo: dict | None = None) -> bytes:
    e, i, j = Wd.tile_at(advice["entries"], t)
    return _pub(priv[e]["a"]["bcs"][i], priv[e]["b"]["bcs"][j], _out_bc(lo or _leaf(params, advice, t)))


def _op(params, advice, priv, t: int, lo: dict | None = None) -> dict:
    return SC.opening(advice, priv, params, t, lo or _leaf(params, advice, t))


def _passed(params) -> SC.Pass:
    coins, raw, man = PASSES[params["context"]]
    return SC.read_pass(raw, coins.commitment if coins else None, coins is None)


def test_rows_are_the_schemes_operands(served):
    """Each formed row is the fp8-p row of the operands `PearlC.encode` forms for a tile holding it, and the refused row is
    none of the call's kept rows."""
    scheme, ctx, noise, words = served
    assert words["a"][M] is None and all(w is not None for w in words["a"][:M] + words["b"])
    from verity.protocols.accounting.work.pouw.protocol import Tile
    ops = scheme.encode(ctx, noise, _rows(1, M)[1:2], _rows(2, N)[3:4], Tile((1,), (3,)))
    assert words["a"][1] == L.words_fp8_p(ops.a.codes[0], ops.pa[0])
    assert words["b"][3] == L.words_fp8_p(ops.b.codes[0], ops.pb[0])


def test_the_window_is_padded_to_a_grid_level_with_filler(served):
    """The calls' kept tiles, then a filler entry of zero rows up to the padded count, then empty entries to the bound; the
    window statement holds on the advice, each side's root is under one-stage's served domain of its entry's port, and
    replaying the salts gives the same commitment."""
    *_, words = served
    params, public, advice, priv = _window(words)
    work = Wd.bucket(2 * M * N)
    assert (advice["real"], public["level"], work.credited, work.n) == (2 * M * N, work.level, 24, 26)
    kinds = [e["kind"] for e in advice["entries"]]
    assert kinds == ["call", "call", "filler", "none"]
    call, filler = advice["entries"][0], advice["entries"][2]
    assert call["a"]["keep"] == list(range(M)) and call["a"]["rows"] == M + 1 and (call["lo"], call["hi"]) == (0, M * N)
    assert (filler["lo"], filler["hi"]) == (24, 26) and (filler["a"]["rows"], filler["b"]["rows"]) == (1, 2)
    assert not priv[2]["a"]["words"].any() and not priv[2]["b"]["words"].any()
    assert Wd.window_problem(advice["entries"], work, params["entries"]) is None
    _, _, part = SC.population(K, work.n)
    for e in (0, 2):
        for side in "ab":
            dom, schema = SC.side_domain(params, public["level"], e, side, advice["entries"][e][side]["rows"])
            assert dom.domain_id.hex() == R.served_domain(part["program"], SC.PT.digest(part), f"{e}/{side}", schema,
                                                           advice["entries"][e][side]["rows"], params["context"])
    assert priv[0]["b"]["bcs"][2] == L.commit("fp8-p", SC.SIDES["b"], words["b"][2], priv[0]["b"]["salts"][2])
    salts = {e: {s: priv[e][s]["salts"] for s in "ab"} for e in range(3)}
    again, advice2, _ = SC.commit_window(params, _calls(words), public["items"]["root_W"], {**salts, "entries": priv["entries"]})
    assert again == public and advice2["entries"] == advice["entries"]


def test_register_makes_exactly_the_public_items_public(served):
    """The record's window holds exactly the registered public items (work, check, sampling, root_W) and nothing of the
    calls: no shape, no call index, no row count.  One more item, a root_W that is no registered root, or a commitment
    whose level the range refuses, is refused."""
    *_, words = served
    params, public, _, _ = _window(words)
    rec = _register(params, public)
    assert rec["window"] == {"kind": "served", "context": params["context"], "public": public["items"]}
    assert set(rec["window"]["public"]) == set(Wd.REGISTERED) and rec["population"] == 26 and rec["scheme"] == S.SCHEME
    assert rec["law"] == {"law": "subset", "k": rec["window"]["public"]["sampling"]["law"]["k"]}
    assert [r["port"] for r in rec["roots"]] == [SC.PORT, "W"]
    assert S.check_public(SC.spec_of(params, public["level"]), rec).ok and SC.public_problem(params, rec).ok
    text = json.dumps(rec)
    for leak in ('"matmul"', '"shape"', '"real_rows"', '"rows"', '"keep"'):
        assert leak not in text
    with pytest.raises(S.Refused) as e:
        _register(params, public, shapes=[[16, K, N]])
    assert _codes(e) == {"public-items"}
    with pytest.raises(S.Refused) as e:
        _register(params, public, root_W="00" * 64)
    assert _codes(e) == {"public-value"}
    undeclared = SC.spec_of(params, public["level"])
    undeclared = S.Spec(undeclared.values, undeclared.selection, undeclared.check, undeclared.public[:2], undeclared.context)
    assert set(S.check_public(undeclared, rec).detail["undeclared"]) == {f"leaves:{SC.PORT}", f"domain:{SC.PORT}", "leaves:W",
                                                                          "domain:W"}
    with pytest.raises(ValueError, match="outside"):
        SC.work_at({**params, "range": [100, 1 << 40]}, public["level"])


def _challenger(params, units=None, coins=None) -> SC.Challenger:
    """The challenger holding the epoch's coins that salted the window's pass; `units` a stand-in for Lean's draw."""
    sampler = SC.python_sampler if units is None else (lambda law, n, src: list(units)[:law["k"]])
    return SC.Challenger(params, coins or PASSES[params["context"]][0], sampler, clock=lambda: 0.0)


def _held(ch, name, rec, **kw):
    """`ch` receives window `rec` from its pass (the manifest its parameters name)."""
    return ch.register(name, rec, manifest=PASSES[ch.params["context"]][1], **kw)


def test_the_challenger_holds_every_item_to_its_own(served):
    """The challenger's stream registers the honest record and its receipt names it.  An extra public key, a raised level
    (more credit), a K′ other than its own, another request's record (its own context) or a field one-stage doesn't define
    is refused, and so is a second registration while the first has no outcome."""
    *_, words = served
    params, public, _, _ = _window(words)
    rec = _register(params, public)
    ch = _challenger(params)

    def refused(r) -> set[str]:
        with pytest.raises(S.Refused) as e:
            _held(ch, "x", r)
        return _codes(e)

    extra = json.loads(json.dumps(rec))
    extra["window"]["public"]["shapes"] = [[16, K, N]]
    assert refused(extra) == {"public-items"}
    raised = json.loads(json.dumps(rec))
    raised["window"]["public"]["work"]["level"] += 1
    assert refused(raised) == {"public-value", "population"}
    k = json.loads(json.dumps(rec))
    k["window"]["public"]["sampling"]["law"]["k"] -= 1
    assert refused(k) == {"public-value"}
    oparams, opublic, _, _ = _window(words, run="r-other")
    assert "public-context" in refused(_register(oparams, opublic))
    assert refused({**rec, "tiles": [2, 3]}) == {"keys"}
    assert "x" not in ch.regs
    reg = _held(ch, "honest", rec)
    assert reg.receipt["registration"] == R.digest(rec) == reg.digest and reg.order == 0
    assert refused(rec) == {"open"}


def test_select_draws_once_over_the_draw_coin_opened_after_the_registration(served):
    """The draw is the law's subset:K′ over the padded population, over the key of the epoch's draw coin, which the
    challenger opens after the registration (its digest) and checks against its commitment; Lean's draw on that key's
    stream is `stream_subset`'s.  A second select is refused, and so is a retry after a failed draw: the coin opens once.
    A replay from the recorded opening draws the same units; a tampered recorded coin is refused."""
    *_, words = served
    params, public, _, _ = _window(words)
    rec = _register(params, public)
    k = SC.law_of(params, public["level"])["k"]
    ch = _challenger(params)
    reg = _held(ch, "honest", rec)
    assert ch.passes == {"honest": "coin"} and "draw" not in ch.openings
    d = ch.select("honest")
    assert len(d.units) == k and d.draw["population"] == 26 and d.units == sorted(d.units) and all(0 <= u < 26 for u in d.units)
    o = ch.openings["draw"]
    assert o.after == bytes.fromhex(reg.digest) and coin_reason(ch.coins.commitment, o, "draw", o.after) is None
    st = ch.regs["honest"][0]
    src = SC.draw_bytes(SC.draw_key(o.coin, st, reg), d.law, 26)
    assert len(src) == 2 * k * SC.width(26) + 64 and d.units == SC.stream_subset(k, 26, src)
    assert ch.coins_of("honest") == {"commitment": ch.coins.commitment.to_json(), "pass": "coin", "draw": o.to_json()}
    with pytest.raises(S.Refused) as e:
        ch.select("honest")
    assert _codes(e) == {"drawn"}

    again = SC.Challenger(params, SC.Recorded(ch.coins.commitment, [o]), clock=lambda: 0.0)
    _held(again, "honest", rec)
    assert again.select("honest").draw == d.draw
    forged = CoinOpening(o.epoch, o.label, o.after, bytes([o.coin[0] ^ 1]) + o.coin[1:], o.salt, o.path)
    tampered = SC.Challenger(params, SC.Recorded(ch.coins.commitment, [forged]), clock=lambda: 0.0)
    _held(tampered, "honest", rec)
    with pytest.raises(S.Refused) as e:
        tampered.select("honest")
    assert _codes(e) == {"coin"} and "does not open against" in e.value.verdict.detail["coin"]

    params, public, _, _ = _window(words)
    rec = _register(params, public)
    bad = SC.Challenger(params, PASSES[params["context"]][0], lambda law, n, src: list(range(law["k"] + 1)), clock=lambda: 0.0)
    _held(bad, "honest", rec)
    with pytest.raises(S.Refused):
        bad.select("honest")
    with pytest.raises(S.Refused) as e:
        bad.select("honest")
    assert _codes(e) == {"coin"} and "already open" in e.value.verdict.detail["coin"]


def test_a_window_whose_pass_coin_is_foreign_early_mislabelled_or_tampered_is_refused(served):
    """The challenger re-starts the window's pass's epoch from its manifest under its own coin commitment: a salt coin of
    another challenger, opened after what it had not recorded, labelled the draw coin or tampered is refused, and so is a
    pass whose commitment is not the epoch's, whose salt is not its coin's, which carries no coin or names no pass, or
    which is not the pass the window's parameters name.  A pass recorded before epoch coins (a v1 manifest, a beacon's
    salt) is received only as a recorded replay, and a manifest of any other schema not at all.  Nothing refused opens a
    coin."""
    *_, words = served
    params, public, _, _ = _window(words)
    rec = _register(params, public)
    coins, raw, man = PASSES[params["context"]]
    opening = CoinOpening.from_json(man["salt_opening"])

    def refused(m, recorded=False, code="coin"):
        ch = _challenger(params)
        with pytest.raises(S.Refused) as e:
            ch.register("x", rec, manifest=None if m is None else _raw(m), recorded=recorded)
        assert _codes(e) == {code} and not ch.openings and "x" not in ch.regs
        return e.value.verdict.detail[code]

    for case, why in (("foreign", "coin does not open against the epoch's coin commitment"),
                      ("early", "coin was not opened after what the verifier recorded"),
                      ("mislabelled", "coin is the epoch's 'draw' coin, not its 'salt' coin"),
                      ("tampered", "coin does not open against the epoch's coin commitment")):
        assert refused(SC.coin_variant(man, coins.commitment, opening, case)) == f"the salt's {why}", case
    assert "not the epoch's" in refused({**man, "coins": _coins(8).commitment.to_json()})
    assert "salt is not its coin's" in refused(SC.coin_variant({**man, "salt": "00" * 32}, coins.commitment, opening, "resalted"))
    assert "salt is not its coin's" in refused(SC.coin_variant(man, coins.commitment, opening, "resalted"))
    assert "no salt coin" in refused({k: v for k, v in man.items() if k not in ("coins", "salt_opening")})
    assert "names no pass" in refused(None)
    assert "not the one the window's parameters name" in refused({**man, "note": "another pass"}, code="manifest")
    assert _held(_challenger(params), "honest", rec).digest == R.digest(rec)

    bparams, bpublic, _, _ = _window(words, run="r-beacon", recorded=True)
    brec, braw = _register(bparams, bpublic), PASSES[bparams["context"]][1]
    with pytest.raises(S.Refused) as e:
        _challenger(bparams, coins=_coins()).register("x", brec, manifest=braw)
    assert _codes(e) == {"coin"} and f"is {PD.MANIFEST_SCHEMA_RECORDED}, recorded before epoch coins" in e.value.verdict.detail["coin"]
    _, _, bman = PASSES[bparams["context"]]
    for schema in ("verity-vllm/pouw-pearl-c-run/v0", None):
        ch = _challenger(bparams, coins=_coins())
        with pytest.raises(S.Refused) as e:
            ch.register("x", brec, manifest=_raw({**bman, "schema": schema}), recorded=True)
        assert f"schema {schema!r}, not {PD.MANIFEST_SCHEMA}" in e.value.verdict.detail["coin"]
    ch = _challenger(bparams, coins=_coins())
    ch.register("honest", brec, manifest=braw, recorded=True)
    assert ch.passes == {"honest": "recorded beacon"} and len(ch.select("honest").units) == SC.law_of(bparams, bpublic["level"])["k"]


def test_the_challenger_on_the_serving_controllers_state_holds_its_coins(tmp_path, monkeypatch, capsys):
    """e2e.py's coins under ``POUW_COIN_STATE`` and `served_commit.py challenger --state` on the same file publish one
    commitment and open the one salt coin; the state file is its owner's alone, and a missing one is refused, never
    replaced by fresh coins."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pearl_c_vllm"))
    import e2e
    state, after = tmp_path / "state" / "r1", bytes(range(64))
    monkeypatch.setenv("POUW_COIN_STATE", str(state))
    served = e2e.epoch_coins(type("A", (), {"retain": None})(), 0)
    assert stat.S_IMODE(state.stat().st_mode) == 0o600
    assert e2e.epoch_coins(type("A", (), {"retain": None})(), 0).commitment == served.commitment
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"op": "salt", "after": after.hex(), "out": f"{tmp_path}/salt.json"}) + "\n"))
    (tmp_path / "params.json").write_text("{}")
    assert SC.main(["challenger", f"{tmp_path}/params.json", f"{tmp_path}/log", "--state", str(state)]) == 0
    assert json.loads((tmp_path / "log" / "coins.json").read_text()) == served.commitment.to_json()
    assert json.loads((tmp_path / "salt.json").read_text()) == served.open("salt", after).to_json()
    with pytest.raises(SystemExit, match="no challenger state"):
        SC.main(["challenger", f"{tmp_path}/params.json", f"{tmp_path}/log2", "--state", str(tmp_path / "none")])


def test_stream_subset_is_the_exact_partial_fisher_yates_on_one_stream():
    """`stream_subset` reads `width(n - i)`-byte words in order, skips a word at or above the largest multiple of the
    range, and runs out on a short stream, as Lean's ``Draw.subset`` does."""
    n, k = 26, 3
    w = SC.width(n)
    word = lambda x: x.to_bytes(w, "big")  # noqa: E731
    top = (1 << (8 * w)) - (1 << (8 * w)) % n
    src = word(top) + word(5) + word(0) + word(199)          # top is refused; then 5 % 26, 0 % 25, 199 % 24 = 7
    assert SC.stream_subset(k, n, src) == [1, 5, 9]
    assert SC.stream_subset(k, n, src[:-1]) is None
    key = SC.derive(b"\x01" * 32, SC.DRAW_DOMAIN, {"x": "y"})
    units = SC.stream_subset(8, 1000, SC.draw_bytes(key, {"law": "subset", "k": 8}, 1000))
    assert len(set(units)) == 8 and units == sorted(units) and all(0 <= u < 1000 for u in units)


def test_check_opens_the_drawn_units_rows_only(served, tmp_path):
    """A proof's public file opens a drawn unit, a call's tile or the filler's, only when its rows are that unit's rows of
    this window: another row, another window's rows, another window's registration, or an undrawn unit is rejected."""
    *_, words = served
    params, public, advice, priv = _window(words)
    rec, passed = _register(params, public), _passed(params)
    SC.save(tmp_path / "p.npz", priv)
    priv = SC.load(tmp_path / "p.npz")
    draw = {"law": "subset", "population": 26, "k": 3, "units": [5, 13, 25]}
    for t in draw["units"]:
        assert SC.check(params, rec, draw, t, _unit_pub(params, advice, priv, t), _op(params, advice, priv, t), passed=passed) is None
    assert _op(params, advice, priv, 25)["entry_value"]["kind"] == "filler"
    op = _op(params, advice, priv, 5)
    chk = lambda t, pub, op, **kw: SC.check(params, rec, draw, t, pub, op, passed=passed, **kw)  # noqa: E731
    assert "not entry 0's row" in chk(5, _unit_pub(params, advice, priv, 6), op)
    assert "not drawn" in chk(6, _unit_pub(params, advice, priv, 6), _op(params, advice, priv, 6))
    assert chk(6, _unit_pub(params, advice, priv, 6), _op(params, advice, priv, 6), drawn=False) is None
    assert "not the unit's row" in chk(6, _unit_pub(params, advice, priv, 6), op, drawn=False)
    oparams, opublic, oadvice, opriv = _window(words, run="r-other")
    orec = _register(oparams, opublic)
    assert "row" in chk(5, _unit_pub(oparams, oadvice, opriv, 5), op)
    assert "window" in SC.check(params, orec, draw, 5, _unit_pub(params, advice, priv, 5), op, passed=passed)
    forged = {**op, "entry_value": {**op["entry_value"], "hi": op["entry_value"]["hi"] + 1}}
    assert "not the registered window's entry" in chk(5, _unit_pub(params, advice, priv, 5), forged)


def test_check_holds_the_units_output_to_the_passs_tile_leaf(served):
    """A unit's output row must be the opened leaf and place, the place word (i, j)'s in the pass's tile holding it, and the
    leaf that tile's under the pass's tile commitment: a unit proving another leaf (another tile's, or the other pass's),
    at another place, or opened under another pass, is rejected; a filler unit's leaf is the zero rows' tile's."""
    *_, words = served
    params, public, advice, priv = _window(words)
    rec, passed = _register(params, public), _passed(params)
    draw = {"units": [5, 13, 25]}
    chk = lambda t, lo, p=passed: SC.check(params, rec, draw, t, _unit_pub(params, advice, priv, t, lo),  # noqa: E731
                                           _op(params, advice, priv, t, lo), passed=p)
    leaves, _ = TILES["r-request"]
    assert "not tile 0's under the pass's tile commitment" in chk(5, _leaf(params, advice, 5, leaf=leaves[1]))
    oparams, *_ = _window(words, run="r-other")
    assert "not tile 1's" in chk(13, _leaf(params, advice, 13, leaf=TILES["r-other"][0][1]))
    assert "place" in chk(5, _leaf(params, advice, 5, at=W.leaf_at(2, 1, 16)))
    assert "zero rows' tile's" in chk(25, _leaf(params, advice, 25, leaf=leaves[0]))
    assert "place" in chk(25, _leaf(params, advice, 25, at=W.leaf_at(0, 0, 16)))
    assert "not the one the window's parameters name" in chk(5, _leaf(params, advice, 5), _passed(oparams))
    assert "no pass" in chk(5, _leaf(params, advice, 5), None)
    lo = _leaf(params, advice, 5)
    tampered = {**lo, "leaf": leaves[1].hex()}
    assert "output row is not the opened tile leaf" in SC.check(params, rec, draw, 5, _unit_pub(params, advice, priv, 5, lo),
                                                                _op(params, advice, priv, 5, tampered), passed=passed)
    assert "not the unit's tile 0" in chk(5, {**lo, "tile": 1})
    assert "not tile 0's" in chk(5, {**lo, "tile_salt": "00" * 192})


def _session(t: int) -> str:
    """Unit `t`'s session record's SHA-512, as the firewall delivers it and Lean's verdict names it."""
    return hashlib.sha512(f"session {t}".encode()).hexdigest()


def _unit(t: int, lean: bool = True, opening: bool = True, session: int | None = None) -> dict:
    stage = {"definition": "Pc8TileHidden", "circuit_sha512": "11" * 64, "class_sha512": "22" * 64, "ands": 7, "k_log": 3,
             "binding": "hash scratch"}
    return {"unit": t, "statement": SC.statement(stage, b"pub"),
            "verdicts": {"lean": {"of_record": True, "accepted": lean, "why": "", "digest": "33" * 64,
                                  "sessions": [_session(t if session is None else session)]},
                         "opening": {"of_record": False, "accepted": opening, "why": ""}}}


def test_the_outcome_publishes_exactly_the_public_items(served):
    """The outcome over every drawn unit accepts only when each unit's Lean verdict and opening check accept; PoUW's public
    items of it are exactly `window.PUBLIC`; it waits for the proof's delivery, is final, and units other than the draw's
    are refused, and so is a Lean verdict about sessions other than the delivered ones.  A control stream replays the
    honest draw: a reject there is final, and its failure budget stops it."""
    *_, words = served
    params, public, _, _ = _window(words)
    rec = _register(params, public)
    ch = _challenger(params, [20, 3, 25, 7, 11])
    _held(ch, "honest", rec)
    d = ch.select("honest")
    assert d.units == [3, 7, 20, 25]
    units, delivered = [_unit(t) for t in d.units], [_session(t) for t in d.units]
    with pytest.raises(S.Pending):
        ch.outcome("honest", units)
    assert ch.deliver("honest", delivered) is None
    with pytest.raises(S.Refused) as e:
        ch.outcome("honest", [_unit(t, session=99 if t == 7 else None) for t in d.units])
    assert _codes(e) == {"sessions"}
    with pytest.raises(S.Refused) as e:
        ch.outcome("honest", units[:-1])
    assert _codes(e) == {"units"}
    o, pub, body = ch.outcome("honest", units)
    assert o.accepted and o.profile is not None and body["audit"]["proved_units"] == d.units and body["sessions"] == delivered
    assert set(pub) == {p[0] for p in Wd.PUBLIC} and pub["verdict"] is True and pub["drawn"] == d.units
    assert pub["work"] == public["items"]["work"] and pub["root_W"] == public["items"]["root_W"]
    assert body["audit"]["verdicts"]["lean"]["sessions"] == delivered
    for again in (lambda: ch.outcome("honest", units), lambda: ch.deliver("honest", delivered)):
        with pytest.raises(S.Refused) as e:
            again()
        assert _codes(e) == {"final"}
    for i, bad in enumerate(([_unit(t, lean=t != 7) for t in d.units], [_unit(t, opening=False) for t in d.units])):
        name = f"ctl-{i}"
        ch.register(name, rec, stream=name, replay="honest")
        assert ch.select(name).units == d.units
        ch.deliver(name, delivered)
        o, pub, body = ch.outcome(name, bad)
        assert not o.accepted and o.verdict.codes == ("reject",) and pub["verdict"] is False and o.profile is None
        with pytest.raises(S.Refused) as e:
            _held(ch, f"{name}-retry", rec, stream=name)
        assert _codes(e) == {"stopped"}
    assert body["audit"]["verdicts"]["opening"]["units"] == [False] * 4


def test_lean_and_rust_verdicts_are_read_from_their_outputs(tmp_path):
    """Lean's verdict is of record and names, for each session it printed a VERDICT of, that session record's SHA-512 (the
    bytes `deliver` hashes); it accepts only when every session's does.  Rust's is beside it, not of record."""
    a, b = tmp_path / "s-a", tmp_path / "s-b"
    for d in (a, b):
        d.mkdir()
        (d / "session.json").write_text(json.dumps({"session": d.name}))
    sha = {d: hashlib.sha512((d / "session.json").read_bytes()).hexdigest() for d in (a, b)}
    line = {"session": f"{a}/", "accepted": False, "why": "ring-switch claim 2 mismatch"}
    (tmp_path / "lean.out").write_text(f"setup: 3 ms\nVERDICT {json.dumps(line)}\n")
    v = SC.lean_verdict(tmp_path / "lean.out")
    assert v["of_record"] and v["accepted"] is False and "ring-switch" in v["why"] and len(v["digest"]) == 128
    assert v["sessions"] == [sha[a]] and v["digest"] == hashlib.sha512(SC.canonical_json(line)).hexdigest()
    two = [{"session": str(a), "accepted": True, "why": ""}, {"session": str(b), "accepted": True, "why": ""}]
    (tmp_path / "two.out").write_text("".join(f"VERDICT {json.dumps(x)}\n" for x in two))
    assert SC.lean_verdict(tmp_path / "two.out")["sessions"] == [sha[a], sha[b]] and SC.lean_verdict(tmp_path / "two.out")["accepted"]
    (tmp_path / "two.out").write_text("".join(f"VERDICT {json.dumps(x)}\n" for x in (two[0], {**two[1], "accepted": False, "why": "w"})))
    assert SC.lean_verdict(tmp_path / "two.out")["accepted"] is False and SC.lean_verdict(tmp_path / "two.out")["why"] == "w"
    (tmp_path / "gone.out").write_text(f"VERDICT {json.dumps({**line, 'session': str(tmp_path / 'gone')})}\n")
    (tmp_path / "bare.out").write_text('VERDICT {"accepted":true,"why":""}\n')
    assert SC.lean_verdict(tmp_path / "gone.out")["sessions"] == [None] == SC.lean_verdict(tmp_path / "bare.out")["sessions"]
    (tmp_path / "none.out").write_text("")
    assert SC.lean_verdict(tmp_path / "none.out")["accepted"] is None and SC.lean_verdict(tmp_path / "none.out")["sessions"] == []
    (tmp_path / "prove.out").write_text('LIVE\t{"verdict": {"accepted": true, "why": ""}}\n')
    assert SC.rust_verdict(tmp_path / "prove.out")["accepted"] is True and not SC.rust_verdict(tmp_path / "prove.out")["of_record"]


def test_the_cli_runs_the_window_through_the_service(served, tmp_path, monkeypatch, capsys):
    """The job's steps through `main`: register (and the refused extra item), then one challenger process's requests,
    as the job sends them a line at a time: its coin commitment, its salt coin opened after the pass's weights' root (as
    serving asks for it), the refused edited record and the passes whose salt coin is a control's, the honest one,
    Lean's draw (a stand-in, on the draw coin's key) and the refused second select, the opening checks, the delivery, the
    outcome and the refused second one.  `replay` re-runs them from the recorded coins byte for byte, and refuses a
    tampered record."""
    *_, words = served
    params, public, advice, priv = _window(words)
    coins, raw, man = PASSES[params["context"]]
    (tmp_path / "params.json").write_text(json.dumps(params))
    (tmp_path / "manifest.json").write_bytes(raw)
    opening = CoinOpening.from_json(man["salt_opening"])
    c = tmp_path / "commit"
    c.mkdir()
    (c / "commitment.json").write_text(json.dumps(public))
    p, t = str(tmp_path / "params.json"), str(tmp_path)
    assert SC.main(["register", str(c), p, f"{t}/registration.json"]) == 0
    assert SC.main(["register", str(c), p, f"{t}/x.json", "--extra", "shapes=[[16,128,4]]", "--expect-refused"]) == 0
    assert SC.main(["edit", f"{t}/registration.json", f"{t}/extra.json", 'window.public.calls=2']) == 0
    (tmp_path / "coins.json").write_text(json.dumps(coins.commitment.to_json()))
    (tmp_path / "opening.json").write_text(json.dumps(opening.to_json()))
    (tmp_path / "resalt.json").write_text(json.dumps({**man, "salt": "00" * 32}))
    for case in SC.CASES:
        src = "resalt.json" if case == "resalted" else "manifest.json"
        assert SC.main(["coin-variant", f"{t}/{src}", f"{t}/coins.json", f"{t}/opening.json", f"{t}/coin-{case}.json", case]) == 0
    lean = tmp_path / "lean"
    lean.write_text(f"#!{sys.executable}\nimport json, sys\na = sys.argv\np, k = int(a[a.index('--population') + 1]), "
                    "int(a[a.index('--k') + 1])\nw = (p.bit_length() + 64 + 7) // 8\n"
                    "sys.exit(1) if len(bytes.fromhex(a[a.index('--stream') + 1])) != 2 * k * w + 64 else None\n"
                    "print('DRAW ' + json.dumps({'law': 'subset', 'population': p, 'k': k, 'units': list(range(p - k, p))}))\n")
    lean.chmod(lean.stat().st_mode | stat.S_IEXEC)
    k = SC.law_of(params, public["level"])["k"]
    drawn = list(range(26 - k, 26))                  # the stand-in's draw holds the filler's tiles 24 and 25
    rec = json.loads((tmp_path / "registration.json").read_text())
    (tmp_path / "drawn.json").write_text(json.dumps({"units": drawn}))
    rows = []
    for u in drawn:
        (tmp_path / f"pub{u}.bin").write_bytes(_unit_pub(params, advice, priv, u))
        (tmp_path / f"op{u}.json").write_text(json.dumps(_op(params, advice, priv, u)))
        assert SC.main(["check", p, f"{t}/registration.json", f"{t}/drawn.json", str(u),
                        f"{t}/pub{u}.bin", f"{t}/op{u}.json", "--pass", f"{t}/manifest.json", "--coins", f"{t}/coins.json"]) == 0
        (tmp_path / f"open{u}.json").write_text(json.dumps({"of_record": False, "accepted": True, "why": ""}))
        (tmp_path / f"s{u}").mkdir()
        (tmp_path / f"s{u}" / "session.json").write_text(json.dumps({"unit": u}))
        (tmp_path / f"lean{u}.out").write_text(f"VERDICT {json.dumps({'session': f'{t}/s{u}/', 'accepted': True, 'why': ''})}\n")
        (tmp_path / f"stage{u}.json").write_text(json.dumps({"definition": "d", "circuit_sha512": "1", "class_sha512": "2",
                                                             "ands": 1, "k_log": 1, "binding": "hash scratch"}))
        rows.append({"unit": u, "stage": f"{t}/stage{u}.json", "pub": f"{t}/pub{u}.bin", "lean": f"{t}/lean{u}.out",
                     "opening": f"{t}/open{u}.json", "session": f"{t}/s{u}/session.json"})
    (tmp_path / "units.json").write_text(json.dumps(rows))
    honest = {"registration": f"{t}/registration.json", "manifest": f"{t}/manifest.json"}
    reqs = [{"op": "coins"},
            {"op": "salt", "after": opening.after.hex(), "out": f"{t}/salt.json"},
            {"op": "register", "name": "extra", "registration": f"{t}/extra.json", "manifest": f"{t}/manifest.json"},
            *({"op": "register", "name": case, "registration": f"{t}/registration.json", "manifest": f"{t}/coin-{case}.json"}
              for case in SC.CASES),
            {"op": "register", "name": "honest", **honest, "out": f"{t}/receipt.json"},
            {"op": "select", "name": "honest", "out": f"{t}/draw.json"},
            {"op": "select", "name": "honest", "out": f"{t}/draw2.json"},
            {"op": "outcome", "name": "honest", "units": f"{t}/units.json"},
            {"op": "deliver", "name": "honest", "units": f"{t}/units.json"},
            {"op": "outcome", "name": "honest", "units": f"{t}/units.json", "out": f"{t}/outcome"},
            {"op": "outcome", "name": "honest", "units": f"{t}/units.json", "out": f"{t}/outcome2"},
            {"op": "quit"}, {"op": "select", "name": "honest"}]
    monkeypatch.setattr(sys, "stdin", io.StringIO("".join(json.dumps(r) + "\n" for r in reqs)))
    monkeypatch.setattr(SC, "Coins", lambda epoch: _coins())      # the process's coins are the ones that salted the pass
    capsys.readouterr()
    assert SC.main(["challenger", p, f"{t}/log", "--lean", str(lean)]) == 0
    replies = capsys.readouterr().out.splitlines()
    assert [x.split(" ", 1)[0] for x in replies] == ["COINS", "OPENED", "REFUSED", *["REFUSED"] * len(SC.CASES), "REGISTERED",
                                                     "DRAWN", "REFUSED", "PENDING", "DELIVERED", "ACCEPT", "REFUSED"]
    assert all('"coin"' in r for r in replies[3:3 + len(SC.CASES)]) and "salt is not its coin's" in replies[2 + len(SC.CASES)]
    log = tmp_path / "log"
    assert json.loads((log / "coins.json").read_text()) == coins.commitment.to_json()
    assert json.loads((tmp_path / "salt.json").read_text()) == man["salt_opening"]
    salt, draw = (json.loads(x) for x in (log / "openings.jsonl").read_text().splitlines())
    assert salt == man["salt_opening"] and draw["label"] == "draw" and draw["after"] == R.digest(rec)
    d = json.loads((tmp_path / "draw.json").read_text())
    assert d["units"] == drawn and json.loads((tmp_path / "receipt.json").read_text())["registration"] == R.digest(rec)
    assert not (tmp_path / "draw2.json").exists() and not (tmp_path / "outcome2").exists()
    o = json.loads((tmp_path / "outcome" / "outcome.json").read_text())
    assert o["accepted"] and set(o["public"]) == {x[0] for x in Wd.PUBLIC} and o["public"]["drawn"] == drawn
    assert o["coins"] == {"commitment": coins.commitment.to_json(), "pass": "coin", "draw": draw}
    assert Fraction(o["public"]["sampling"]["delta"]) == Fraction(1, 10)
    audit = json.loads((tmp_path / "outcome" / "audit.json").read_text())
    assert audit["sessions"] == [hashlib.sha512((tmp_path / f"s{u}" / "session.json").read_bytes()).hexdigest() for u in drawn]
    assert audit["audit"]["verdicts"]["lean"]["sessions"] == audit["sessions"]
    assert len((log / "requests.jsonl").read_text().splitlines()) == len(reqs) - 2

    assert SC.main(["replay", p, str(log), f"{t}/replay", "--lean", str(lean)]) == 0
    assert capsys.readouterr().out.startswith("SAME ")
    assert (tmp_path / "replay").is_dir() and SC.replay(params, log, tmp_path / "replay2", SC.python_sampler)  # another sampler: other units
    forged = {**draw, "coin": ("00" if draw["coin"][:2] != "00" else "01") + draw["coin"][2:]}
    (log / "openings.jsonl").write_text(json.dumps(salt) + "\n" + json.dumps(forged) + "\n")
    assert SC.main(["replay", p, str(log), f"{t}/replay3", "--lean", str(lean)]) == 1
    out = capsys.readouterr().out
    assert out.startswith("DIFFERENT ") and "does not open against" in out


def _staged(d: Path, rec: dict, proof: int) -> Path:
    s = d / "sessions" / "l0000-1"
    s.mkdir(parents=True)
    (d / "stage.json").write_text(json.dumps({"circuit_sha512": "11", "class_sha512": "22", "ands": 7, "k_log": 3}))
    (d / "pub.bin").write_bytes(b"h\n" + bytes(384))
    (s / "session.json").write_text(json.dumps(rec))
    (s / "verifier.json").write_text(json.dumps({"host": str(d)}))
    (s / "t.rep0.bin").write_bytes(bytes(proof))
    return d


def test_shape_compares_what_the_prover_sends(tmp_path):
    """Two units' proofs have one shape when the circuit, the public file, each proof file's size and the session record's
    skeleton agree; the verifier's own verdict and timings, and its own files, are not the prover's."""
    rec = {"streams": [{"rounds": [{"k": 0, "msg_len": 96, "coins": ["ab" * 16]}]}], "verdict": {"verify_s": 1.5}, "g": 0.25}
    a = _staged(tmp_path / "a", rec, 100)
    b = _staged(tmp_path / "b", {**rec, "verdict": {"verify_s": 12.25, "why": "x"}, "g": 3.125}, 100)
    assert SC.shape(a, b)["same"]
    c = _staged(tmp_path / "c", {**rec, "streams": [{"rounds": [{"k": 0, "msg_len": 97, "coins": ["ab" * 16]}]}]}, 100)
    v = SC.shape(a, c)
    assert not v["same"] and any("msg_len" in x for x in v["diffs"])
    assert not SC.shape(a, _staged(tmp_path / "d", rec, 101))["same"]
