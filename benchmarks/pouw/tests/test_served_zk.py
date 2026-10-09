"""served_zk's window through the proof service (`served_zk/served_commit.py` over P1, `sampled_proofs.service`): a
window of served calls' side rows committed as C-Flock commits a unit's rows, its tiles padded with filler tiles to a
grid level, registered with exactly PoUW's public items, received by a `Stream`, drawn once, each drawn tile's proof
opened against the window root, delivered once, and the final outcome publishing exactly the public items."""
from __future__ import annotations

import hashlib
import io
import json
import random
import shutil
import stat
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from verity.ml.commitments.merkle import tree_levels
from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.pouw import window as Wd
from verity.core.protocols.pouw.audit import Epoch, Workload, weights_root
from verity.core.protocols.pouw.circuit import leaves as L
from verity.core.protocols.pouw.coins import Challenger as Coins
from verity.core.protocols.pouw.coins import CoinOpening, coin_reason
from verity.core.protocols.pouw.protocol import Context, Matmul, Shape
from verity.core.protocols.pouw.schemes import pearl_c as C
from verity.core.protocols.pouw.schemes.pearl_c_device import DEVICES
from verity.core.protocols.sampled_proofs import service as S
from verity.core.protocols.sampled_proofs.one_stage import registration as R

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "served_zk"))
import served_commit as SC  # noqa: E402
import of_record as OR  # noqa: E402

#: the challenger's checks are of record in Lean (`of_record`), which `lake` builds
LEAN = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
PD = SC._pd()

K, M, N = 128, 3, 4


def _rows(seed: int, n: int) -> list[bytes]:
    x = np.random.default_rng(seed).standard_normal((n, K)).astype("<f4")
    return [r.tobytes() for r in x]


def _coins(seed: int = 7) -> Coins:
    """The epoch's coin challenger on a seeded entropy (the challenger's own generator stands in)."""
    return Coins(0, random.Random(seed).randbytes)


def _manifest(run: str, coins: Coins | None) -> dict:
    """A served pass's manifest as the challenger reads it: its calls, its weights' roots and its salt, the salt coin of
    `coins` opened after the weights' root, or (None) a beacon's, as a pass recorded before epoch coins carries."""
    roots = {2: hashlib.sha512(run.encode()).digest()}
    work = Workload((Matmul(5, Shape(16, K, N), 2),))
    man = {"run": run, "matmuls": [{"index": 5, "m": 16, "k": K, "n": N, "weight": 2}], "weights": {"2": {"root": roots[2].hex()}},
           "commitment": {"root": hashlib.sha512(f"{run}/tiles".encode()).hexdigest(), "count": 1}}
    if coins is None:
        beacon = bytes(32)
        return {**man, "schema": PD.MANIFEST_SCHEMA_RECORDED, "beacon": beacon.hex(),
                "salt": Epoch.unverified(work, roots, beacon).salt.hex()}
    opening = coins.open("salt", weights_root(roots))
    return {**man, "schema": PD.MANIFEST_SCHEMA, "coins": coins.commitment.to_json(), "salt_opening": opening.to_json(),
            "salt": Epoch.start(work, roots, coins.commitment, opening).salt.hex()}


def _raw(man: dict) -> bytes:
    return json.dumps(man, sort_keys=True).encode()


def _params(run: str, raw: bytes) -> dict:
    return {"context": f"served-zk/{run}/{hashlib.sha512(raw).hexdigest()}", "deployment": f"served-zk/{run}",
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


def _calls(words) -> list[dict]:
    """Two calls of the same rows: 2·M·N = 24 real tiles, grid level 23, padded to 26 with a filler entry of 1 x 2 zero rows."""
    return [{"record": {"matmul": m, "shape": [16, K, N], "tile0": m, "weight": 2, "real_rows": M + 1}, "words": words}
            for m in (5, 9)]


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
    public, advice, priv = SC.commit_window(params, _calls(words), root_w, SC.root_h3_of(man))
    return params, public, advice, priv


def _register(params, public, **extra):
    return SC.register(params, public, extra)


def _codes(e: pytest.ExceptionInfo) -> set[str]:
    return set(e.value.verdict.codes)


def _pub(bca: bytes, bcb: bytes) -> bytes:
    """A tile proof's public file as `verity_flock.circuit.write` lays it out: its header, then instance 0's b || c in port
    order (the A row, the B row, the tile digest's row)."""
    schemas = {"a0": L.ROW_SCHEMA["fp8-pi", SC.SIDES["a"]], "b0": L.ROW_SCHEMA["fp8-pi", SC.SIDES["b"]],
               "o0": L.ROW_SCHEMA["tile", SC.SIDES["a"]]}
    return json.dumps({"instances": 1, "frame_v3": {"schemas": schemas}}).encode() + b"\n" + bca + bcb + bytes(128)


def _unit_pub(advice, priv, t: int) -> bytes:
    e, i, j = Wd.tile_at(SC.entries_of(advice), t)
    return _pub(priv[e]["a"]["bcs"][i], priv[e]["b"]["bcs"][j])


def test_rows_are_the_schemes_operands(served):
    """Each formed row is the fp8-p row of the operands `PearlC.encode` forms for a tile holding it, and the refused row is
    none of the call's kept rows."""
    scheme, ctx, noise, words = served
    assert words["a"][M] is None and all(w is not None for w in words["a"][:M] + words["b"])
    from verity.core.protocols.pouw.protocol import Tile
    ops = scheme.encode(ctx, noise, _rows(1, M)[1:2], _rows(2, N)[3:4], Tile((1,), (3,)))
    assert words["a"][1] == L.words_fp8_p(ops.a.codes[0], ops.pa[0])
    assert words["b"][3] == L.words_fp8_p(ops.b.codes[0], ops.pb[0])


def test_the_window_is_padded_to_a_grid_level_with_filler(served):
    """The calls' kept tiles, then a filler entry of zero rows up to the padded count, then empty entries to the bound, each
    an entry row; the window statement holds on the rows, each side tree holds its kept rows only under the window's side
    domain (from the window root's one-stage served domain), the record keeps the rows' places in the call, and replaying
    the salts gives the same commitment."""
    *_, words = served
    params, public, advice, priv = _window(words)
    work = Wd.bucket(2 * M * N)
    assert (advice["real"], public["level"], work.credited, work.n) == (2 * M * N, work.level, 24, 26)
    entries = SC.entries_of(advice)
    assert [e.kind for e in entries] == ["call", "call", "filler", "none"]
    call, filler = entries[0], entries[2]
    assert (call.n_a, call.n_b, call.lo, call.hi) == (M, N, 0, M * N) and (filler.lo, filler.hi) == (24, 26)
    assert (filler.n_a, filler.n_b) == (1, 2) and filler.record == filler.root_h3 == bytes(64) and entries[3] == Wd.Entry("none")
    root_h3 = SC.root_h3_of(PASSES[params["context"]][2])
    assert entries[0].root_h3 == entries[1].root_h3 == root_h3 and advice["root_h3"] == root_h3.hex()
    assert advice["records"][0]["keep"] == {"a": list(range(M)), "b": list(range(N))} and advice["records"][2:] == [None, None]
    assert call.record == Wd.record_digest(advice["records"][0]) and len(priv[0]["a"]["bcs"]) == M
    assert not priv[2]["a"]["words"].any() and not priv[2]["b"]["words"].any()
    assert Wd.window_problem([e.row() for e in entries], work, params["entries"], root_h3) is None
    _, _, part = SC.population(K, work.n)
    (wroot,) = [r for r in public["roots"] if r["port"] == SC.PORT]
    assert wroot["domain"] == R.served_domain(part["program"], SC.PT.digest(part), SC.PORT, SC.ENTRY_SCHEMA, params["entries"],
                                              params["context"])
    for e in (0, 2):
        for side in "ab":
            dom, schema = SC.side_domain(params, public["level"], side)
            assert dom == Wd.side_domain(bytes.fromhex(wroot["domain"]), side) and dom.count == 1 << Wd.DEPTHS[side]
            leaves = SC.side_leaves(dom, schema, priv[e][side]["bcs"])
            root = tree_levels(dom, leaves + [dom.pad(r) for r in range(len(leaves), dom.count)])[-1][0]
            assert root == (entries[e].root_a if side == "a" else entries[e].root_b)
    assert priv[0]["b"]["bcs"][2] == L.commit("fp8-pi", SC.SIDES["b"], Wd.side_row(words["b"][2], 2), priv[0]["b"]["salts"][2])
    assert [Wd.side_index(r) for r in priv[0]["a"]["words"]] == advice["records"][0]["keep"]["a"]
    assert (call.tile0, call.cols, call.m, entries[1].tile0) == (5, 1, 16, 9) and filler.m == filler.tile0 == 0
    salts = {e: {s: priv[e][s]["salts"] for s in "ab"} for e in range(3)}
    again, advice2, _ = SC.commit_window(params, _calls(words), public["items"]["root_W"], root_h3,
                                         {**salts, "entries": priv["entries"]})
    assert again == public and advice2["entries"] == advice["entries"]
    for bad in ({}, {"commitment": {"root": "ab" * 32}}):
        with pytest.raises(SystemExit, match="not a SHA-512"):
            SC.root_h3_of(bad)


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


@LEAN
def test_the_challenger_on_the_serving_controllers_state_holds_its_coins(served, tmp_path, monkeypatch, capsys):
    """e2e.py's coins under ``POUW_COIN_STATE`` and `served_commit.py challenger --state` on the same file publish one
    commitment and open the one salt coin; the state file is its owner's alone, and a missing one is refused, never
    replaced by fresh coins."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pearl_c_vllm"))
    import e2e
    *_, words = served
    params, *_ = _window(words)
    state, after = tmp_path / "state" / "r1", bytes(range(64))
    monkeypatch.setenv("POUW_COIN_STATE", str(state))
    controller = e2e.epoch_coins(type("A", (), {"retain": None})(), 0)
    assert stat.S_IMODE(state.stat().st_mode) == 0o600
    assert e2e.epoch_coins(type("A", (), {"retain": None})(), 0).commitment == controller.commitment
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"op": "salt", "after": after.hex(), "out": f"{tmp_path}/salt.json"}) + "\n"))
    (tmp_path / "params.json").write_text(json.dumps(params))
    capsys.readouterr()
    assert SC.main(["challenger", f"{tmp_path}/params.json", f"{tmp_path}/log", "--state", str(state)]) == 0
    assert capsys.readouterr().out.startswith("OPENED")
    assert json.loads((tmp_path / "log" / "coins.json").read_text()) == controller.commitment.to_json()
    assert json.loads((tmp_path / "salt.json").read_text()) == controller.open("salt", after).to_json()
    with pytest.raises(SystemExit, match="no challenger state"):
        SC.main(["challenger", f"{tmp_path}/params.json", f"{tmp_path}/log2", "--state", str(tmp_path / "none")])


@LEAN
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
    assert refused(raised) == {"public-value"}
    k = json.loads(json.dumps(rec))
    k["window"]["public"]["sampling"]["law"]["k"] -= 1
    assert refused(k) == {"public-value"}
    oparams, opublic, _, _ = _window(words, run="r-other")
    assert refused(_register(oparams, opublic)) == {"window"}
    assert refused({**rec, "tiles": [2, 3]}) == {"keys"}
    assert "x" not in ch.regs
    reg = _held(ch, "honest", rec)
    assert reg.receipt["registration"] == R.digest(rec) == reg.digest and reg.order == 0
    assert refused(rec) == {"open"}


@LEAN
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
    assert len(src) == 2 * k * CC.width(26) + 64 and d.units == CC.stream_subset(k, 26, src)
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


@LEAN
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
    for schema, code, why in (("verity-vllm/pouw-pearl-c-run/v0", "coin", f"schema 'verity-vllm/pouw-pearl-c-run/v0', not {PD.MANIFEST_SCHEMA}"),
                              (None, "manifest", "the pass's manifest names no schema")):
        ch = _challenger(bparams, coins=_coins())
        with pytest.raises(S.Refused) as e:
            ch.register("x", brec, manifest=_raw({**bman, "schema": schema}), recorded=True)
        assert why in e.value.verdict.detail[code]
    assert (CC.MANIFEST_SCHEMA, CC.MANIFEST_RECORDED) == (PD.MANIFEST_SCHEMA, PD.MANIFEST_SCHEMA_RECORDED)
    ch = _challenger(bparams, coins=_coins())
    ch.register("honest", brec, manifest=braw, recorded=True)
    assert ch.passes == {"honest": "recorded beacon"} and len(ch.select("honest").units) == SC.law_of(bparams, bpublic["level"])["k"]


def test_stream_subset_is_the_exact_partial_fisher_yates_on_one_stream():
    """`stream_subset` reads `width(n - i)`-byte words in order, skips a word at or above the largest multiple of the
    range, and runs out on a short stream, as Lean's ``Draw.subset`` does."""
    n, k = 26, 3
    w = CC.width(n)
    word = lambda x: x.to_bytes(w, "big")  # noqa: E731
    top = (1 << (8 * w)) - (1 << (8 * w)) % n
    src = word(top) + word(5) + word(0) + word(199)          # top is refused; then 5 % 26, 0 % 25, 199 % 24 = 7
    assert CC.stream_subset(k, n, src) == [1, 5, 9]
    assert CC.stream_subset(k, n, src[:-1]) is None
    key = SC.derive(b"\x01" * 32, SC.DRAW_DOMAIN, {"x": "y"})
    units = CC.stream_subset(8, 1000, SC.draw_bytes(key, {"law": "subset", "k": 8}, 1000))
    assert len(set(units)) == 8 and units == sorted(units) and all(0 <= u < 1000 for u in units)


@LEAN
def test_check_opens_the_drawn_units_rows_only(served, tmp_path):
    """A proof's public file opens a drawn unit, a call's tile or the filler's, only when its rows are that unit's rows of
    this window: another row, another window's rows, another window's registration, or an undrawn unit is rejected."""
    *_, words = served
    params, public, advice, priv = _window(words)
    rec = _register(params, public)
    SC.save(tmp_path / "p.npz", priv)
    priv = SC.load(tmp_path / "p.npz")
    draw = {"law": "subset", "population": 26, "k": 3, "units": [5, 13, 25]}
    for t in draw["units"]:
        assert SC.check(params, rec, draw, t, _unit_pub(advice, priv, t), SC.opening(advice, priv, params, t)) is None
    assert SC.opening(advice, priv, params, 25)["entry_value"]["kind"] == "filler"
    op = SC.opening(advice, priv, params, 5)
    assert "not entry 0's row" in SC.check(params, rec, draw, 5, _unit_pub(advice, priv, 6), op)
    assert "not drawn" in SC.check(params, rec, draw, 6, _unit_pub(advice, priv, 6), SC.opening(advice, priv, params, 6))
    assert SC.check(params, rec, draw, 6, _unit_pub(advice, priv, 6), SC.opening(advice, priv, params, 6), drawn=False) is None
    assert "not the unit's row" in SC.check(params, rec, draw, 6, _unit_pub(advice, priv, 6), op, drawn=False)
    oparams, opublic, oadvice, opriv = _window(words, run="r-other")
    orec = _register(oparams, opublic)
    assert "row" in SC.check(params, rec, draw, 5, _unit_pub(oadvice, opriv, 5), op)
    assert "window" in SC.check(params, orec, draw, 5, _unit_pub(advice, priv, 5), op)
    forged = {**op, "entry_value": {**op["entry_value"], "hi": op["entry_value"]["hi"] + 1}}
    assert "not the registered window's entry" in SC.check(params, rec, draw, 5, _unit_pub(advice, priv, 5), forged)
    for bad in ({**op["entry_value"], "kind": "tile"}, {**op["entry_value"], "root_a": "00"}, {"kind": "call"}, None):
        assert SC.check(params, rec, draw, 5, _unit_pub(advice, priv, 5), {**op, "entry_value": bad}) == "the opening names no entry"


def _session(t: int) -> str:
    """Unit `t`'s session record's SHA-512, as the firewall delivers it and Lean's verdict names it."""
    return hashlib.sha512(f"session {t}".encode()).hexdigest()


def _unit(t: int, lean: bool = True, opening: bool = True, session: int | None = None) -> dict:
    stage = {"definition": "Pc8TileSide", "circuit_sha512": "11" * 64, "class_sha512": "22" * 64, "ands": 7, "k_log": 3,
             "binding": "hash scratch"}
    return {"unit": t, "statement": SC.statement(stage, b"pub"),
            "verdicts": {"lean": {"of_record": True, "accepted": lean, "why": "", "digest": "33" * 64,
                                  "sessions": [_session(t if session is None else session)]},
                         "opening": {"of_record": False, "accepted": opening, "why": ""}}}


@LEAN
def test_the_outcome_publishes_exactly_the_public_items(served):
    """The outcome over every drawn unit accepts only when each unit's Lean verdict and opening check accept; PoUW's public
    items of it are exactly `window.PUBLIC`; it waits for the proof's delivery, is final, and units other than the draw's
    are refused, and so is a Lean verdict about sessions other than the delivered ones.  A control stream replays the
    honest draw: a reject there is final, and its failure budget stops it."""
    *_, words = served
    params, public, _, _ = _window(words)
    rec = _register(params, public)
    ch = _challenger(params)
    _held(ch, "honest", rec)
    d = ch.select("honest")
    units, delivered = [_unit(t) for t in d.units], [_session(t) for t in d.units]
    with pytest.raises(S.Pending):
        ch.outcome("honest", units)
    assert ch.deliver("honest", delivered) is None
    with pytest.raises(S.Refused) as e:
        ch.outcome("honest", [_unit(t, session=99 if t == d.units[1] else None) for t in d.units])
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
    for i, bad in enumerate(([_unit(t, lean=t != d.units[1]) for t in d.units], [_unit(t, opening=False) for t in d.units])):
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


def _through(ch, rec, name="honest"):
    """Window `rec` received, drawn, delivered and its outcome, every drawn unit accepted."""
    _held(ch, name, rec)
    d = ch.select(name)
    ch.deliver(name, [_session(t) for t in d.units])
    return ch.outcome(name, [_unit(t) for t in d.units])


@LEAN
def test_lean_is_of_record_and_python_its_witness(served, tmp_path):
    """Each of the window's checks runs Lean's port and Python's reference on one input: the witness holds both verdicts,
    the same, Lean's of record, and the published items are Lean's.  Undone (`Record(lean=False)`), the references alone
    give the same outcome and witness no Lean verdict."""
    *_, words = served
    params, public, advice, priv = _window(words)
    rec = _register(params, public)
    ch = _challenger(params)
    o, pub, _ = _through(ch, rec)
    w = ch.witness["honest"]
    assert [e["check"] for e in w] == ["registered", "pass", "registration", "public", "draw", "answer"]
    assert all(e["same"] and e["of_record"] == "lean" and e["lean"] == e["python"] for e in w)
    answer = w[-1]["lean"]
    assert o.accepted and answer["verdict"] == "accept" and pub == answer["public"]
    record = OR.Record(lean=False)
    py = SC.Challenger(params, _coins(), SC.python_sampler, clock=lambda: 0.0, record=record)
    o2, pub2, _ = _through(py, rec)
    assert o2.accepted and pub2 == pub and all(e["lean"] is None and e["of_record"] == "python" for e in py.witness["honest"])
    t = ch.draws["honest"].units[0]
    lean = OR.Record()
    assert SC.check(params, rec, ch.draws["honest"].draw, t, _unit_pub(advice, priv, t), SC.opening(advice, priv, params, t),
                    record=lean) is None
    assert [(e["check"], e["same"], e["lean"]) for e in lean.witness] == [("opening", True, None),
                                                                           ("prelude", True, {"reason": None})]
    lean.close()


@LEAN
def test_r1_r6_of_a_window_are_leans(served):
    """One-stage's R1-R6 of a PoUW registration, honest and with each field changed, through the switch: Lean's codes are
    Python's, none for the honest record and some for each change."""
    *_, words = served
    params, public, _, _ = _window(words)
    rec = _register(params, public)
    ell, n = SC.level_of(rec), rec["population"]
    prog, q = SC.PT.population_program(SC.template(K), n), SC.PT.template_instance_query(SC.template(K))
    law, ports = SC.law_of(params, ell), SC.ports(params, ell)
    p = OR.registration_params(prog, q, law, S.SCHEME, ports, [], params["delta"])
    record = OR.Record()

    def codes(r) -> list[str]:
        return record.registration(p, r, lambda: R.check(r, prog, q, law, S.SCHEME, ports, None, (), [], None, None, None).codes)

    assert codes(rec) == []
    root = rec["roots"][0]
    changes = {"protocol": "verity/other/v1", "program": "00" * 64, "partition": "00" * 64, "query": "00" * 64,
               "population": n + 1, "law": {**law, "k": law["k"] + 1}, "scheme": "frame-v3/hm96", "leaf_layer": "x",
               "anchors": [{"x": 1}], "roots": [{**root, "domain": "00" * 64}, *rec["roots"][1:]]}
    got = {k: codes({**rec, k: v}) for k, v in changes.items()}
    got["leaves"] = codes({**rec, "roots": [{**root, "leaves": root["leaves"] + 1}, *rec["roots"][1:]]})
    got["one-root"] = codes({**rec, "roots": rec["roots"][:1]})
    assert all(got.values()), got
    assert all(e["same"] for e in record.witness) and len(record.witness) == len(got) + 1
    record.close()


@LEAN
def test_the_public_items_rule_refuses_only_what_r1_r6_refuse(served):
    """`service.check_public` has no Lean form: R1-R6 decide what it reads of a PoUW registration, so on generated
    registrations (keys added and dropped, roots renamed, repeated, dropped or given fields, the scheme, query, leaf layer,
    anchors and registered values changed, several at once) it refuses only what Lean's R1-R6 refuse, and refuses some."""
    *_, words = served
    params, public, _, _ = _window(words)
    rec = _register(params, public)
    ell, n = SC.level_of(rec), rec["population"]
    spec = SC.spec_of(params, ell)
    prog, q = SC.PT.population_program(SC.template(K), n), SC.PT.template_instance_query(SC.template(K))
    law, ports = SC.law_of(params, ell), SC.ports(params, ell)
    p = OR.registration_params(prog, q, law, S.SCHEME, ports, [], params["delta"])
    record = OR.Record()

    def mutate(rng: random.Random, r: dict) -> None:
        roots = r.get("roots")
        match rng.randrange(12):
            case 0:
                r[rng.choice(("tiles", "registered", "values", "extra"))] = rng.choice(([2, 3], [], {"x": 1}, None))
            case 1:
                r.pop(rng.choice(sorted(r)), None)
            case 2 if roots:
                roots[rng.randrange(len(roots))]["port"] = rng.choice(("W", SC.PORT, "X", "values", 7))
            case 3 if roots:
                roots.append(dict(rng.choice(roots)))
            case 4 if roots:
                roots.pop(rng.randrange(len(roots)))
            case 5 if roots:
                e = roots[rng.randrange(len(roots))]
                e[rng.choice(("leaves", "domain", "root", "extra", "value"))] = rng.choice((1, "00" * 64, None, "x"))
            case 6 if roots:
                roots[rng.randrange(len(roots))].pop(rng.choice(("leaves", "domain", "root", "port")), None)
            case 7:
                r["scheme"] = rng.choice(("frame-v3/hm96", S.SCHEME + "x", None))
            case 8:
                r["query"] = rng.choice(("00" * 64, None, r.get("query")))
            case 9:
                r["leaf_layer"] = rng.choice(("x", "00" * 64, None))
            case 10:
                r["anchors"] = rng.choice(([{"x": 1}], [], None))
            case _:
                r["population"] = rng.choice((n, n + 1, 0))

    refused = both = 0
    for seed in range(300):
        rng = random.Random(seed)
        r = json.loads(json.dumps(rec))
        for _ in range(rng.choice((1, 1, 2, 3))):
            mutate(rng, r)
        py = list(S.check_public(spec, r).codes)
        lean = record.registration(p, r, lambda r=r: R.check(r, prog, q, law, S.SCHEME, ports, None, (), [], None, None,
                                                             None).codes)
        assert record.public(py, lean) == lean, (seed, py, lean)
        refused += bool(py)
        both += bool(lean)
    assert refused >= 100 and both >= refused, (refused, both)
    assert all(e["same"] for e in record.witness)
    record.close()


@LEAN
def test_a_disagreement_rejects_the_session(served, monkeypatch):
    """A Python reference that disagrees with Lean, at each check, refuses the session as ``lean-python-disagree: <check>``
    with both verdicts in the witness; nothing is retried, and a refused draw is never redrawn."""
    *_, words = served
    params, public, advice, priv = _window(words)
    rec = _register(params, public)

    def disagrees(check: str, patch, step) -> S.Refused:
        ch = _challenger(params, coins=_coins())
        with monkeypatch.context() as m:
            patch(m)
            with pytest.raises(S.Refused) as e:
                step(ch)
        assert _codes(e) == {f"{OR.DISAGREE}: {check}"}
        bad = [x for x in ch.witness["honest"] if not x["same"]]
        assert [x["check"] for x in bad] == [check] and bad[0]["lean"] != bad[0]["python"]
        return ch

    disagrees("registered", lambda m: m.setattr(OR, "py_registered", lambda x: {"reason": "forced"}),
              lambda ch: _held(ch, "honest", rec))
    disagrees("pass", lambda m: m.setattr(OR, "py_pass", lambda x: {"how": "forced"}), lambda ch: _held(ch, "honest", rec))
    disagrees("registration", lambda m: m.setattr(SC.R, "check", lambda *a: SC.Verdict(False, ("R1-law",), {})),
              lambda ch: _held(ch, "honest", rec))
    disagrees("public", lambda m: m.setattr(SC.S, "check_public", lambda *a: SC.Verdict(False, ("keys",), {})),
              lambda ch: _held(ch, "honest", rec))
    ch = disagrees("draw", lambda m: m.setattr(OR, "py_draw", lambda x: {"refused": ["draw", "forced"]}),
                   lambda ch: (_held(ch, "honest", rec), ch.select("honest")))
    with pytest.raises(S.Refused) as e:
        ch.select("honest")
    assert _codes(e) == {"coin"} and "already open" in e.value.verdict.detail["coin"]
    case = CC.case
    ch = disagrees("answer", lambda m: m.setattr(OR.CC, "case", lambda j: {**case(j), "reason": "forced"}),
                   lambda ch: _through(ch, rec))
    with pytest.raises(S.Refused) as e:
        ch.outcome("honest", [_unit(t) for t in ch.draws["honest"].units])
    assert _codes(e) == {"final"}
    t = ch.draws["honest"].units[0]
    monkeypatch.setattr(OR, "py_opening", lambda x: "forced")
    record = OR.Record()
    assert SC.check(params, rec, ch.draws["honest"].draw, t, _unit_pub(advice, priv, t), SC.opening(advice, priv, params, t),
                    record=record) == f"{OR.DISAGREE}: opening"
    assert record.witness[0]["python"] == "forced" and record.witness[0]["lean"] is None
    record.close()
    monkeypatch.undo()
    op, served_domain = SC.opening(advice, priv, params, t), SC.window_domain
    monkeypatch.setattr(SC, "window_domain", lambda params, ell: served_domain({**params, "context": "other"}, ell))
    record = OR.Record()
    assert SC.check(params, rec, ch.draws["honest"].draw, t, _unit_pub(advice, priv, t), op, record=record) == \
        f"{OR.DISAGREE}: prelude"
    assert [(e["check"], e["same"]) for e in record.witness] == [("opening", True), ("prelude", False)]
    record.close()


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


@LEAN
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
    lean.write_text(f"#!{sys.executable}\nimport json, sys\nfrom verity.core.protocols.pouw.core_check import stream_subset\n"
                    "a = sys.argv\np, k = int(a[a.index('--population') + 1]), int(a[a.index('--k') + 1])\n"
                    "w = (p.bit_length() + 64 + 7) // 8\nsrc = bytes.fromhex(a[a.index('--stream') + 1])\n"
                    "sys.exit(1) if len(src) != 2 * k * w + 64 else None\n"
                    "print('DRAW ' + json.dumps({'law': 'subset', 'population': p, 'k': k, 'units': stream_subset(k, p, src)}))\n")
    lean.chmod(lean.stat().st_mode | stat.S_IEXEC)
    rec = json.loads((tmp_path / "registration.json").read_text())
    dry = _challenger(params, coins=_coins())        # the process's coins: its draw is the one the job opens
    _held(dry, "honest", rec)
    drawn = dry.select("honest").units
    rows = []
    for u in drawn:
        (tmp_path / f"pub{u}.bin").write_bytes(_unit_pub(advice, priv, u))
        (tmp_path / f"op{u}.json").write_text(json.dumps(SC.opening(advice, priv, params, u)))
        assert SC.check(params, rec, {"units": drawn}, u, _unit_pub(advice, priv, u), SC.opening(advice, priv, params, u)) is None
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
    witness = [json.loads(x) for x in (log / "witness.jsonl").read_text().splitlines()]
    assert [w["reply"] for w in witness] == [x.split(" ", 1)[0] for x in replies]
    noted = [e for w in witness for e in w["of_record"]]
    assert {e["check"] for e in noted} >= {"registered", "registration", "pass", "draw", "answer"}
    assert all(e["same"] and e["of_record"] == "lean" for e in noted)
    accepted = [e for w in witness if w["reply"] == "ACCEPT" for e in w["of_record"]]
    assert accepted and all(e in o["of_record"] for e in accepted)

    assert SC.main(["replay", p, str(log), f"{t}/replay", "--lean", str(lean)]) == 0
    assert capsys.readouterr().out.startswith("SAME ")
    assert (tmp_path / "replay").is_dir() and not SC.replay(params, log, tmp_path / "replay2", SC.python_sampler)  # the twin
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


@pytest.mark.slow
def test_a_staged_unit_opens_under_the_window_root(served, tmp_path):
    """stage_tile.py stages a call's tile and a filler tile of a committed window as `Pc8TileSide` units, each in its own
    process, on the committer's side rows and salts: each public file's rows are the side trees' leaves, so the opening
    accepts.  Nothing cheaper catches a unit whose committed rows are not the side rows (the index field dropped or moved),
    which only the staged public file shows; about a minute at K = 128."""
    *_, words = served
    params, public, advice, priv = _window(words)
    rec = _register(params, public)
    (tmp_path / "advice.json").write_text(json.dumps(advice))
    (tmp_path / "params.json").write_text(json.dumps(params))
    SC.save(tmp_path / "prover.npz", priv)
    stage = Path(__file__).resolve().parents[1] / "served_zk" / "stage_tile.py"
    draw = {"law": "subset", "population": 26, "k": 2, "units": [7, 25]}
    runs = {t: subprocess.Popen([sys.executable, str(stage), str(tmp_path), str(tmp_path / "params.json"), str(t),
                                 str(tmp_path / f"u{t}")], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            for t in draw["units"]}
    for t, p in runs.items():
        _, err = p.communicate()
        assert p.returncode == 0, f"unit {t}: {err[-2000:]}"
        out = tmp_path / f"u{t}"
        st = json.loads((out / "stage.json").read_text())
        assert st["definition"] == SC.template(K).id and st["row_schemas"][:2] == [
            [L.ROW_SCHEMA["fp8-pi", SC.SIDES[s]], 16 * L.ROW_WORDS["fp8-pi"](K)] for s in "ab"]
        op = json.loads((out / "opening.json").read_text())
        assert SC.check(params, rec, draw, t, (out / "pub.bin").read_bytes(), op) is None, t
    assert json.loads((tmp_path / "u25" / "stage.json").read_text())["kind"] == "filler"
